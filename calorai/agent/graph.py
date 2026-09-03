r"""The LangGraph agent.

    ingest -> load_context ---> agent <-> tools -> (end)
           \-> vision_extract -/

* ingest         - classify text / image / image+text
* load_context   - today's totals, the most recent meal, the memory profile card
* vision_extract - (only when there's a photo) a separate vision model pulls out
                   food items + a confidence per item; runs in parallel with
                   load_context
* agent          - the text model with tools bound; decides to call a tool or reply
* tools          - run the tool calls, loop back to agent (capped by AGENT_MAX_LOOPS)

State is persisted per thread by a SQLite checkpointer, so a conversation
survives a restart and each user's thread is isolated.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
from functools import lru_cache

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from calorai.agent.context import get_context
from calorai.agent.memory import MEMORY_TOOLS, render_profile_card
from calorai.agent.prompts import build_system_prompt
from calorai.agent.recovery import looks_like_leaked_tool_call, recover_tool_calls, strip_leaked_markup
from calorai.agent.state import AgentState
from calorai.agent.tools import LOGGING_TOOLS
from calorai.config import CHECKPOINT_PATH, ensure_data_dir, get_settings
from calorai.db import repositories as repo
from calorai.models.gateway import get_text_model
from calorai.vision.extract import extract_food_from_image

logger = logging.getLogger("calorai.graph")

AGENT_TOOLS = LOGGING_TOOLS + MEMORY_TOOLS


def _ingest(state: AgentState) -> dict:
    has_image = bool(state.get("image_path"))
    text = state.get("user_text", "")
    if has_image and text:
        input_type = "image+text"
    elif has_image:
        input_type = "image"
    else:
        input_type = "text"
    # Always clear the previous turn's vision result; the image branch re-populates it.
    return {"input_type": input_type, "awaiting_user": False, "vision_result": None}


def _load_context(state: AgentState) -> dict:
    """Best-effort context load. A DB hiccup here should not kill the turn - but
    it must not pass silently either: each failure is logged, and the parts that
    failed are recorded so the agent can hedge instead of stating wrong totals."""
    ctx = get_context()
    out: dict = {"today_totals": None, "last_meal": None, "memory_card": "",
                 "context_degraded": None, "user_name": None}
    degraded: list[str] = []

    try:
        user = repo.get_user(ctx.user_id)
        if user and user.name and user.name.lower() != "guest":
            out["user_name"] = user.name
    except Exception:
        logger.warning("load_context: user row failed for %s", ctx.user_id, exc_info=True)

    try:
        totals = repo.daily_totals(ctx.user_id, ctx.local_date).rounded()
        out["today_totals"] = {
            "kcal": totals.kcal,
            "protein_g": totals.protein_g,
            "carbs_g": totals.carbs_g,
            "fat_g": totals.fat_g,
            "meal_count": totals.meal_count,
        }
    except Exception:
        logger.warning("load_context: daily totals failed for user %s", ctx.user_id, exc_info=True)
        degraded.append("today's totals")

    try:
        recent = repo.get_recent_meals(ctx.user_id, limit=1)
        if recent:
            m = recent[0]
            out["last_meal"] = {
                "meal_id": m.id,
                "description": m.description,
                "meal_date": m.meal_date,
                "items": [{"name": i.name, "quantity": i.quantity} for i in m.items],
            }
    except Exception:
        logger.warning("load_context: recent meal failed for user %s", ctx.user_id, exc_info=True)
        degraded.append("your recent meals")

    try:
        out["memory_card"] = render_profile_card(ctx.user_id)
    except Exception:
        logger.warning("load_context: profile card failed for user %s", ctx.user_id, exc_info=True)
        degraded.append("your saved profile")

    if degraded:
        out["context_degraded"] = degraded
    return out


def _vision_extract(state: AgentState) -> dict:
    image_path = state.get("image_path")
    if not image_path:
        return {}
    result = extract_food_from_image(image_path, caption=state.get("user_text") or None)
    return {"vision_result": result.to_context()}


def _after_ingest(state: AgentState) -> list[str]:
    if state.get("input_type") in ("image", "image+text"):
        return ["load_context", "vision_extract"]
    return ["load_context"]


_HISTORY_TURNS = 8  # user turns kept in the prompt; older context lives in memory + DB


def _tool_rounds_this_turn(messages: list) -> int:
    """Tool results since the last user message - i.e. the agent<->tools loop count
    for the CURRENT turn only. Counting the whole checkpointed history would
    permanently unbind tools after a few turns."""
    last_human = -1
    for i, m in enumerate(messages):
        if m.type == "human":
            last_human = i
    return sum(1 for m in messages[last_human + 1:] if m.type == "tool")


def _recent_window(messages: list) -> list:
    """Keep the last few user turns (and everything after each), so the prompt
    doesn't grow without bound. Slices only at user-message boundaries, so
    tool-call / tool-result pairs are never split."""
    human_idxs = [i for i, m in enumerate(messages) if m.type == "human"]
    if len(human_idxs) <= _HISTORY_TURNS:
        return messages
    return messages[human_idxs[-_HISTORY_TURNS]:]


def _agent(state: AgentState) -> dict:
    system = build_system_prompt(
        user_name=state.get("user_name"),
        memory_card=state.get("memory_card", ""),
        today_totals=state.get("today_totals"),
        last_meal=state.get("last_meal"),
        vision_result=state.get("vision_result"),
        context_degraded=state.get("context_degraded"),
    )

    model = get_text_model(streaming=True)
    if _tool_rounds_this_turn(state["messages"]) < get_settings().agent_max_loops:
        model = model.bind_tools(AGENT_TOOLS)
    else:
        # Loop cap for THIS turn reached: force a plain-text answer.
        system += (
            "\n\n(You have already used several tools on this message. Reply to the user now in "
            "plain text. If the recent tool results were errors, say plainly that something went "
            "wrong and nothing was changed - do NOT claim a change succeeded.)"
        )

    messages = [SystemMessage(content=system), *_recent_window(state["messages"])]
    try:
        response = model.invoke(messages)
    except Exception:
        # Model/API failure - keep the graph (and the conversation) alive.
        return {
            "messages": [
                AIMessage(
                    content="Sorry, I hit a snag reaching my brain just now. "
                    "Nothing you've logged is lost - try that again in a moment."
                )
            ]
        }

    text = response.content if isinstance(response.content, str) else ""

    # GLM-5.2 quirk 1: a tool call written into the message text instead of
    # returned structured. Parse it back so the tool actually runs.
    if not response.tool_calls and looks_like_leaked_tool_call(text):
        recovered = recover_tool_calls(text)
        if recovered:
            return {"messages": [AIMessage(content=strip_leaked_markup(text), tool_calls=recovered)]}

    # GLM-5.2 quirk 2: a structured tool call with a null id, which the tool node
    # cannot build a ToolMessage for. Give every call a real id.
    if response.tool_calls and any(not tc.get("id") for tc in response.tool_calls):
        fixed = [{**tc, "id": tc.get("id") or f"call_{secrets.token_hex(4)}"} for tc in response.tool_calls]
        return {"messages": [AIMessage(content=response.content, tool_calls=fixed)]}

    return {"messages": [response]}


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


@lru_cache(maxsize=1)
def _checkpointer() -> SqliteSaver:
    ensure_data_dir()
    # Its own file, separate from the application tables, so LangGraph's per-node
    # state writes never lock out a meal or memory write (and vice versa).
    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return SqliteSaver(conn)


@lru_cache(maxsize=1)
def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("ingest", _ingest)
    graph.add_node("load_context", _load_context)
    graph.add_node("vision_extract", _vision_extract)
    graph.add_node("agent", _agent)
    graph.add_node("tools", ToolNode(AGENT_TOOLS))

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", _after_ingest, ["load_context", "vision_extract"])
    graph.add_edge("load_context", "agent")
    graph.add_edge("vision_extract", "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=_checkpointer())
