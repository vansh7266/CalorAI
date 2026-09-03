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
from calorai.config import DB_PATH, ensure_data_dir, get_settings
from calorai.db import repositories as repo
from calorai.models.gateway import get_text_model
from calorai.vision.extract import extract_food_from_image

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
    return {"input_type": input_type, "awaiting_user": False}


def _load_context(state: AgentState) -> dict:
    """Best-effort context load. A DB hiccup here should not kill the turn - the
    agent can still answer, just with less context."""
    ctx = get_context()
    out: dict = {"today_totals": None, "last_meal": None, "memory_card": ""}

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
        pass

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
        pass

    try:
        out["memory_card"] = render_profile_card(ctx.user_id)
    except Exception:
        pass

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


def _agent(state: AgentState) -> dict:
    system = build_system_prompt(
        memory_card=state.get("memory_card", ""),
        today_totals=state.get("today_totals"),
        last_meal=state.get("last_meal"),
        vision_result=state.get("vision_result"),
    )

    model = get_text_model(streaming=True)
    tool_rounds = sum(1 for m in state["messages"] if m.type == "tool")
    if tool_rounds < get_settings().agent_max_loops:
        model = model.bind_tools(AGENT_TOOLS)
    else:
        # Loop cap reached: force a plain-text answer instead of another tool call.
        system += "\n\n(You have already used several tools this turn. Reply to the user now in plain text.)"

    messages = [SystemMessage(content=system), *state["messages"]]
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
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
