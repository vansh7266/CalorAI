"""The LangGraph agent.

    ingest -> load_context -> agent <-> tools -> (end)

* ingest        - classify the input, put the user text on the state
* load_context  - pull today's totals and the most recent meal into the state
                  (the memory card is added here in phase 2, vision in phase 3)
* agent         - the text model with tools bound; decides to call a tool or reply
* tools         - run the tool calls, loop back to agent (capped by AGENT_MAX_LOOPS)

State is persisted per thread by a SQLite checkpointer, so a conversation
survives a restart and each user's thread is isolated.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from calorai.agent.context import get_context
from calorai.agent.prompts import build_system_prompt
from calorai.agent.state import AgentState
from calorai.agent.tools import LOGGING_TOOLS
from calorai.config import DB_PATH, ensure_data_dir, get_settings
from calorai.db import repositories as repo
from calorai.models.gateway import get_text_model


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
    ctx = get_context()
    totals = repo.daily_totals(ctx.user_id, ctx.local_date).rounded()
    today = {
        "kcal": totals.kcal,
        "protein_g": totals.protein_g,
        "carbs_g": totals.carbs_g,
        "fat_g": totals.fat_g,
        "meal_count": totals.meal_count,
    }

    last_meal = None
    recent = repo.get_recent_meals(ctx.user_id, limit=1)
    if recent:
        m = recent[0]
        last_meal = {
            "meal_id": m.id,
            "description": m.description,
            "meal_date": m.meal_date,
            "items": [{"name": i.name, "quantity": i.quantity} for i in m.items],
        }

    return {"today_totals": today, "last_meal": last_meal}


def _agent(state: AgentState) -> dict:
    system = build_system_prompt(
        memory_card=state.get("memory_card", ""),
        today_totals=state.get("today_totals"),
        last_meal=state.get("last_meal"),
    )

    model = get_text_model()
    tool_rounds = sum(1 for m in state["messages"] if m.type == "tool")
    if tool_rounds < get_settings().agent_max_loops:
        model = model.bind_tools(LOGGING_TOOLS)
    else:
        # Loop cap reached: force a plain-text answer instead of another tool call.
        system += "\n\n(You have already used several tools this turn. Reply to the user now in plain text.)"

    messages = [SystemMessage(content=system), *state["messages"]]
    return {"messages": [model.invoke(messages)]}


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
    graph.add_node("agent", _agent)
    graph.add_node("tools", ToolNode(LOGGING_TOOLS))

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "load_context")
    graph.add_edge("load_context", "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=_checkpointer())
