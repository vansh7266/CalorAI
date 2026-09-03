"""Graph structure tests (no live model calls)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from calorai.agent.graph import _route_after_agent, build_app


def test_graph_compiles_with_expected_nodes():
    app = build_app()
    nodes = set(app.get_graph().nodes)
    assert {"ingest", "load_context", "agent", "tools"} <= nodes


def test_route_to_tools_when_tool_calls_present():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "log_meal", "args": {"items": []}, "id": "1", "type": "tool_call"}],
    )
    assert _route_after_agent({"messages": [HumanMessage(content="hi"), msg]}) == "tools"


def test_route_to_end_on_plain_reply():
    msg = AIMessage(content="Logged it!")
    assert _route_after_agent({"messages": [HumanMessage(content="hi"), msg]}) == END
