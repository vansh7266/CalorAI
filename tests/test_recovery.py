"""Recovering tool calls the model wrote as plain text."""

from __future__ import annotations

from calorai.agent.recovery import looks_like_leaked_tool_call, recover_tool_calls, strip_leaked_markup

LEAK = (
    '<tool_call>log_meal<arg_key>items</arg_key>'
    '<arg_value>[{"name": "daal", "quantity": 1, "unit": "bowl"}, '
    '{"name": "rice", "quantity": 1, "unit": "plate"}]</arg_value>'
    '<arg_key>meal_type</arg_key><arg_value>lunch</arg_value></tool_call>'
)


def test_detects_leak():
    assert looks_like_leaked_tool_call(LEAK)
    assert not looks_like_leaked_tool_call("Logged your lunch, about 400 cal.")


def test_recovers_arg_key_value_format():
    calls = recover_tool_calls(LEAK)
    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "log_meal"
    assert call["type"] == "tool_call"
    assert call["args"]["meal_type"] == "lunch"
    assert call["args"]["items"][0]["name"] == "daal"
    assert call["args"]["items"][1]["unit"] == "plate"


def test_recovers_json_body_format():
    leak = '<tool_call>get_daily_totals{"arguments": {"date": "yesterday"}}</tool_call>'
    calls = recover_tool_calls(leak)
    assert calls and calls[0]["name"] == "get_daily_totals"
    assert calls[0]["args"] == {"date": "yesterday"}


def test_strip_leaked_markup():
    text = "Sure! " + LEAK + " done."
    cleaned = strip_leaked_markup(text)
    assert "<tool_call>" not in cleaned and "arg_key" not in cleaned
    assert "Sure!" in cleaned


def test_no_false_positive_recovery():
    assert recover_tool_calls("just a normal reply about 2 rotis") == []


def test_agent_node_normalizes_null_tool_call_id(tmp_path, monkeypatch):
    """GLM-5.2 sometimes returns a structured tool call with id=None."""
    from datetime import datetime, timezone

    from langchain_core.messages import AIMessage, HumanMessage

    from calorai.agent import context as ctxmod
    from calorai.agent import graph as graphmod
    from calorai.agent.context import TurnContext
    from calorai.db import database, repositories as repo

    db_file = tmp_path / "t.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)

    class _NullIdModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_daily_totals", "args": {}, "id": None, "type": "tool_call"}],
            )

    monkeypatch.setattr(graphmod, "get_text_model", lambda **k: _NullIdModel())
    user = repo.create_user("t", "UTC")
    token = ctxmod.set_context(TurnContext(user.id, "x", "UTC", datetime(2026, 9, 3, 12, tzinfo=timezone.utc)))
    try:
        out = graphmod._agent({"messages": [HumanMessage(content="how am I doing")]})
    finally:
        ctxmod.reset_context(token)

    tc = out["messages"][0].tool_calls[0]
    assert tc["id"] and isinstance(tc["id"], str)
