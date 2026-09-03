"""Error handling and isolation: the agent should degrade, not crash."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from calorai.agent import context as ctxmod
from calorai.agent import graph as graphmod
from calorai.agent import runner as runnermod
from calorai.agent.context import TurnContext
from calorai.agent.graph import _agent, _load_context
from calorai.agent.tools import get_daily_totals, log_meal
from calorai.db import database, repositories as repo


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)
    yield


def _ctx(user_id: str):
    return TurnContext(user_id, "turn_x", "UTC", datetime(2026, 9, 3, 13, 0, tzinfo=timezone.utc))


def test_agent_node_survives_model_failure(monkeypatch):
    class _Boom:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            raise RuntimeError("api down")

    monkeypatch.setattr(graphmod, "get_text_model", lambda **k: _Boom())
    user = repo.create_user("t", "UTC")
    token = ctxmod.set_context(_ctx(user.id))
    try:
        out = _agent({"messages": [HumanMessage(content="had 2 rotis")]})
    finally:
        ctxmod.reset_context(token)
    msg = out["messages"][0]
    assert isinstance(msg, AIMessage) and "snag" in msg.content.lower()


def test_load_context_survives_db_failure(monkeypatch):
    user = repo.create_user("t", "UTC")
    token = ctxmod.set_context(_ctx(user.id))
    try:
        monkeypatch.setattr(graphmod.repo, "daily_totals", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
        monkeypatch.setattr(graphmod.repo, "get_recent_meals", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
        out = _load_context({})
    finally:
        ctxmod.reset_context(token)
    # the turn survives (no exception) ...
    assert out["today_totals"] is None and out["last_meal"] is None
    # ... but the failure is surfaced, not swallowed, so the agent can hedge
    assert out["context_degraded"] and "today's totals" in out["context_degraded"]


def test_load_context_clean_when_healthy(monkeypatch):
    user = repo.create_user("t", "UTC")
    token = ctxmod.set_context(_ctx(user.id))
    try:
        out = _load_context({})
    finally:
        ctxmod.reset_context(token)
    assert out["context_degraded"] is None


def test_run_turn_never_raises(monkeypatch):
    monkeypatch.setattr(runnermod, "build_app", lambda: (_ for _ in ()).throw(RuntimeError("graph broke")))
    reply = runnermod.run_turn("hi", user_id="usr_x", thread_id="usr_x")
    assert "something went wrong" in reply.lower()
    assert "safe" in reply.lower()


def test_multi_user_isolation():
    alice = repo.create_user("alice", "UTC")
    bob = repo.create_user("bob", "UTC")

    ta = ctxmod.set_context(_ctx(alice.id))
    log_meal.invoke({"items": [{"name": "roti", "quantity": 4, "unit": "piece"}]})
    a_total = get_daily_totals.invoke({})["totals"]["kcal"]
    ctxmod.reset_context(ta)

    tb = ctxmod.set_context(_ctx(bob.id))
    log_meal.invoke({"items": [{"name": "chai", "quantity": 1, "unit": "cup"}]})
    b_total = get_daily_totals.invoke({})["totals"]["kcal"]
    ctxmod.reset_context(tb)

    assert a_total == 420        # 4 roti, bob's chai not counted
    assert b_total == 90         # just the chai
