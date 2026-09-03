"""CLI command + onboarding tests (no live model calls)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from rich.console import Console

from calorai.agent import context as ctxmod
from calorai.agent.context import TurnContext
from calorai.agent.tools import log_meal
from calorai.cli import commands, onboarding
from calorai.db import database, repositories as repo


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    monkeypatch.setattr(onboarding, "SESSION_FILE", tmp_path / ".session")
    database.init_db(db_file)
    yield


def _console() -> Console:
    return Console(record=True, width=100)


def _seed_meal(user):
    token = ctxmod.set_context(
        TurnContext(user.id, "t", "UTC", datetime(2026, 9, 3, 13, 0, tzinfo=timezone.utc))
    )
    try:
        log_meal.invoke({"items": [{"name": "roti", "quantity": 2, "unit": "piece"}]})
    finally:
        ctxmod.reset_context(token)


def test_dispatch_help_and_unknown():
    con = _console()
    user = repo.create_user("t", "UTC")
    assert commands.dispatch(con, user, "/help") is True
    assert commands.dispatch(con, user, "/bogus") is True
    out = con.export_text()
    assert "/totals" in out and "unknown command '/bogus'" in out


def test_dispatch_quit_returns_false():
    assert commands.dispatch(_console(), repo.create_user("t", "UTC"), "/quit") is False


def test_totals_and_history_reflect_db():
    user = repo.create_user("t", "UTC")
    _seed_meal(user)

    con = _console()
    commands.dispatch(con, user, "/totals")
    commands.dispatch(con, user, "/history")
    out = con.export_text()
    assert "210 kcal" in out or "210" in out
    assert "roti" in out


def test_memory_and_forget():
    user = repo.create_user("t", "UTC")
    con = _console()
    commands.dispatch(con, user, "/memory")
    assert "haven't remembered" in con.export_text()

    mem = repo.upsert_memory(user_id=user.id, type="diet", key="diet", content="vegetarian")
    con = _console()
    commands.dispatch(con, user, "/memory")
    assert "vegetarian" in con.export_text()

    con = _console()
    commands.dispatch(con, user, f"/forget {mem.id}")
    assert "forgotten" in con.export_text()
    assert repo.get_active_memory(user.id) == []


def test_resolve_user_first_run_then_resume(monkeypatch):
    con = _console()
    monkeypatch.setattr("rich.prompt.Prompt.ask", staticmethod(lambda *a, **k: "Vansh"))

    user = onboarding.resolve_user(con, None)
    assert user.name == "Vansh" and user.id.startswith("usr_")

    # session file now points at this user -> silent resume
    again = onboarding.resolve_user(_console(), None)
    assert again.id == user.id


def test_resolve_user_explicit_unknown_id_falls_through(monkeypatch):
    monkeypatch.setattr("rich.prompt.Prompt.ask", staticmethod(lambda *a, **k: "Guest"))
    con = _console()
    user = onboarding.resolve_user(con, "usr_doesnotexist")
    assert user.name == "Guest"
    assert "No user found" in con.export_text()


def test_detect_timezone_env_override(monkeypatch):
    monkeypatch.setenv("CALORAI_TZ", "Asia/Kolkata")
    assert onboarding.detect_timezone() == "Asia/Kolkata"
