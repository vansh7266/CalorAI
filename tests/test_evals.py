"""Tests for the eval grader itself (offline)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from calorai.agent import context as ctxmod
from calorai.agent.context import TurnContext
from calorai.agent.tools import log_meal
from calorai.db import database, repositories as repo
from evals.graders import grade


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)
    yield


def _user_with_meal():
    user = repo.create_user("g", "UTC")
    token = ctxmod.set_context(
        TurnContext(user.id, "t", "UTC", datetime(2026, 9, 3, 13, 0, tzinfo=timezone.utc))
    )
    try:
        log_meal.invoke({"items": [{"name": "roti", "quantity": 3, "unit": "piece"}]})
    finally:
        ctxmod.reset_context(token)
    return user


def test_grader_all_pass():
    user = _user_with_meal()
    res = grade(
        {
            "meals_today": 1,
            "has_items": ["roti"],
            "item_quantity": {"roti": 3},
            "total_kcal_between": [280, 340],
        },
        user,
        "Logged 3 rotis, about 315 cal.",
    )
    assert res.passed


def test_grader_flags_failures():
    user = _user_with_meal()
    res = grade({"meals_today": 2, "item_quantity": {"roti": 5}}, user, "ok")
    assert not res.passed
    failing = [c.name for c in res.checks if not c.passed]
    assert "meals_today == 2" in failing


def test_grader_reply_checks():
    user = _user_with_meal()
    res = grade({"reply_contains_any": ["315", "316"], "reply_asks": True}, user, "You're at 315 cal. Want more detail?")
    assert res.passed

    res = grade({"reply_asks": True}, user, "Logged it.")
    assert not res.passed


def test_grader_memory_check():
    user = _user_with_meal()
    repo.upsert_memory(user_id=user.id, type="diet", key="diet", content="Vegetarian, no eggs")
    assert grade({"memory_has": {"type": "diet", "contains": "veg"}}, user, "").passed
    assert not grade({"memory_has": {"type": "goal", "contains": "veg"}}, user, "").passed
