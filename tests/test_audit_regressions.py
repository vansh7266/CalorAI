"""Regression tests for the issues found in the external audit."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from calorai.agent import context as ctxmod
from calorai.agent import graph as g
from calorai.agent.context import TurnContext
from calorai.agent.tools import get_daily_totals, log_meal, update_meal
from calorai.db import database, repositories as repo


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)
    user = repo.create_user("A", "UTC")
    token = ctxmod.set_context(
        TurnContext(user.id, "turn", "UTC", datetime(2026, 9, 3, 13, 0, tzinfo=timezone.utc))
    )
    yield user
    ctxmod.reset_context(token)


# --- P0-1: per-turn tool budget, not lifetime -------------------------------

def _state(n_prior_human: int, n_prior_tool: int) -> dict:
    msgs: list = []
    for i in range(n_prior_human):
        msgs.append(HumanMessage(content=f"m{i}"))
        msgs.append(AIMessage(content="", tool_calls=[{"name": "get_daily_totals", "args": {}, "id": f"c{i}", "type": "tool_call"}]))
        msgs.append(ToolMessage(content="{}", tool_call_id=f"c{i}"))
    msgs.append(HumanMessage(content="had 2 rotis"))
    for j in range(n_prior_tool):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "log_meal", "args": {}, "id": f"t{j}", "type": "tool_call"}]))
        msgs.append(ToolMessage(content="{}", tool_call_id=f"t{j}"))
    return {"messages": msgs}


def test_tool_budget_is_per_turn(monkeypatch):
    captured: dict = {}

    class Spy:
        def bind_tools(self, tools):
            captured["tools"] = len(tools)
            return self

        def invoke(self, _m):
            return AIMessage(content="ok")

    monkeypatch.setattr(g, "get_text_model", lambda **k: Spy())

    # 10 prior turns each used a tool, but THIS turn has used none -> tools bound
    g._agent(_state(n_prior_human=10, n_prior_tool=0))
    assert captured.get("tools", 0) > 0

    # this turn has already looped 4 times -> tools withheld
    captured.clear()
    g._agent(_state(n_prior_human=2, n_prior_tool=4))
    assert "tools" not in captured


def test_history_window_keeps_recent_turns_only():
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(content=f"turn {i}"))
        msgs.append(AIMessage(content=f"reply {i}"))
    window = g._recent_window(msgs)
    assert window[0].content == "turn 12"  # last 8 human turns
    assert len(window) == 16


# --- P0-2: vision result cleared each turn ----------------------------------

def test_ingest_clears_vision_result():
    assert g._ingest({"image_path": None, "user_text": "hi"})["vision_result"] is None
    assert g._ingest({"image_path": "x.jpg", "user_text": ""})["vision_result"] is None


# --- P1: multi-change corrections are atomic --------------------------------

def test_partial_correction_rolls_back(env):
    res = log_meal.invoke({"items": [{"name": "roti", "quantity": 2, "unit": "piece"}]})
    meal_id = res["meal_id"]
    out = update_meal.invoke({"meal_id": meal_id, "item_name": "roti", "new_quantity": 5, "remove_item": "ghost"})
    assert "error" in out
    assert repo.get_meal(meal_id).items[0].quantity == 2  # unchanged - nothing committed


# --- P1: invalid quantity rejected -----------------------------------------

def test_negative_quantity_rejected(env):
    # rejected at the schema boundary before anything is written
    with pytest.raises(Exception):
        log_meal.invoke({"items": [{"name": "roti", "quantity": -2, "unit": "piece"}]})
    assert get_daily_totals.invoke({})["totals"]["kcal"] == 0
    # update_meal validates its own numeric arg and returns a clean error
    res = log_meal.invoke({"items": [{"name": "roti", "quantity": 1, "unit": "piece"}]})
    bad = update_meal.invoke({"meal_id": res["meal_id"], "item_name": "roti", "new_quantity": -1})
    assert "error" in bad
    assert get_daily_totals.invoke({})["totals"]["kcal"] == 105  # unchanged


# --- P1: emptying a meal removes it from the count -------------------------

def test_removing_last_item_soft_deletes_meal(env):
    res = log_meal.invoke({"items": [{"name": "chai", "quantity": 1, "unit": "cup"}]})
    update_meal.invoke({"meal_id": res["meal_id"], "remove_item": "chai"})
    totals = get_daily_totals.invoke({})["totals"]
    assert totals["kcal"] == 0 and totals["meal_count"] == 0


# --- P1: description stays consistent with items --------------------------

def test_description_regenerated_on_correction(env):
    res = log_meal.invoke({"items": [{"name": "roti", "quantity": 2, "unit": "piece"}]})
    update_meal.invoke({"meal_id": res["meal_id"], "item_name": "roti", "new_quantity": 4})
    assert "4" in repo.get_meal(res["meal_id"]).description
    assert repo.get_meal(res["meal_id"]).description != "2 roti"


# --- meal_type + date validation -----------------------------------------

def test_bad_meal_type_rejected(env):
    res = log_meal.invoke({"items": [{"name": "roti", "quantity": 1, "unit": "piece"}]})
    out = update_meal.invoke({"meal_id": res["meal_id"], "meal_type": "midnight feast"})
    assert "error" in out


def test_impossible_iso_date_not_stored(env):
    out = log_meal.invoke({"items": [{"name": "roti", "quantity": 1, "unit": "piece"}], "eaten_when": "2026-99-99"})
    assert out.get("logged") is True
    assert out["meal_date"] == "2026-09-03"  # fell back to today, not the junk date


# --- image source provenance --------------------------------------------

def test_source_from_context_not_model(env, monkeypatch):
    import calorai.agent.tools as tools_mod

    ctx = TurnContext(env.id, "t", "UTC", datetime(2026, 9, 3, 13, tzinfo=timezone.utc), source="image")
    monkeypatch.setattr(tools_mod, "get_context", lambda: ctx)
    res = log_meal.invoke({"items": [{"name": "dal", "quantity": 1, "unit": "bowl"}]})
    assert repo.get_meal(res["meal_id"]).source == "image"  # not "text" from a tool arg
