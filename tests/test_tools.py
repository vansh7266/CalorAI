"""Agent tool tests. Real DB, real seed nutrition (no model calls - every food
used here is in the seed table)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from calorai.agent import context as ctxmod
from calorai.agent.context import TurnContext
from calorai.agent.tools import (
    delete_meal,
    get_daily_totals,
    get_meals,
    log_meal,
    lookup_nutrition,
    update_meal,
)
from calorai.db import database, repositories as repo


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)

    user = repo.create_user("Tester", "UTC")
    token = ctxmod.set_context(
        TurnContext(
            user_id=user.id,
            turn_id="turn_test",
            timezone_name="UTC",
            now_utc=datetime(2026, 9, 3, 13, 0, tzinfo=timezone.utc),  # 1pm -> lunch
        )
    )
    yield user
    ctxmod.reset_context(token)


def _log(items, **kw):
    return log_meal.invoke({"items": items, **kw})


def test_log_meal_and_totals():
    res = _log([{"name": "roti", "quantity": 2, "unit": "piece"}, {"name": "dal", "quantity": 1, "unit": "bowl"}])
    assert res["logged"] is True
    assert res["meal_type"] == "lunch"  # inferred from 1pm
    assert res["kcal"] == 210 + 180
    assert res["today"]["kcal"] == 390

    totals = get_daily_totals.invoke({})
    assert totals["totals"]["kcal"] == 390


def test_correction_updates_not_doubles():
    res = _log([{"name": "roti", "quantity": 2, "unit": "piece"}])
    meal_id = res["meal_id"]
    assert res["today"]["kcal"] == 210

    upd = update_meal.invoke({"meal_id": meal_id, "item_name": "roti", "new_quantity": 3})
    assert upd["updated"] is True
    assert "roti quantity -> 3.0" in upd["changes"]
    assert len(upd["items"]) == 1          # still one row
    assert upd["today"]["kcal"] == 315     # 3 * 105, not 525


def test_update_remove_and_add_item():
    res = _log([{"name": "roti", "quantity": 2, "unit": "piece"}, {"name": "dal", "quantity": 1, "unit": "bowl"}])
    meal_id = res["meal_id"]

    upd = update_meal.invoke({"meal_id": meal_id, "remove_item": "dal"})
    assert [i["name"] for i in upd["items"]] == ["roti"]
    assert upd["today"]["kcal"] == 210

    upd = update_meal.invoke({"meal_id": meal_id, "add_item": {"name": "curd", "quantity": 1, "unit": "bowl"}})
    assert "curd" in [i["name"] for i in upd["items"]]
    assert upd["today"]["kcal"] == 210 + 90


def test_update_unknown_meal_and_item():
    assert "error" in update_meal.invoke({"meal_id": "meal_nope", "item_name": "roti", "new_quantity": 3})

    res = _log([{"name": "roti", "quantity": 1, "unit": "piece"}])
    err = update_meal.invoke({"meal_id": res["meal_id"], "item_name": "biryani", "new_quantity": 2})
    assert "error" in err and "biryani" in err["error"]


def test_delete_meal():
    res = _log([{"name": "roti", "quantity": 5, "unit": "piece"}])
    assert get_daily_totals.invoke({})["totals"]["kcal"] == 525

    out = delete_meal.invoke({"meal_id": res["meal_id"]})
    assert out["deleted"] is True
    assert out["today"]["kcal"] == 0

    again = delete_meal.invoke({"meal_id": res["meal_id"]})
    assert again["deleted"] is False


def test_get_meals_recent_and_by_date():
    _log([{"name": "chai", "quantity": 1, "unit": "cup"}])
    _log([{"name": "roti", "quantity": 2, "unit": "piece"}], eaten_when="yesterday")

    recent = get_meals.invoke({})
    assert recent["count"] == 2

    today = get_meals.invoke({"date": "today"})
    assert today["count"] == 1 and today["meals"][0]["items"][0]["name"] == "chai"

    yday = get_meals.invoke({"date": "yesterday"})
    assert yday["count"] == 1 and yday["meals"][0]["meal_date"] == "2026-09-02"


def test_lookup_nutrition_does_not_log():
    out = lookup_nutrition.invoke({"items": [{"name": "samosa", "quantity": 2, "unit": "piece"}]})
    assert out["items"][0]["per_unit"]["kcal"] == 130
    assert out["items"][0]["for_quantity"]["kcal"] == 260
    assert get_daily_totals.invoke({})["totals"]["meal_count"] == 0
