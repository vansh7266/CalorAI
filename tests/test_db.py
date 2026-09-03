"""Database layer tests. Each test runs against a fresh temp database."""

from __future__ import annotations

import importlib

import pytest

from calorai.db import database, repositories as repo


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)
    # repositories calls get_connection() with no arg -> uses patched DB_PATH
    yield


def _roti_item(qty: float) -> dict:
    return {
        "name": "roti",
        "quantity": qty,
        "unit": "piece",
        "kcal_per_unit": 100,
        "protein_g_per_unit": 3,
        "carbs_g_per_unit": 18,
        "fat_g_per_unit": 2,
        "nutrition_source": "seed",
    }


def test_create_user_and_fetch():
    user = repo.create_user("Vansh", "Asia/Kolkata")
    assert user.id.startswith("usr_")
    again = repo.get_user(user.id)
    assert again is not None and again.name == "Vansh" and again.timezone == "Asia/Kolkata"


def test_insert_meal_and_daily_totals():
    user = repo.create_user("t")
    repo.insert_meal(
        user_id=user.id,
        eaten_at="2026-09-03T08:00:00+00:00",
        meal_date="2026-09-03",
        meal_type="breakfast",
        description="2 rotis",
        source="text",
        items=[_roti_item(2)],
    )
    totals = repo.daily_totals(user.id, "2026-09-03").rounded()
    assert totals.kcal == 200
    assert totals.protein_g == 6
    assert totals.meal_count == 1


def test_correction_updates_not_doubles():
    user = repo.create_user("t")
    meal = repo.insert_meal(
        user_id=user.id,
        eaten_at="2026-09-03T08:00:00+00:00",
        meal_date="2026-09-03",
        meal_type="breakfast",
        description="2 rotis",
        source="text",
        items=[_roti_item(2)],
    )
    item_id = meal.items[0].id

    repo.set_item_quantity(item_id, 3, turn_id="turn-2")

    refreshed = repo.get_meal(meal.id)
    assert len(refreshed.items) == 1  # still one row, not two
    assert refreshed.items[0].quantity == 3

    totals = repo.daily_totals(user.id, "2026-09-03").rounded()
    assert totals.kcal == 300  # 3 * 100, not 2*100 + 3*100

    edits = repo.get_meal_edits(meal.id)
    assert any(e["field"] == "item:roti:quantity" and e["new_value"] == "3" for e in edits)


def test_soft_delete_excludes_from_totals():
    user = repo.create_user("t")
    meal = repo.insert_meal(
        user_id=user.id,
        eaten_at="2026-09-03T08:00:00+00:00",
        meal_date="2026-09-03",
        meal_type="lunch",
        description="rice",
        source="text",
        items=[_roti_item(5)],
    )
    assert repo.daily_totals(user.id, "2026-09-03").kcal == 500

    assert repo.soft_delete_meal(meal.id) is True
    assert repo.daily_totals(user.id, "2026-09-03").kcal == 0
    assert repo.soft_delete_meal(meal.id) is False  # already deleted

    # still retrievable with include_deleted
    deleted = repo.get_meals_for_date(user.id, "2026-09-03", include_deleted=True)
    assert len(deleted) == 1 and deleted[0].status == "deleted"


def test_memory_upsert_supersedes():
    user = repo.create_user("t")
    repo.upsert_memory(user_id=user.id, type="goal", key="protein_target", content="140g protein/day",
                       structured_value={"protein_g": 140})
    repo.upsert_memory(user_id=user.id, type="goal", key="protein_target", content="160g protein/day",
                       structured_value={"protein_g": 160})

    active = repo.get_active_memory(user.id, types=["goal"])
    assert len(active) == 1
    assert active[0].content == "160g protein/day"
    assert active[0].structured_value == {"protein_g": 160}


def test_memory_search_and_deactivate():
    user = repo.create_user("t")
    mem = repo.upsert_memory(user_id=user.id, type="diet", key="diet", content="vegetarian, no eggs")
    hits = repo.search_memory(user.id, "vegetarian")
    assert len(hits) == 1 and hits[0].id == mem.id

    assert repo.deactivate_memory(mem.id) is True
    assert repo.search_memory(user.id, "vegetarian") == []


def test_item_order_is_stable():
    user = repo.create_user("t")
    names = ["paratha", "chai", "curd", "pickle"]
    meal = repo.insert_meal(
        user_id=user.id,
        eaten_at="2026-09-03T08:00:00+00:00",
        meal_date="2026-09-03",
        meal_type="breakfast",
        description="breakfast plate",
        source="text",
        items=[{**_roti_item(1), "name": n} for n in names],
    )
    assert [i.name for i in meal.items] == names
    assert [i.name for i in repo.get_meal(meal.id).items] == names


def test_add_and_remove_item():
    user = repo.create_user("t")
    meal = repo.insert_meal(
        user_id=user.id,
        eaten_at="2026-09-03T20:00:00+00:00",
        meal_date="2026-09-03",
        meal_type="dinner",
        description="dinner",
        source="text",
        items=[_roti_item(2)],
    )
    lassi = {**_roti_item(1), "name": "lassi", "kcal_per_unit": 150}
    new_item_id = repo.add_item_to_meal(meal.id, lassi, turn_id="t3")
    assert new_item_id is not None

    meal = repo.get_meal(meal.id)
    assert [i.name for i in meal.items] == ["roti", "lassi"]
    assert repo.daily_totals(user.id, "2026-09-03").rounded().kcal == 350  # 200 + 150

    assert repo.remove_item(new_item_id, turn_id="t4") is True
    assert repo.daily_totals(user.id, "2026-09-03").rounded().kcal == 200


def test_nutrition_cache_roundtrip():
    assert repo.get_cached_nutrition("paneer paratha") is None
    repo.put_cached_nutrition(
        "Paneer Paratha",
        {"kcal_per_unit": 280, "protein_g_per_unit": 9, "carbs_g_per_unit": 30, "fat_g_per_unit": 14, "unit": "piece"},
        source="model",
    )
    cached = repo.get_cached_nutrition("paneer paratha")
    assert cached is not None and cached["kcal_per_unit"] == 280 and cached["source"] == "model"
