"""Memory subsystem tests. The reflection model call is stubbed."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from calorai.agent import context as ctxmod
from calorai.agent import memory
from calorai.agent.context import TurnContext
from calorai.agent.memory import _derive_key, recall_memory, render_profile_card, run_reflection, save_memory
from calorai.agent.tools import log_meal
from calorai.db import database, repositories as repo


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)

    user = repo.create_user("Mem", "UTC")
    token = ctxmod.set_context(
        TurnContext(user.id, "turn_test", "UTC", datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc))
    )
    yield user
    ctxmod.reset_context(token)


def test_derive_key():
    assert _derive_key("diet", "vegetarian", None) == "diet"
    assert _derive_key("goal", "hit 140g protein a day", None) == "protein_target"
    assert _derive_key("goal", "stay under 1800 kcal", None) == "calorie_target"
    assert _derive_key("routine", "eggs and toast", "breakfast") == "usual_breakfast"


def test_profile_card_empty_then_populated():
    assert render_profile_card(ctxmod.get_context().user_id) == ""

    save_memory.invoke({"kind": "diet", "content": "vegetarian, no eggs"})
    save_memory.invoke({"kind": "goal", "content": "targets 140 g protein per day",
                        "structured_value": {"protein_g": 140}})

    card = render_profile_card(ctxmod.get_context().user_id)
    assert "vegetarian, no eggs" in card
    assert "140 g protein" in card


def test_save_memory_supersedes():
    save_memory.invoke({"kind": "goal", "content": "140 g protein", "key": "protein_target"})
    save_memory.invoke({"kind": "goal", "content": "160 g protein", "key": "protein_target"})
    active = repo.get_active_memory(ctxmod.get_context().user_id, types=["goal"])
    assert len(active) == 1 and active[0].content == "160 g protein"


def test_recall_memory_keyword_and_routine():
    save_memory.invoke(
        {
            "kind": "routine",
            "content": "usual breakfast: 2 eggs, 2 toast, black coffee",
            "meal_type": "breakfast",
            "routine_items": [
                {"name": "boiled egg", "quantity": 2, "unit": "egg"},
                {"name": "toast", "quantity": 2, "unit": "slice"},
                {"name": "black coffee", "quantity": 1, "unit": "cup"},
            ],
        }
    )

    out = recall_memory.invoke({"query": "my usual"})
    assert out["found"] >= 1
    routine = next(m for m in out["memories"] if m["kind"] == "routine")
    assert routine["structured_value"]["items"][0]["name"] == "boiled egg"

    # use_count bumped
    row = repo.get_memory(routine["memory_id"])
    assert row.use_count == 1


def test_routine_suggestion_after_three_identical_meals():
    user = ctxmod.get_context().user_id
    for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
        repo.insert_meal(
            user_id=user, eaten_at=f"{day}T08:00:00+00:00", meal_date=day, meal_type="breakfast",
            description="eggs and toast", source="text",
            items=[
                {"name": "boiled egg", "quantity": 2, "unit": "egg", "kcal_per_unit": 78,
                 "protein_g_per_unit": 6, "carbs_g_per_unit": 0.6, "fat_g_per_unit": 5},
                {"name": "toast", "quantity": 2, "unit": "slice", "kcal_per_unit": 80,
                 "protein_g_per_unit": 2.5, "carbs_g_per_unit": 13, "fat_g_per_unit": 1.5},
            ],
        )
    card = render_profile_card(user)
    assert "3+ times" in card and "usual breakfast" in card

    save_memory.invoke({"kind": "routine", "content": "usual breakfast", "meal_type": "breakfast"})
    assert "3+ times" not in render_profile_card(user)


def test_reflection_saves_new_fact(monkeypatch):
    user = ctxmod.get_context().user_id

    class _Stub:
        def invoke(self, _prompt):
            return memory._Reflection(should_save=True, kind="diet", content="vegetarian")

    monkeypatch.setattr(memory, "_run_reflection_model", lambda _p: _Stub().invoke(_p))
    rec = run_reflection(user, "turn_x", "i'm vegetarian btw", "Got it, noted!")
    assert rec is not None and rec.content == "vegetarian"
    assert rec.learned_via == "inferred"  # reflection output is a model inference, not "stated"
    assert any(r.content == "vegetarian" for r in repo.get_active_memory(user, types=["diet"]))


def test_reflection_skips_when_nothing_durable(monkeypatch):
    user = ctxmod.get_context().user_id

    class _Stub:
        def invoke(self, _prompt):
            return memory._Reflection(should_save=False)

    monkeypatch.setattr(memory, "_run_reflection_model", lambda _p: _Stub().invoke(_p))
    assert run_reflection(user, "turn_y", "had some cake", "Logged ~350 cal of cake.") is None
    assert repo.get_active_memory(user) == []


def test_refining_a_routine_consolidates_instead_of_piling_up():
    """The user's real Windows session: 'my usual is toast and milk' then
    'toast and milk is only for breakfast' left three overlapping routine rows.
    It should end as one."""
    user = ctxmod.get_context().user_id

    save_memory.invoke({"kind": "routine", "content": "usual meal is toast and milk"})
    save_memory.invoke(
        {"kind": "routine", "content": "usual breakfast is toast and milk", "meal_type": "breakfast"}
    )
    rows = repo.get_active_memory(user, types=["routine"])
    assert len(rows) == 1
    assert "breakfast" in rows[0].content

    # a vague inference must not displace the stated routine
    from calorai.agent.memory import run_reflection

    class _Stub:
        def invoke(self, _p):
            return memory._Reflection(
                should_save=True, kind="routine",
                content="toast and milk is only for breakfast, not for other times",
            )

    import calorai.agent.memory as m
    _orig = m._run_reflection_model
    m._run_reflection_model = lambda _p: _Stub().invoke(_p)
    try:
        run_reflection(user, "turn_z", "toast and milk is only for breakfast", "Updated!")
    finally:
        m._run_reflection_model = _orig

    rows = repo.get_active_memory(user, types=["routine"])
    assert len(rows) == 1
    assert rows[0].learned_via == "stated"


def test_distinct_routines_are_kept_separate():
    user = ctxmod.get_context().user_id
    save_memory.invoke({"kind": "routine", "content": "usual breakfast is 2 eggs and toast", "meal_type": "breakfast"})
    save_memory.invoke({"kind": "routine", "content": "usual lunch is dal and rice", "meal_type": "lunch"})
    assert len(repo.get_active_memory(user, types=["routine"])) == 2
