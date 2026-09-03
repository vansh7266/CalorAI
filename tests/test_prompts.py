"""System-prompt assembly tests."""

from __future__ import annotations

from calorai.agent.prompts import build_system_prompt


def test_bare_prompt_has_instructions_no_context_block():
    text = build_system_prompt()
    assert "You are CalorAI" in text
    assert "DECIDING WHETHER TO LOG OR ASK" in text
    assert "--- current context ---" not in text


def test_today_totals_render():
    text = build_system_prompt(
        today_totals={"kcal": 630, "protein_g": 14.5, "carbs_g": 90.0, "fat_g": 25.0, "meal_count": 1}
    )
    assert "--- current context ---" in text
    assert "Today so far: 630 kcal, 14.5 g protein" in text


def test_last_meal_render():
    text = build_system_prompt(
        last_meal={
            "meal_id": "meal_abc",
            "description": "3 paratha, chai",
            "items": [{"name": "plain paratha", "quantity": 3}, {"name": "chai", "quantity": 1}],
        }
    )
    assert "meal_abc" in text
    assert "3x plain paratha" in text


def test_memory_card_render():
    text = build_system_prompt(memory_card="- vegetarian\n- protein target 140 g/day")
    assert "What you know about this user" in text
    assert "protein target 140 g/day" in text


def test_user_name_render():
    assert "The user's name is Vineet." in build_system_prompt(user_name="Vineet")
    assert "The user's name is" not in build_system_prompt(user_name=None)
