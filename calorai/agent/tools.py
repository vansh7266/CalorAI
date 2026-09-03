"""The tools the agent calls.

Read/write split: `get_*` and `lookup_nutrition` never mutate; `log_meal`,
`update_meal`, `delete_meal` do. Each tool returns a plain dict and never raises
- an error comes back as ``{"error": "..."}`` so the agent can recover or ask
the user.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from calorai.agent.context import get_context
from calorai.db import repositories as repo
from calorai.db.records import Meal, MealItem
from calorai.nutrition.resolver import normalize_name, resolve, resolve_many


class FoodItem(BaseModel):
    name: str = Field(description="the food, singular and specific: 'aloo paratha', not 'parathas'")
    quantity: float = Field(default=1.0, description="how many units")
    unit: str = Field(default="serving", description="unit for quantity: piece, cup, bowl, glass, slice, plate...")


# --- shared helpers ---


def _totals(user_id: str, date: str) -> dict:
    t = repo.daily_totals(user_id, date).rounded()
    return {"date": date, "kcal": t.kcal, "protein_g": t.protein_g, "carbs_g": t.carbs_g,
            "fat_g": t.fat_g, "meal_count": t.meal_count}


def _item_view(i: MealItem) -> dict:
    return {
        "item_id": i.id,
        "name": i.name,
        "quantity": i.quantity,
        "unit": i.unit,
        "kcal": round(i.kcal),
        "protein_g": round(i.protein_g, 1),
        "nutrition_source": i.nutrition_source,
        "confidence": round(i.confidence, 2),
    }


def _meal_view(m: Meal) -> dict:
    return {
        "meal_id": m.id,
        "meal_date": m.meal_date,
        "meal_type": m.meal_type,
        "description": m.description,
        "kcal": round(m.kcal),
        "protein_g": round(m.protein_g, 1),
        "items": [_item_view(i) for i in m.items],
    }


def _describe(items: list[FoodItem]) -> str:
    parts = []
    for it in items:
        q = int(it.quantity) if float(it.quantity).is_integer() else it.quantity
        parts.append(f"{q} {it.name}" if q != 1 else it.name)
    return ", ".join(parts)


def _resolved_rows(items: list[FoodItem]) -> tuple[list[dict], list[str]]:
    estimates = resolve_many([(it.name, it.unit) for it in items])
    rows, unresolved = [], []
    for it, est in zip(items, estimates):
        if est.source == "failed":
            unresolved.append(it.name)
        rows.append(
            {
                "name": normalize_name(it.name),
                "quantity": float(it.quantity),
                "unit": it.unit or est.unit,
                "confidence": est.confidence,
                **est.as_item_fields(),
            }
        )
    return rows, unresolved


def _eaten_at_for(meal_date: str) -> str:
    ctx = get_context()
    if meal_date == ctx.local_date:
        return ctx.now_utc.isoformat(timespec="seconds")
    return f"{meal_date}T12:00:00+00:00"


def _match_item(meal: Meal, name: str) -> MealItem | str | None:
    """Return the one item matching `name`, 'AMBIGUOUS' if several match, or None."""
    target = normalize_name(name)
    exact = [i for i in meal.items if normalize_name(i.name) == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return "AMBIGUOUS"
    partial = [i for i in meal.items if target in normalize_name(i.name) or normalize_name(i.name) in target]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        return "AMBIGUOUS"
    return None


def _owned_meal(meal_id: str) -> Meal | None:
    ctx = get_context()
    meal = repo.get_meal(meal_id)
    if meal is None or meal.user_id != ctx.user_id:
        return None
    return meal


# --- write tools ---


@tool
def log_meal(
    items: list[FoodItem],
    meal_type: str | None = None,
    eaten_when: str | None = None,
    note: str | None = None,
) -> dict:
    """Log a new meal the user just told you about.

    items: the foods and how much of each.
    meal_type: breakfast | lunch | dinner | snack. Omit to infer from the time of day.
    eaten_when: 'today' (default), 'yesterday', a weekday, or YYYY-MM-DD.
    note: a short free-text note if the message had context worth keeping.

    Use this only for a NEW meal. To change something already logged, use update_meal.
    """
    try:
        if not items:
            return {"error": "no items to log"}
        ctx = get_context()
        meal_date = ctx.resolve_date(eaten_when)
        mtype = meal_type or ctx.meal_type_for_now()
        rows, unresolved = _resolved_rows(items)
        meal = repo.insert_meal(
            user_id=ctx.user_id,
            eaten_at=_eaten_at_for(meal_date),
            meal_date=meal_date,
            meal_type=mtype,
            description=note or _describe(items),
            source="text",
            items=rows,
        )
        result = {"logged": True, **_meal_view(meal), "today": _totals(ctx.user_id, ctx.local_date)}
        if unresolved:
            result["could_not_estimate_calories_for"] = unresolved
        return result
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"could not log the meal: {exc}"}


@tool
def update_meal(
    meal_id: str,
    item_name: str | None = None,
    new_quantity: float | None = None,
    remove_item: str | None = None,
    add_item: FoodItem | None = None,
    meal_type: str | None = None,
    eaten_when: str | None = None,
) -> dict:
    """Correct or edit an existing meal. Pass its meal_id plus only the changes:

    item_name + new_quantity : set that item's quantity ("3 rotis not 2")
    remove_item              : drop that item ("actually no dal")
    add_item                 : add a food that was missed
    meal_type / eaten_when   : fix when it was or which meal it counts as

    Never use log_meal for a correction - that would double-count.
    """
    try:
        ctx = get_context()
        meal = _owned_meal(meal_id)
        if meal is None:
            return {"error": f"no meal found with id {meal_id}"}

        changed: list[str] = []

        if item_name is not None and new_quantity is not None:
            match = _match_item(meal, item_name)
            if match is None:
                return {"error": f"'{item_name}' is not in that meal", "items": [i.name for i in meal.items]}
            if match == "AMBIGUOUS":
                return {"error": f"more than one item matches '{item_name}'", "items": [i.name for i in meal.items]}
            repo.set_item_quantity(match.id, new_quantity, turn_id=ctx.turn_id)
            changed.append(f"{match.name} quantity -> {new_quantity}")

        if remove_item is not None:
            match = _match_item(meal, remove_item)
            if match is None or match == "AMBIGUOUS":
                return {"error": f"could not uniquely match '{remove_item}' to remove",
                        "items": [i.name for i in meal.items]}
            repo.remove_item(match.id, turn_id=ctx.turn_id)
            changed.append(f"removed {match.name}")

        if add_item is not None:
            est = resolve(add_item.name, add_item.unit)
            repo.add_item_to_meal(
                meal_id,
                {
                    "name": normalize_name(add_item.name),
                    "quantity": float(add_item.quantity),
                    "unit": add_item.unit or est.unit,
                    "confidence": est.confidence,
                    **est.as_item_fields(),
                },
                turn_id=ctx.turn_id,
            )
            changed.append(f"added {add_item.name}")

        if meal_type is not None:
            repo.update_meal_field(meal_id, "meal_type", meal_type, turn_id=ctx.turn_id)
            changed.append(f"meal_type -> {meal_type}")

        if eaten_when is not None:
            new_date = ctx.resolve_date(eaten_when)
            repo.update_meal_field(meal_id, "meal_date", new_date, turn_id=ctx.turn_id)
            repo.update_meal_field(meal_id, "eaten_at", _eaten_at_for(new_date), turn_id=ctx.turn_id)
            changed.append(f"date -> {new_date}")

        if not changed:
            return {"error": "no changes were given"}

        updated = repo.get_meal(meal_id)
        out = {"updated": True, "changes": changed, **_meal_view(updated),
               "today": _totals(ctx.user_id, ctx.local_date)}
        if updated.meal_date != ctx.local_date:
            out["that_day"] = _totals(ctx.user_id, updated.meal_date)
        return out
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"could not update the meal: {exc}"}


@tool
def delete_meal(meal_id: str) -> dict:
    """Remove a meal the user says they did not eat / logged by mistake.
    To change part of a meal, use update_meal instead."""
    try:
        ctx = get_context()
        meal = _owned_meal(meal_id)
        if meal is None:
            return {"error": f"no meal found with id {meal_id}"}
        ok = repo.soft_delete_meal(meal_id, turn_id=ctx.turn_id)
        return {
            "deleted": ok,
            "meal_id": meal_id,
            "note": "was already deleted" if not ok else "removed",
            "today": _totals(ctx.user_id, ctx.local_date),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"could not delete the meal: {exc}"}


# --- read tools ---


@tool
def get_daily_totals(date: str | None = None) -> dict:
    """Current calorie and macro totals for a day (default today). Use this to
    answer 'how am I doing?' style questions. `date` accepts 'today', 'yesterday',
    a weekday, or YYYY-MM-DD."""
    try:
        ctx = get_context()
        day = ctx.resolve_date(date)
        meals = repo.get_meals_for_date(ctx.user_id, day)
        return {
            "totals": _totals(ctx.user_id, day),
            "meals": [
                {"meal_id": m.id, "meal_type": m.meal_type, "description": m.description, "kcal": round(m.kcal)}
                for m in meals
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"could not read totals: {exc}"}


@tool
def get_meals(date: str | None = None) -> dict:
    """Look up meals the user has logged.

    date omitted        -> the 10 most recent meals
    date = "yesterday"  -> every meal that day (also accepts "today", a weekday, or YYYY-MM-DD)

    Use this to resolve "same as yesterday" (look up that day, then log_meal with
    the same items), to find a meal to correct, or to answer "what did I eat on ...".
    """
    try:
        ctx = get_context()
        if date:
            meals = repo.get_meals_for_date(ctx.user_id, ctx.resolve_date(date))
        else:
            meals = repo.get_recent_meals(ctx.user_id, limit=10)
        return {"count": len(meals), "meals": [_meal_view(m) for m in meals]}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"could not read meals: {exc}"}


@tool
def lookup_nutrition(items: list[FoodItem]) -> dict:
    """Estimate calories and macros for foods WITHOUT logging them. Use this only
    when the user is asking a question ('how many calories in a samosa?'), not
    when they are telling you what they ate."""
    try:
        estimates = resolve_many([(it.name, it.unit) for it in items])
        return {
            "items": [
                {
                    "name": it.name,
                    "unit": est.unit,
                    "per_unit": {
                        "kcal": round(est.kcal_per_unit),
                        "protein_g": round(est.protein_g_per_unit, 1),
                        "carbs_g": round(est.carbs_g_per_unit, 1),
                        "fat_g": round(est.fat_g_per_unit, 1),
                    },
                    "for_quantity": {
                        "quantity": it.quantity,
                        "kcal": round(est.kcal_per_unit * it.quantity),
                        "protein_g": round(est.protein_g_per_unit * it.quantity, 1),
                    },
                    "source": est.source,
                }
                for it, est in zip(items, estimates)
            ]
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"could not look up nutrition: {exc}"}


LOGGING_TOOLS = [log_meal, update_meal, delete_meal, get_daily_totals, get_meals, lookup_nutrition]
