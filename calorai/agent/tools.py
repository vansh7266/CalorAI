"""The tools the agent calls.

Read/write split: `get_*` and `lookup_nutrition` never mutate; `log_meal`,
`update_meal`, `delete_meal` do. Each tool returns a plain dict and never raises
- an error comes back as ``{"error": "..."}`` so the agent can recover or ask
the user.
"""

from __future__ import annotations

import logging
import math
from datetime import timezone

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from calorai.agent.context import get_context
from calorai.db import repositories as repo
from calorai.db.records import Meal, MealItem
from calorai.nutrition.resolver import normalize_name, resolve, resolve_many

logger = logging.getLogger("calorai.tools")

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


def _positive_finite(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("quantity must be a positive number")
    return float(value)


class FoodItem(BaseModel):
    name: str = Field(description="the food, singular and specific: 'aloo paratha', not 'parathas'")
    quantity: float = Field(default=1.0, gt=0, description="how many units (> 0)")
    unit: str = Field(default="serving", description="unit for quantity: piece, cup, bowl, glass, slice, plate...")

    @field_validator("quantity")
    @classmethod
    def _q(cls, v: float) -> float:
        return _positive_finite(v)

    @field_validator("name", "unit")
    @classmethod
    def _nonblank(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must not be blank")
        return v


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
                "quantity": _positive_finite(it.quantity),
                "unit": est.unit,  # the unit the resolved macros are actually for
                "confidence": max(0.0, min(1.0, est.confidence)),
                **est.as_item_fields(),
            }
        )
    return rows, unresolved


def _eaten_at_for(meal_date: str) -> str:
    ctx = get_context()
    if meal_date == ctx.local_date:
        return ctx.now_utc.isoformat(timespec="seconds")
    # A past day: use local noon on that date, converted to UTC.
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        tz = ZoneInfo(ctx.timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        tz = timezone.utc
    local_noon = _dt.fromisoformat(f"{meal_date}T12:00:00").replace(tzinfo=tz)
    return local_noon.astimezone(timezone.utc).isoformat(timespec="seconds")


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


def _valid_meal_type(value: str | None) -> str | None:
    """Normalize to one of the known meal types, or None (caller infers/asks)."""
    if not value:
        return None
    v = value.strip().lower()
    aliases = {"brunch": "breakfast", "supper": "dinner", "tea": "snack", "dessert": "snack"}
    v = aliases.get(v, v)
    return v if v in MEAL_TYPES else None




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
        mtype = _valid_meal_type(meal_type) or ctx.meal_type_for_now()
        rows, unresolved = _resolved_rows(items)
        meal = repo.insert_meal(
            user_id=ctx.user_id,
            eaten_at=_eaten_at_for(meal_date),
            meal_date=meal_date,
            meal_type=mtype,
            description=(note or _describe(items))[:500],
            source=ctx.source,  # trusted per-turn input type, not a model argument
            items=rows,
        )
        result = {"logged": True, **_meal_view(meal), "today": _totals(ctx.user_id, ctx.local_date)}
        if unresolved:
            result["could_not_estimate_calories_for"] = unresolved
        return result
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception:
        logger.exception("log_meal failed")
        return {"error": "could not log that meal"}


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
            return {"error": "no meal with that id"}

        # 1. validate every requested change first - nothing is written yet
        ops: list[dict] = []
        changed: list[str] = []

        if new_quantity is not None:
            if item_name is None:
                return {"error": "which item's quantity? pass item_name"}
            try:
                qty = _positive_finite(new_quantity)
            except ValueError as exc:
                return {"error": str(exc)}
            match = _match_item(meal, item_name)
            if match is None:
                return {"error": f"'{item_name}' is not in that meal", "items": [i.name for i in meal.items]}
            if match == "AMBIGUOUS":
                return {"error": f"more than one item matches '{item_name}'", "items": [i.name for i in meal.items]}
            ops.append({"op": "set_quantity", "item_id": match.id, "quantity": qty})
            changed.append(f"{match.name} -> {qty}")

        if remove_item is not None:
            match = _match_item(meal, remove_item)
            if match is None or match == "AMBIGUOUS":
                return {"error": f"could not uniquely match '{remove_item}'", "items": [i.name for i in meal.items]}
            ops.append({"op": "remove_item", "item_id": match.id})
            changed.append(f"removed {match.name}")

        if add_item is not None:
            est = resolve(add_item.name, add_item.unit)
            ops.append({"op": "add_item", "item": {
                "name": normalize_name(add_item.name),
                "quantity": _positive_finite(add_item.quantity),
                "unit": est.unit,
                "confidence": max(0.0, min(1.0, est.confidence)),
                **est.as_item_fields(),
            }})
            changed.append(f"added {add_item.name}")

        if meal_type is not None:
            mt = _valid_meal_type(meal_type)
            if mt is None:
                return {"error": f"meal type must be one of {', '.join(MEAL_TYPES)}"}
            ops.append({"op": "set_field", "field": "meal_type", "value": mt})
            changed.append(f"meal_type -> {mt}")

        if eaten_when is not None:
            new_date = ctx.resolve_date(eaten_when)
            ops.append({"op": "set_field", "field": "meal_date", "value": new_date})
            ops.append({"op": "set_field", "field": "eaten_at", "value": _eaten_at_for(new_date)})
            changed.append(f"date -> {new_date}")

        if not ops:
            return {"error": "no changes were given"}

        # 2. apply them all in one transaction (all-or-nothing)
        repo.apply_meal_edits(meal_id, ops, turn_id=ctx.turn_id)

        updated = repo.get_meal(meal_id)
        out = {"updated": True, "changes": changed, **_meal_view(updated),
               "today": _totals(ctx.user_id, ctx.local_date)}
        if updated.status == "deleted":
            out["note"] = "that removed the last item, so the meal is gone"
        elif updated.meal_date != ctx.local_date:
            out["that_day"] = _totals(ctx.user_id, updated.meal_date)
        return out
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception:
        logger.exception("update_meal failed")
        return {"error": "could not update that meal"}


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
    except Exception:
        logger.exception("delete_meal failed")
        return {"error": "could not delete that meal"}


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
    except Exception:  # pragma: no cover - defensive
        return {"error": "could not read your totals right now"}


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
    except Exception:  # pragma: no cover - defensive
        return {"error": "could not read your meals right now"}


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
    except Exception:  # pragma: no cover - defensive
        return {"error": "could not look up that nutrition info right now"}


LOGGING_TOOLS = [log_meal, update_meal, delete_meal, get_daily_totals, get_meals, lookup_nutrition]
