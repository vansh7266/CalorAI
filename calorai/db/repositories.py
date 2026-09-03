"""Data-access functions for every table.

All SQL lives here and every query is parameterised. Callers get back the data
classes from `records.py`, never raw rows.
"""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
from datetime import datetime, timezone

from calorai.db.database import get_connection, transaction
from calorai.db.records import DailyTotals, Meal, MealItem, MemoryRecord, User

# Values at or above this are treated as malformed model/vision output, not real
# food. Mirrors the CHECK bounds in schema.sql.
_MAX_NUMERIC = 1e6


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _bounded(value: float, *, allow_zero: bool = True) -> float:
    """Reject NaN / infinity / out-of-range numbers before they reach SQLite,
    where a bare ``> 0`` CHECK would happily accept +infinity."""
    v = float(value)
    low_ok = v >= 0 if allow_zero else v > 0
    if not math.isfinite(v) or not low_ok or v >= _MAX_NUMERIC:
        raise ValueError("numeric value out of range")
    return v


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Users ---


def create_user(name: str = "guest", timezone_name: str = "UTC") -> User:
    user = User(id=new_id("usr"), name=name.strip() or "guest", timezone=timezone_name, created_at=_utc_now())
    with transaction() as conn:
        conn.execute(
            "INSERT INTO users (id, name, timezone, created_at) VALUES (?, ?, ?, ?)",
            (user.id, user.name, user.timezone, user.created_at),
        )
    return user


def get_user(user_id: str) -> User | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return User.from_row(row) if row else None
    finally:
        conn.close()


def update_user(user_id: str, *, name: str | None = None, timezone_name: str | None = None) -> None:
    sets, params = [], []
    if name is not None:
        sets.append("name = ?")
        params.append(name.strip() or "guest")
    if timezone_name is not None:
        sets.append("timezone = ?")
        params.append(timezone_name)
    if not sets:
        return
    params.append(user_id)
    with transaction() as conn:
        conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", params)


# --- Meals ---


def insert_meal(
    *,
    user_id: str,
    eaten_at: str,
    meal_date: str,
    meal_type: str | None,
    description: str,
    source: str,
    items: list[dict],
) -> Meal:
    """Create a meal and its items in one transaction. `items` are dicts with the
    meal_items columns (minus id / meal_id / created_at)."""
    now = _utc_now()
    meal_id = new_id("meal")
    with transaction() as conn:
        conn.execute(
            """INSERT INTO meals
               (id, user_id, eaten_at, meal_date, meal_type, description, source, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
            (meal_id, user_id, eaten_at, meal_date, meal_type, description, source, now, now),
        )
        for position, item in enumerate(items):
            _insert_item(conn, meal_id, item, now, position)
    return get_meal(meal_id)  # type: ignore[return-value]


def _insert_item(conn: sqlite3.Connection, meal_id: str, item: dict, now: str, position: int) -> str:
    item_id = new_id("item")
    conn.execute(
        """INSERT INTO meal_items
           (id, meal_id, position, name, quantity, unit, kcal_per_unit, protein_g_per_unit,
            carbs_g_per_unit, fat_g_per_unit, confidence, nutrition_source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            meal_id,
            position,
            item["name"],
            _bounded(item.get("quantity", 1), allow_zero=False),
            item.get("unit", "serving"),
            _bounded(item.get("kcal_per_unit", 0)),
            _bounded(item.get("protein_g_per_unit", 0)),
            _bounded(item.get("carbs_g_per_unit", 0)),
            _bounded(item.get("fat_g_per_unit", 0)),
            max(0.0, min(1.0, float(item.get("confidence", 1.0)))),
            item.get("nutrition_source", "model"),
            now,
        ),
    )
    return item_id


def get_meal(meal_id: str) -> Meal | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM meals WHERE id = ?", (meal_id,)).fetchone()
        if not row:
            return None
        items = [
            MealItem.from_row(r)
            for r in conn.execute(
                "SELECT * FROM meal_items WHERE meal_id = ? ORDER BY position, created_at, id", (meal_id,)
            )
        ]
        return Meal.from_row(row, items)
    finally:
        conn.close()


def get_meals_for_date(user_id: str, meal_date: str, *, include_deleted: bool = False) -> list[Meal]:
    return _get_meals(
        "WHERE user_id = ? AND meal_date = ?" + ("" if include_deleted else " AND status = 'active'"),
        (user_id, meal_date),
    )


def get_meals_between(user_id: str, start_date: str, end_date: str) -> list[Meal]:
    return _get_meals(
        "WHERE user_id = ? AND meal_date BETWEEN ? AND ? AND status = 'active'",
        (user_id, start_date, end_date),
    )


def get_recent_meals(user_id: str, limit: int = 10) -> list[Meal]:
    return _get_meals("WHERE user_id = ? AND status = 'active'", (user_id,), order="ORDER BY eaten_at DESC, created_at DESC, id DESC", limit=limit)


def _get_meals(where: str, params: tuple, *, order: str = "ORDER BY eaten_at ASC", limit: int | None = None) -> list[Meal]:
    conn = get_connection()
    try:
        sql = f"SELECT * FROM meals {where} {order}"
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)
        meal_rows = conn.execute(sql, params).fetchall()
        if not meal_rows:
            return []
        ids = [r["id"] for r in meal_rows]
        placeholders = ",".join("?" * len(ids))
        item_rows = conn.execute(
            f"SELECT * FROM meal_items WHERE meal_id IN ({placeholders}) ORDER BY position, created_at, id", ids
        ).fetchall()
        items_by_meal: dict[str, list[MealItem]] = {}
        for r in item_rows:
            items_by_meal.setdefault(r["meal_id"], []).append(MealItem.from_row(r))
        return [Meal.from_row(r, items_by_meal.get(r["id"], [])) for r in meal_rows]
    finally:
        conn.close()


def soft_delete_meal(meal_id: str, *, turn_id: str | None = None) -> bool:
    with transaction() as conn:
        row = conn.execute("SELECT status FROM meals WHERE id = ?", (meal_id,)).fetchone()
        if not row or row["status"] == "deleted":
            return False
        conn.execute(
            "UPDATE meals SET status = 'deleted', updated_at = ? WHERE id = ?", (_utc_now(), meal_id)
        )
        _record_edit(conn, meal_id, "status", "active", "deleted", turn_id)
    return True


def update_meal_field(meal_id: str, field: str, value: str | None, *, turn_id: str | None = None) -> bool:
    if field not in {"meal_type", "eaten_at", "meal_date", "description"}:
        raise ValueError(f"field '{field}' is not editable")
    with transaction() as conn:
        row = conn.execute(f"SELECT {field} FROM meals WHERE id = ?", (meal_id,)).fetchone()
        if not row:
            return False
        old = row[field]
        conn.execute(
            f"UPDATE meals SET {field} = ?, updated_at = ? WHERE id = ?", (value, _utc_now(), meal_id)
        )
        _record_edit(conn, meal_id, field, str(old), str(value), turn_id)
    return True


# --- Meal items ---


def set_item_quantity(item_id: str, quantity: float, *, turn_id: str | None = None) -> bool:
    with transaction() as conn:
        row = conn.execute("SELECT meal_id, name, quantity FROM meal_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE meal_items SET quantity = ? WHERE id = ?", (float(quantity), item_id))
        conn.execute("UPDATE meals SET updated_at = ? WHERE id = ?", (_utc_now(), row["meal_id"]))
        _record_edit(conn, row["meal_id"], f"item:{row['name']}:quantity", str(row["quantity"]), str(quantity), turn_id)
    return True


def set_item_nutrition(item_id: str, macros: dict, *, source: str) -> None:
    with transaction() as conn:
        conn.execute(
            """UPDATE meal_items
               SET kcal_per_unit = ?, protein_g_per_unit = ?, carbs_g_per_unit = ?,
                   fat_g_per_unit = ?, nutrition_source = ?
               WHERE id = ?""",
            (
                float(macros.get("kcal_per_unit", 0)),
                float(macros.get("protein_g_per_unit", 0)),
                float(macros.get("carbs_g_per_unit", 0)),
                float(macros.get("fat_g_per_unit", 0)),
                source,
                item_id,
            ),
        )


def add_item_to_meal(meal_id: str, item: dict, *, turn_id: str | None = None) -> str | None:
    with transaction() as conn:
        meal = conn.execute("SELECT id FROM meals WHERE id = ?", (meal_id,)).fetchone()
        if not meal:
            return None
        next_pos = conn.execute(
            "SELECT COALESCE(MAX(position) + 1, 0) AS p FROM meal_items WHERE meal_id = ?", (meal_id,)
        ).fetchone()["p"]
        item_id = _insert_item(conn, meal_id, item, _utc_now(), next_pos)
        conn.execute("UPDATE meals SET updated_at = ? WHERE id = ?", (_utc_now(), meal_id))
        _record_edit(conn, meal_id, "item:add", None, item["name"], turn_id)
        return item_id


def remove_item(item_id: str, *, turn_id: str | None = None) -> bool:
    with transaction() as conn:
        row = conn.execute("SELECT meal_id, name FROM meal_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM meal_items WHERE id = ?", (item_id,))
        conn.execute("UPDATE meals SET updated_at = ? WHERE id = ?", (_utc_now(), row["meal_id"]))
        _record_edit(conn, row["meal_id"], "item:remove", row["name"], None, turn_id)
    return True


_EDITABLE_MEAL_FIELDS = {"meal_type", "meal_date", "eaten_at", "description"}


def apply_meal_edits(meal_id: str, ops: list[dict], *, turn_id: str | None = None) -> bool:
    """Apply several validated changes to one meal in a SINGLE transaction - all
    or nothing. `ops` entries:
        {"op": "set_quantity", "item_id": ..., "quantity": ...}
        {"op": "remove_item",  "item_id": ...}
        {"op": "add_item",     "item": {...}}   # meal_items dict, see _insert_item
        {"op": "set_field",    "field": "meal_type"|"meal_date"|"eaten_at", "value": ...}
    After the ops, the meal's description is regenerated from its remaining items,
    and a meal left with zero items is soft-deleted.
    """
    now = _utc_now()
    with transaction() as conn:
        meal = conn.execute("SELECT id, status FROM meals WHERE id = ?", (meal_id,)).fetchone()
        if not meal:
            return False

        for op in ops:
            kind = op["op"]
            if kind == "set_quantity":
                row = conn.execute(
                    "SELECT name, quantity FROM meal_items WHERE id = ? AND meal_id = ?",
                    (op["item_id"], meal_id),
                ).fetchone()
                if not row:
                    raise ValueError("item not in meal")
                new_qty = _bounded(op["quantity"], allow_zero=False)
                conn.execute("UPDATE meal_items SET quantity = ? WHERE id = ?", (new_qty, op["item_id"]))
                _record_edit(conn, meal_id, f"item:{row['name']}:quantity", str(row["quantity"]), str(new_qty), turn_id)
            elif kind == "remove_item":
                row = conn.execute(
                    "SELECT name FROM meal_items WHERE id = ? AND meal_id = ?", (op["item_id"], meal_id)
                ).fetchone()
                if not row:
                    raise ValueError("item not in meal")
                conn.execute("DELETE FROM meal_items WHERE id = ?", (op["item_id"],))
                _record_edit(conn, meal_id, "item:remove", row["name"], None, turn_id)
            elif kind == "add_item":
                pos = conn.execute(
                    "SELECT COALESCE(MAX(position) + 1, 0) AS p FROM meal_items WHERE meal_id = ?", (meal_id,)
                ).fetchone()["p"]
                _insert_item(conn, meal_id, op["item"], now, pos)
                _record_edit(conn, meal_id, "item:add", None, op["item"]["name"], turn_id)
            elif kind == "set_field":
                field = op["field"]
                if field not in _EDITABLE_MEAL_FIELDS:
                    raise ValueError(f"field '{field}' is not editable")
                old = conn.execute(f"SELECT {field} FROM meals WHERE id = ?", (meal_id,)).fetchone()[field]
                conn.execute(f"UPDATE meals SET {field} = ? WHERE id = ?", (op["value"], meal_id))
                _record_edit(conn, meal_id, field, str(old), str(op["value"]), turn_id)
            else:
                raise ValueError(f"unknown op {kind!r}")

        remaining = conn.execute(
            "SELECT name, quantity, unit FROM meal_items WHERE meal_id = ? ORDER BY position, created_at, id", (meal_id,)
        ).fetchall()

        if remaining:
            desc = ", ".join(
                (f"{int(r['quantity']) if float(r['quantity']).is_integer() else r['quantity']} {r['name']}"
                 if r["quantity"] != 1 else r["name"])
                for r in remaining
            )
            conn.execute("UPDATE meals SET description = ?, updated_at = ? WHERE id = ?", (desc, now, meal_id))
        else:
            # nothing left in the meal - soft-delete it so it stops counting
            conn.execute("UPDATE meals SET status = 'deleted', updated_at = ? WHERE id = ?", (now, meal_id))
            _record_edit(conn, meal_id, "status", "active", "deleted", turn_id)

        conn.execute("UPDATE meals SET updated_at = ? WHERE id = ?", (now, meal_id))
    return True


# --- Daily totals (always computed, never stored) ---


def daily_totals(user_id: str, meal_date: str) -> DailyTotals:
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT
                   COALESCE(SUM(i.quantity * i.kcal_per_unit), 0)      AS kcal,
                   COALESCE(SUM(i.quantity * i.protein_g_per_unit), 0) AS protein_g,
                   COALESCE(SUM(i.quantity * i.carbs_g_per_unit), 0)   AS carbs_g,
                   COALESCE(SUM(i.quantity * i.fat_g_per_unit), 0)     AS fat_g,
                   COUNT(DISTINCT m.id)                                AS meal_count
               FROM meals m
               LEFT JOIN meal_items i ON i.meal_id = m.id
               WHERE m.user_id = ? AND m.meal_date = ? AND m.status = 'active'""",
            (user_id, meal_date),
        ).fetchone()
        return DailyTotals(
            date=meal_date,
            kcal=row["kcal"],
            protein_g=row["protein_g"],
            carbs_g=row["carbs_g"],
            fat_g=row["fat_g"],
            meal_count=row["meal_count"],
        )
    finally:
        conn.close()


# --- Meal edit audit ---


def _record_edit(
    conn: sqlite3.Connection, meal_id: str, field: str, old: str | None, new: str | None, turn_id: str | None
) -> None:
    conn.execute(
        "INSERT INTO meal_edits (meal_id, field, old_value, new_value, turn_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (meal_id, field, old, new, turn_id, _utc_now()),
    )


def get_meal_edits(meal_id: str) -> list[dict]:
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT field, old_value, new_value, turn_id, created_at FROM meal_edits WHERE meal_id = ? ORDER BY id",
            (meal_id,),
        )]
    finally:
        conn.close()


# --- Memory ---


def upsert_memory(
    *,
    user_id: str,
    type: str,
    key: str,
    content: str,
    structured_value: dict | list | None = None,
    meal_type: str | None = None,
    learned_via: str = "stated",
    confidence: float = 1.0,
    source_turn_id: str | None = None,
) -> MemoryRecord:
    """Write a memory. Any existing active row with the same (user, type, key) is
    marked 'superseded' first, so history is preserved."""
    now = _utc_now()
    mem_id = new_id("mem")
    payload = json.dumps(structured_value) if structured_value is not None else None
    with transaction() as conn:
        conn.execute(
            "UPDATE memory SET status = 'superseded', updated_at = ? WHERE user_id = ? AND type = ? AND key = ? AND status = 'active'",
            (now, user_id, type, key),
        )
        conn.execute(
            """INSERT INTO memory
               (id, user_id, type, key, content, structured_value, meal_type, status,
                learned_via, confidence, source_turn_id, use_count, created_at, updated_at, last_used_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, 0, ?, ?, NULL)""",
            (mem_id, user_id, type, key, content, payload, meal_type, learned_via, confidence, source_turn_id, now, now),
        )
    return get_memory(mem_id)  # type: ignore[return-value]


def get_memory(mem_id: str) -> MemoryRecord | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM memory WHERE id = ?", (mem_id,)).fetchone()
        return MemoryRecord.from_row(row) if row else None
    finally:
        conn.close()


def get_active_memory(user_id: str, *, types: list[str] | None = None) -> list[MemoryRecord]:
    conn = get_connection()
    try:
        sql = "SELECT * FROM memory WHERE user_id = ? AND status = 'active'"
        params: list = [user_id]
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            params.extend(types)
        sql += " ORDER BY type, updated_at DESC"
        return [MemoryRecord.from_row(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def search_memory(user_id: str, query: str, *, limit: int = 10) -> list[MemoryRecord]:
    """Keyword search over active memory (content + key). Small per-user data, so
    LIKE is enough; a vector index would only matter at scale."""
    conn = get_connection()
    try:
        like = f"%{query.strip()}%"
        rows = conn.execute(
            """SELECT * FROM memory
               WHERE user_id = ? AND status = 'active' AND (content LIKE ? OR key LIKE ?)
               ORDER BY use_count DESC, updated_at DESC LIMIT ?""",
            (user_id, like, like, limit),
        ).fetchall()
        return [MemoryRecord.from_row(r) for r in rows]
    finally:
        conn.close()


def deactivate_memory(mem_id: str) -> bool:
    with transaction() as conn:
        cur = conn.execute(
            "UPDATE memory SET status = 'inactive', updated_at = ? WHERE id = ? AND status = 'active'",
            (_utc_now(), mem_id),
        )
        return cur.rowcount > 0


def bump_memory_use(mem_id: str) -> None:
    with transaction() as conn:
        conn.execute(
            "UPDATE memory SET use_count = use_count + 1, last_used_at = ? WHERE id = ?", (_utc_now(), mem_id)
        )


# --- Nutrition cache ---


def get_cached_nutrition(name: str, unit: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM nutrition_cache WHERE name = ? AND unit = ?",
            (name.strip().lower(), unit.strip().lower()),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def put_cached_nutrition(name: str, macros: dict, *, source: str = "model") -> None:
    with transaction() as conn:
        conn.execute(
            """INSERT INTO nutrition_cache
               (name, unit, kcal_per_unit, protein_g_per_unit, carbs_g_per_unit, fat_g_per_unit, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name, unit) DO UPDATE SET
                   kcal_per_unit = excluded.kcal_per_unit,
                   protein_g_per_unit = excluded.protein_g_per_unit,
                   carbs_g_per_unit = excluded.carbs_g_per_unit,
                   fat_g_per_unit = excluded.fat_g_per_unit,
                   source = excluded.source""",
            (
                name.strip().lower(),
                str(macros.get("unit", "serving")).strip().lower(),
                _bounded(macros["kcal_per_unit"]),
                _bounded(macros["protein_g_per_unit"]),
                _bounded(macros["carbs_g_per_unit"]),
                _bounded(macros["fat_g_per_unit"]),
                source,
                _utc_now(),
            ),
        )
