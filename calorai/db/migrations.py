"""Schema migrations, keyed off SQLite's built-in ``PRAGMA user_version``.

`schema.sql` describes the current shape for a *fresh* database. For a database
created by an older release we run the numbered steps below to bring it up to
`CURRENT_VERSION`. Each step is small, forward-only, and runs inside one
transaction.
"""

from __future__ import annotations

import sqlite3

CURRENT_VERSION = 1

# Kept in sync with the CHECK bounds in schema.sql.
_MAX_NUMERIC = 1e6


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _rebuild_nutrition_cache(conn: sqlite3.Connection) -> None:
    """v0 -> v1: primary key changed from (name) to (name, unit), plus finite
    upper bounds. The table is only a regenerable cache, so the safe migration
    is to rebuild it empty; entries repopulate on the next lookup."""
    if "nutrition_cache" in _tables(conn):
        conn.execute("DROP TABLE nutrition_cache")
    conn.execute(
        """CREATE TABLE nutrition_cache (
               name               TEXT NOT NULL,
               unit               TEXT NOT NULL DEFAULT 'serving',
               kcal_per_unit      REAL NOT NULL CHECK (kcal_per_unit >= 0 AND kcal_per_unit < 1e6),
               protein_g_per_unit REAL NOT NULL CHECK (protein_g_per_unit >= 0 AND protein_g_per_unit < 1e6),
               carbs_g_per_unit   REAL NOT NULL CHECK (carbs_g_per_unit >= 0 AND carbs_g_per_unit < 1e6),
               fat_g_per_unit     REAL NOT NULL CHECK (fat_g_per_unit >= 0 AND fat_g_per_unit < 1e6),
               source             TEXT NOT NULL DEFAULT 'seed' CHECK (source IN ('seed', 'model')),
               created_at         TEXT NOT NULL,
               PRIMARY KEY (name, unit)
           )"""
    )


def _rebuild_meal_items(conn: sqlite3.Connection) -> None:
    """v0 -> v1: add finite upper bounds to the per-unit macro CHECKs so a
    malformed (e.g. infinite) value can never be stored. Real data is copied
    across; any pre-existing out-of-range value is clamped so the copy can't
    fail on an already-corrupt row."""
    if "meal_items" not in _tables(conn):
        return
    conn.execute("DROP TABLE IF EXISTS meal_items__v1_new")
    conn.execute(
        """CREATE TABLE meal_items__v1_new (
               id                 TEXT PRIMARY KEY,
               meal_id            TEXT NOT NULL REFERENCES meals (id) ON DELETE CASCADE,
               position           INTEGER NOT NULL DEFAULT 0,
               name               TEXT NOT NULL CHECK (length(trim(name)) > 0),
               quantity           REAL NOT NULL DEFAULT 1 CHECK (quantity > 0 AND quantity < 1e6),
               unit               TEXT NOT NULL DEFAULT 'serving',
               kcal_per_unit      REAL NOT NULL DEFAULT 0 CHECK (kcal_per_unit >= 0 AND kcal_per_unit < 1e6),
               protein_g_per_unit REAL NOT NULL DEFAULT 0 CHECK (protein_g_per_unit >= 0 AND protein_g_per_unit < 1e6),
               carbs_g_per_unit   REAL NOT NULL DEFAULT 0 CHECK (carbs_g_per_unit >= 0 AND carbs_g_per_unit < 1e6),
               fat_g_per_unit     REAL NOT NULL DEFAULT 0 CHECK (fat_g_per_unit >= 0 AND fat_g_per_unit < 1e6),
               confidence         REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
               nutrition_source   TEXT NOT NULL DEFAULT 'model' CHECK (nutrition_source IN ('seed', 'cache', 'model', 'failed')),
               created_at         TEXT NOT NULL
           )"""
    )
    cap = _MAX_NUMERIC - 1
    conn.execute(
        f"""INSERT INTO meal_items__v1_new
               (id, meal_id, position, name, quantity, unit, kcal_per_unit,
                protein_g_per_unit, carbs_g_per_unit, fat_g_per_unit,
                confidence, nutrition_source, created_at)
            SELECT id, meal_id, position, name,
                   MIN(MAX(quantity, 0.0001), {cap}),
                   unit,
                   MIN(MAX(kcal_per_unit, 0), {cap}),
                   MIN(MAX(protein_g_per_unit, 0), {cap}),
                   MIN(MAX(carbs_g_per_unit, 0), {cap}),
                   MIN(MAX(fat_g_per_unit, 0), {cap}),
                   MIN(MAX(confidence, 0), 1),
                   nutrition_source, created_at
            FROM meal_items"""
    )
    conn.execute("DROP TABLE meal_items")
    conn.execute("ALTER TABLE meal_items__v1_new RENAME TO meal_items")


def _to_v1(conn: sqlite3.Connection) -> None:
    _rebuild_nutrition_cache(conn)
    _rebuild_meal_items(conn)


_STEPS = {1: _to_v1}


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Bring `conn`'s database up to CURRENT_VERSION. A brand-new database (no
    tables yet) is left for `schema.sql` to populate and just stamped current."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= CURRENT_VERSION:
        return

    has_tables = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
    ).fetchone()

    if not has_tables:
        conn.execute(f"PRAGMA user_version = {CURRENT_VERSION}")
        return

    for target in range(version + 1, CURRENT_VERSION + 1):
        conn.execute("BEGIN")
        try:
            _STEPS[target](conn)
            conn.execute(f"PRAGMA user_version = {target}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
