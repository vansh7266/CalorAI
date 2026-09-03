"""Upgrade path: a database created by the previous release (no user_version,
old nutrition_cache primary key, loose numeric CHECKs) must migrate cleanly and
keep the user's existing data."""

from __future__ import annotations

import sqlite3

import pytest

from calorai.db import database, repositories as repo
from calorai.db.migrations import CURRENT_VERSION

# The shape shipped by commit 261716b / 9ce488d - before migrations existed.
_OLD_SCHEMA = """
CREATE TABLE users (
    id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT 'guest',
    timezone TEXT NOT NULL DEFAULT 'UTC', created_at TEXT NOT NULL
);
CREATE TABLE meals (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users (id),
    eaten_at TEXT NOT NULL, meal_date TEXT NOT NULL,
    meal_type TEXT, description TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'text', status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE meal_items (
    id TEXT PRIMARY KEY, meal_id TEXT NOT NULL REFERENCES meals (id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0, name TEXT NOT NULL,
    quantity REAL NOT NULL DEFAULT 1 CHECK (quantity > 0),
    unit TEXT NOT NULL DEFAULT 'serving',
    kcal_per_unit REAL NOT NULL DEFAULT 0 CHECK (kcal_per_unit >= 0),
    protein_g_per_unit REAL NOT NULL DEFAULT 0 CHECK (protein_g_per_unit >= 0),
    carbs_g_per_unit REAL NOT NULL DEFAULT 0 CHECK (carbs_g_per_unit >= 0),
    fat_g_per_unit REAL NOT NULL DEFAULT 0 CHECK (fat_g_per_unit >= 0),
    confidence REAL NOT NULL DEFAULT 1.0, nutrition_source TEXT NOT NULL DEFAULT 'model',
    created_at TEXT NOT NULL
);
CREATE TABLE memory (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users (id),
    type TEXT NOT NULL, key TEXT NOT NULL, content TEXT NOT NULL,
    structured_value TEXT, meal_type TEXT, status TEXT NOT NULL DEFAULT 'active',
    learned_via TEXT NOT NULL DEFAULT 'stated', confidence REAL NOT NULL DEFAULT 1.0,
    source_turn_id TEXT, use_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_used_at TEXT
);
CREATE TABLE meal_edits (
    id INTEGER PRIMARY KEY AUTOINCREMENT, meal_id TEXT NOT NULL REFERENCES meals (id),
    field TEXT NOT NULL, old_value TEXT, new_value TEXT, turn_id TEXT, created_at TEXT NOT NULL
);
CREATE TABLE nutrition_cache (
    name TEXT PRIMARY KEY, unit TEXT NOT NULL DEFAULT 'serving',
    kcal_per_unit REAL NOT NULL, protein_g_per_unit REAL NOT NULL,
    carbs_g_per_unit REAL NOT NULL, fat_g_per_unit REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'seed', created_at TEXT NOT NULL
);
"""


@pytest.fixture
def old_db(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    now = "2026-09-01T12:00:00+00:00"
    conn.execute("INSERT INTO users VALUES ('usr_old', 'Sam', 'UTC', ?)", (now,))
    conn.execute(
        "INSERT INTO meals VALUES ('meal_old','usr_old',?,?,'lunch','2 roti','text','active',?,?)",
        (now, "2026-09-01", now, now),
    )
    # an item with a wildly out-of-range value, as the old loose CHECK would allow
    conn.execute(
        "INSERT INTO meal_items VALUES ('item_old','meal_old',0,'roti',2,'piece',105,3,20,1,1.0,'seed',?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO meal_items VALUES ('item_bad','meal_old',1,'ghee',1e15,'tbsp',1e15,0,0,1e15,1.0,'model',?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO nutrition_cache VALUES ('roti','piece',105,3,20,1,'seed',?)", (now,)
    )
    conn.commit()
    conn.close()
    assert sqlite3.connect(path).execute("PRAGMA user_version").fetchone()[0] == 0
    return path


def test_upgrade_from_previous_schema(old_db, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(database, "_schema_applied", False)

    database.init_db(old_db)  # runs migrations, then applies current schema

    conn = sqlite3.connect(old_db)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_VERSION

        # nutrition_cache rebuilt with the composite key (old rows are a regenerable cache)
        pk_cols = [r[1] for r in conn.execute("PRAGMA table_info(nutrition_cache)") if r[5]]
        assert pk_cols == ["name", "unit"]

        # the user's real meal survived
        assert conn.execute("SELECT description FROM meals WHERE id='meal_old'").fetchone()[0] == "2 roti"
        good = conn.execute("SELECT quantity FROM meal_items WHERE id='item_old'").fetchone()[0]
        assert good == 2

        # the previously-out-of-range row was clamped into the new bounds, not dropped
        bad = conn.execute(
            "SELECT quantity, kcal_per_unit FROM meal_items WHERE id='item_bad'"
        ).fetchone()
        assert bad[0] < 1e6 and bad[1] < 1e6

        # new CHECK bounds are actually enforced now
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO meal_items VALUES ('x','meal_old',9,'x',1e9,'g',0,0,0,0,1,'model','t')"
            )
    finally:
        conn.close()

    # and the app works against the upgraded DB
    monkeypatch.setattr(repo, "get_connection", lambda *a, **k: database._connect(old_db))
    est = repo.get_cached_nutrition("roti", "piece")  # composite-key read, no OperationalError
    assert est is None or est["name"] == "roti"
    repo.put_cached_nutrition("dosa", {"unit": "piece", "kcal_per_unit": 160,
                                       "protein_g_per_unit": 3, "carbs_g_per_unit": 30,
                                       "fat_g_per_unit": 3}, source="model")


def test_fresh_db_is_stamped_current(tmp_path, monkeypatch):
    path = tmp_path / "fresh.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(path)
    assert sqlite3.connect(path).execute("PRAGMA user_version").fetchone()[0] == CURRENT_VERSION
