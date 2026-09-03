-- CalorAI database schema. Applied on startup; every statement is idempotent.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;


-- One row per user. No auth: the id is the handle the user keeps to resume.
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL DEFAULT 'guest',
    timezone   TEXT NOT NULL DEFAULT 'UTC',
    created_at TEXT NOT NULL
);


-- One row per logging event. Soft-deleted (status), never hard-deleted, so a
-- correction can always be traced and daily totals recompute cleanly.
CREATE TABLE IF NOT EXISTS meals (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users (id),
    eaten_at    TEXT NOT NULL,               -- ISO-8601 UTC
    meal_date   TEXT NOT NULL,               -- YYYY-MM-DD in the user's local timezone
    meal_type   TEXT,                        -- breakfast | lunch | dinner | snack | NULL
    description TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT 'text', -- text | image | image+text
    status      TEXT NOT NULL DEFAULT 'active', -- active | deleted
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals (user_id, meal_date, status);


-- One row per food item within a meal. Macros are stored per unit, so a
-- correction ("3 rotis not 2") is a single quantity update and the line total
-- recomputes as quantity * *_per_unit. Nothing stores a running total.
CREATE TABLE IF NOT EXISTS meal_items (
    id                 TEXT PRIMARY KEY,
    meal_id            TEXT NOT NULL REFERENCES meals (id) ON DELETE CASCADE,
    position           INTEGER NOT NULL DEFAULT 0, -- order the items were logged in
    name               TEXT NOT NULL,
    quantity           REAL NOT NULL DEFAULT 1,
    unit               TEXT NOT NULL DEFAULT 'serving',
    kcal_per_unit      REAL NOT NULL DEFAULT 0,
    protein_g_per_unit REAL NOT NULL DEFAULT 0,
    carbs_g_per_unit   REAL NOT NULL DEFAULT 0,
    fat_g_per_unit     REAL NOT NULL DEFAULT 0,
    confidence         REAL NOT NULL DEFAULT 1.0,
    nutrition_source   TEXT NOT NULL DEFAULT 'model', -- seed | cache | model
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_meal ON meal_items (meal_id);


-- Durable user facts and routines. One active row per (user, type, key);
-- an update supersedes the old row rather than overwriting it.
CREATE TABLE IF NOT EXISTS memory (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users (id),
    type             TEXT NOT NULL,          -- diet | goal | preference | routine | fact
    key              TEXT NOT NULL,          -- canonical label, e.g. "protein_target"
    content          TEXT NOT NULL,          -- natural language
    structured_value TEXT,                   -- JSON: routine items, {"protein_g": 140}, ...
    meal_type        TEXT,                   -- for routines bound to a time of day
    status           TEXT NOT NULL DEFAULT 'active', -- active | superseded | inactive
    learned_via      TEXT NOT NULL DEFAULT 'stated', -- stated | inferred | confirmed
    confidence       REAL NOT NULL DEFAULT 1.0,
    source_turn_id   TEXT,
    use_count        INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_used_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_user ON memory (user_id, status, type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_active_key
    ON memory (user_id, type, key) WHERE status = 'active';


-- Audit trail for meal corrections. Powers "logged 2, corrected to 3" in the demo.
CREATE TABLE IF NOT EXISTS meal_edits (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meal_id    TEXT NOT NULL REFERENCES meals (id),
    field      TEXT NOT NULL,
    old_value  TEXT,
    new_value  TEXT,
    turn_id    TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edits_meal ON meal_edits (meal_id);


-- Learned food -> per-unit macros. First lookup for any food; filled from the
-- seed list or from a model estimate, then reused so the number never drifts.
CREATE TABLE IF NOT EXISTS nutrition_cache (
    name               TEXT PRIMARY KEY,     -- normalized specific food name
    unit               TEXT NOT NULL DEFAULT 'serving',
    kcal_per_unit      REAL NOT NULL,
    protein_g_per_unit REAL NOT NULL,
    carbs_g_per_unit   REAL NOT NULL,
    fat_g_per_unit     REAL NOT NULL,
    source             TEXT NOT NULL DEFAULT 'seed', -- seed | model
    created_at         TEXT NOT NULL
);
