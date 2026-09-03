"""SQLite connection handling and schema setup.

One database file (path from `config.DB_PATH`). Connections are opened per
call, run in WAL mode with foreign keys on, and hand back `sqlite3.Row` so
callers can use column names.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from calorai.config import DB_PATH, ensure_data_dir
from calorai.db.migrations import CURRENT_VERSION, apply_migrations

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_schema_applied = False

# One writer at a time within the process. The reflection pass writes memory from
# a background thread while a turn may be writing a meal on the main thread;
# serialising here keeps them from racing for SQLite's write lock. Writes are all
# short pure-SQL blocks, so holding this briefly costs nothing. Reads don't take
# it - WAL lets them run concurrently.
_write_lock = threading.Lock()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Bring the database up to date: run migrations for an existing DB, then
    apply the current schema (CREATE ... IF NOT EXISTS). Safe to call repeatedly."""
    global _schema_applied
    path = db_path or DB_PATH
    ensure_data_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    try:
        apply_migrations(conn)
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        if conn.execute("PRAGMA user_version").fetchone()[0] < CURRENT_VERSION:
            conn.execute(f"PRAGMA user_version = {CURRENT_VERSION}")
    finally:
        conn.close()
    if db_path is None:
        _schema_applied = True


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a ready-to-use connection, applying the schema once per process."""
    global _schema_applied
    path = db_path or DB_PATH
    if db_path is None and not _schema_applied:
        init_db(path)
    return _connect(path)


@contextmanager
def transaction(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Run a block of writes atomically, rolling back on any exception.

    Serialised process-wide (see `_write_lock`) and opened with `BEGIN IMMEDIATE`
    so the write lock is taken up front - a deferred `BEGIN` that upgrades to a
    writer only on the first UPDATE can deadlock against another connection that
    is doing the same thing.
    """
    with _write_lock:
        conn = get_connection(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
