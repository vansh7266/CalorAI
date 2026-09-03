"""SQLite connection handling and schema setup.

One database file (path from `config.DB_PATH`). Connections are opened per
call, run in WAL mode with foreign keys on, and hand back `sqlite3.Row` so
callers can use column names.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from calorai.config import DB_PATH, ensure_data_dir

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_schema_applied = False


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the schema if it is not there yet. Safe to call repeatedly."""
    global _schema_applied
    path = db_path or DB_PATH
    ensure_data_dir()
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
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
    """Run a block of writes atomically, rolling back on any exception."""
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
