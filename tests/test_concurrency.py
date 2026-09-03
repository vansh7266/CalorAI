"""A background thread (the reflection pass) writing memory must not lock out a
meal write on the main thread. This reproduces the "database is locked" seen in
an interactive session, where turns come back to back and the previous turn's
reflection thread is still writing."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

from calorai.agent import context as ctxmod
from calorai.agent.context import TurnContext
from calorai.agent.tools import log_meal, update_meal
from calorai.db import database, repositories as repo


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)
    user = repo.create_user("A", "UTC")
    token = ctxmod.set_context(
        TurnContext(user.id, "turn", "UTC", datetime(2026, 9, 4, 13, 0, tzinfo=timezone.utc))
    )
    yield user
    ctxmod.reset_context(token)


def test_meal_writes_survive_a_hammering_background_writer(env):
    stop = threading.Event()
    errors: list[str] = []

    def hammer_memory() -> None:
        i = 0
        while not stop.is_set():
            try:
                repo.upsert_memory(
                    user_id=env.id, type="fact", key=f"k{i % 5}", content=f"note {i}",
                    learned_via="inferred", confidence=0.7,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"memory: {exc}")
            i += 1

    writer = threading.Thread(target=hammer_memory, daemon=True)
    writer.start()
    try:
        res = log_meal.invoke({"items": [{"name": "roti", "quantity": 2, "unit": "piece"}]})
        assert res.get("logged") is True
        meal_id = res["meal_id"]
        for q in (3, 4, 5, 2, 6):
            out = update_meal.invoke({"meal_id": meal_id, "item_name": "roti", "new_quantity": q})
            assert "error" not in out, out
            assert repo.get_meal(meal_id).items[0].quantity == q
    finally:
        stop.set()
        writer.join(timeout=2)

    assert not errors, errors[:3]
    # final correction stuck
    assert repo.get_meal(meal_id).items[0].quantity == 6


def test_parallel_meal_writes_are_serialised(env):
    """Two threads each logging meals - every write lands, none is lost to a lock."""
    made: list[str] = []
    errs: list[str] = []
    collect = threading.Lock()

    def log_a_few(tag: str) -> None:
        token = ctxmod.set_context(  # ContextVar is per-thread
            TurnContext(env.id, f"turn-{tag}", "UTC", datetime(2026, 9, 4, 13, 0, tzinfo=timezone.utc))
        )
        try:
            for n in range(6):
                try:
                    r = log_meal.invoke({"items": [{"name": f"item{tag}{n}", "quantity": 1, "unit": "piece"}]})
                    if r.get("logged"):
                        with collect:
                            made.append(r["meal_id"])
                except Exception as exc:  # noqa: BLE001
                    errs.append(str(exc))
                time.sleep(0.001)
        finally:
            ctxmod.reset_context(token)

    threads = [threading.Thread(target=log_a_few, args=(t,)) for t in ("x", "y")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errs, errs[:3]
    assert len(made) == 12
    assert len(set(made)) == 12  # no id collisions
