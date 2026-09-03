"""Nutrition resolver tests. Seed/cache paths are exercised for real; the model
estimate is stubbed so tests stay offline and deterministic."""

from __future__ import annotations

import pytest

from calorai.db import database, repositories as repo
from calorai.nutrition import resolver, seed
from calorai.nutrition.resolver import NutritionEstimate


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "_schema_applied", False)
    database.init_db(db_file)
    yield


def test_normalize_name():
    assert resolver.normalize_name("  A  Paneer   Paratha ") == "paneer paratha"
    assert resolver.normalize_name("the biryani") == "biryani"


def test_seed_lookup_aliases_and_plurals():
    assert seed.lookup("roti")["kcal_per_unit"] == 105
    assert seed.lookup("chapati") == seed.lookup("roti")       # alias
    assert seed.lookup("paratha") == seed.lookup("plain paratha")
    assert seed.lookup("rotis") == seed.lookup("roti")         # simple plural
    assert seed.lookup("parathas") == seed.lookup("plain paratha")
    assert seed.lookup("nonexistent food") is None


def test_resolve_seed_hit_is_cached():
    est = resolver.resolve("roti", "piece")
    assert est.source == "seed"
    assert est.kcal_per_unit == 105

    cached = repo.get_cached_nutrition("roti")
    assert cached is not None and cached["source"] == "seed"


def test_resolve_second_call_uses_cache_not_model(monkeypatch):
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("model should not be called for a cached food")

    resolver.resolve("dal", "bowl")  # seeds the cache
    monkeypatch.setattr(resolver, "_model_estimate", _boom)
    est = resolver.resolve("dal", "bowl")
    assert est.source in ("seed", "cache")
    assert calls["n"] == 0


def test_resolve_model_path_is_cached(monkeypatch):
    def fake_estimate(name, unit_hint):
        return NutritionEstimate(name, unit_hint or "serving", 350, 12, 40, 15, "model", 0.6)

    monkeypatch.setattr(resolver, "_model_estimate", fake_estimate)
    est = resolver.resolve("dragon fruit smoothie", "glass")
    assert est.source == "model" and est.kcal_per_unit == 350

    cached = repo.get_cached_nutrition("dragon fruit smoothie")
    assert cached is not None and cached["source"] == "model" and cached["kcal_per_unit"] == 350


def test_resolve_failure_returns_zeroed_not_raise(monkeypatch):
    monkeypatch.setattr(resolver, "_model_estimate", lambda *a, **k: None)
    est = resolver.resolve("mystery dish", "plate")
    assert est.source == "failed"
    assert est.kcal_per_unit == 0.0
    assert est.confidence == 0.0


def test_resolve_many_mixed(monkeypatch):
    monkeypatch.setattr(
        resolver, "_model_estimate",
        lambda name, unit: NutritionEstimate(name, unit or "serving", 100, 1, 1, 1, "model", 0.6),
    )
    out = resolver.resolve_many([("roti", "piece"), ("chai", "cup"), ("weird thing", "bowl")])
    assert [e.source for e in out] == ["seed", "seed", "model"]
