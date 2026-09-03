"""Nutrition resolution: seed table -> cache -> model estimate -> cache.

This is a consistency layer, not a knowledge base. The models already know food
macros; the problem is that a fresh estimate drifts between calls, which would
make daily totals wobble. So every food is resolved once and the result is
cached, keyed by its normalized name.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel, Field

from calorai.db import repositories as repo
from calorai.models.gateway import as_structured, get_worker_model
from calorai.nutrition import seed

_ARTICLES = re.compile(r"^(a|an|the|some|my)\s+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    text = name.strip().lower()
    text = _ARTICLES.sub("", text)
    text = _WS.sub(" ", text)
    return text.strip()


@dataclass
class NutritionEstimate:
    name: str
    unit: str
    kcal_per_unit: float
    protein_g_per_unit: float
    carbs_g_per_unit: float
    fat_g_per_unit: float
    source: str  # "seed" | "cache" | "model" | "failed"
    confidence: float

    def as_item_fields(self) -> dict:
        return {
            "unit": self.unit,
            "kcal_per_unit": self.kcal_per_unit,
            "protein_g_per_unit": self.protein_g_per_unit,
            "carbs_g_per_unit": self.carbs_g_per_unit,
            "fat_g_per_unit": self.fat_g_per_unit,
            "nutrition_source": self.source,
        }


class _MacroGuess(BaseModel):
    unit: str = Field(description="the single unit the values are for, e.g. 'piece', 'cup', 'bowl'")
    kcal: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)


def _from_macros(name: str, macros: dict, source: str, confidence: float) -> NutritionEstimate:
    return NutritionEstimate(
        name=name,
        unit=macros["unit"],
        kcal_per_unit=float(macros["kcal_per_unit"]),
        protein_g_per_unit=float(macros["protein_g_per_unit"]),
        carbs_g_per_unit=float(macros["carbs_g_per_unit"]),
        fat_g_per_unit=float(macros["fat_g_per_unit"]),
        source=source,
        confidence=confidence,
    )


def _model_estimate(name: str, unit_hint: str | None) -> NutritionEstimate | None:
    unit_clause = f" Give values per one {unit_hint}." if unit_hint else " Choose the most natural single serving unit."
    prompt = (
        f"Estimate the nutrition of: {name}.{unit_clause} "
        "Use typical real-world values. Reply with the structured fields only."
    )
    try:
        guess: _MacroGuess = as_structured(get_worker_model(), _MacroGuess).invoke(prompt)
    except Exception:
        return None

    return NutritionEstimate(
        name=name,
        unit=(guess.unit or unit_hint or "serving").strip().lower(),
        kcal_per_unit=guess.kcal,
        protein_g_per_unit=guess.protein_g,
        carbs_g_per_unit=guess.carbs_g,
        fat_g_per_unit=guess.fat_g,
        source="model",
        confidence=0.6,
    )


def resolve(name: str, unit_hint: str | None = None) -> NutritionEstimate:
    """Resolve one food's per-unit macros. Never raises - on total failure it
    returns a zeroed estimate with source 'failed' so the meal can still be
    logged and the gap surfaced to the user."""
    key = normalize_name(name)

    cached = repo.get_cached_nutrition(key)
    if cached:
        source = "seed" if cached["source"] == "seed" else "cache"
        return _from_macros(key, cached, source, 1.0 if source == "seed" else 0.7)

    seeded = seed.lookup(key)
    if seeded:
        repo.put_cached_nutrition(key, seeded, source="seed")
        return _from_macros(key, seeded, "seed", 1.0)

    estimate = _model_estimate(name, unit_hint)
    if estimate is not None:
        repo.put_cached_nutrition(
            key,
            {
                "unit": estimate.unit,
                "kcal_per_unit": estimate.kcal_per_unit,
                "protein_g_per_unit": estimate.protein_g_per_unit,
                "carbs_g_per_unit": estimate.carbs_g_per_unit,
                "fat_g_per_unit": estimate.fat_g_per_unit,
            },
            source="model",
        )
        return estimate

    return NutritionEstimate(
        name=key,
        unit=unit_hint or "serving",
        kcal_per_unit=0.0,
        protein_g_per_unit=0.0,
        carbs_g_per_unit=0.0,
        fat_g_per_unit=0.0,
        source="failed",
        confidence=0.0,
    )


def resolve_many(foods: list[tuple[str, str | None]]) -> list[NutritionEstimate]:
    """Resolve several foods, running the (slow) model estimates in parallel.
    Seed and cache hits return immediately; only genuine misses cost a call."""
    if not foods:
        return []
    if len(foods) == 1:
        name, unit = foods[0]
        return [resolve(name, unit)]
    with ThreadPoolExecutor(max_workers=min(4, len(foods))) as pool:
        return list(pool.map(lambda f: resolve(f[0], f[1]), foods))
