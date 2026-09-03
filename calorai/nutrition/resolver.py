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


# Sane per-unit ceilings. Even a kilogram of pure fat is ~9000 kcal, so these
# bounds only ever trip on a model that has returned nonsense (or infinity).
_MAX_KCAL_PER_UNIT = 20_000.0
_MAX_GRAMS_PER_UNIT = 5_000.0


class _MacroGuess(BaseModel):
    unit: str = Field(description="the single unit the values are for, e.g. 'piece', 'cup', 'bowl'")
    kcal: float = Field(ge=0, le=_MAX_KCAL_PER_UNIT)
    protein_g: float = Field(ge=0, le=_MAX_GRAMS_PER_UNIT)
    carbs_g: float = Field(ge=0, le=_MAX_GRAMS_PER_UNIT)
    fat_g: float = Field(ge=0, le=_MAX_GRAMS_PER_UNIT)


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


# Units that describe a countable portion. A seed record in any of these is
# close enough to another (a "plate" of rice vs a "cup" is the same ballpark).
_PORTION_UNITS = {
    "", "serving", "servings", "portion", "portions", "helping", "plate", "plates",
    "piece", "pieces", "pc", "pcs", "slice", "slices", "egg", "eggs", "item", "items",
    "unit", "units", "cup", "cups", "bowl", "bowls", "glass", "glasses", "mug", "mugs",
    "medium", "small", "large", "whole", "scoop", "scoops", "wedge", "sprig", "handful",
    "can", "bottle", "packet", "box", "container", "roti", "chapati",
}
# Weight / volume measures. A seed record's portion macros can't stand in for one.
_MEASURE_UNITS = {
    "g", "gram", "grams", "gm", "gms", "kg", "kilogram", "kilograms",
    "ml", "milliliter", "millilitre", "milliliters", "millilitres",
    "l", "liter", "litre", "liters", "litres", "oz", "ounce", "ounces",
    "lb", "lbs", "pound", "pounds", "tbsp", "tablespoon", "tablespoons",
    "tsp", "teaspoon", "teaspoons",
}


def _normalize_unit(unit: str | None) -> str:
    return (unit or "serving").strip().lower()


def _seed_usable_for(requested_unit: str, seed_unit: str) -> bool:
    """True if the seed's per-portion macros are a reasonable answer for the
    requested unit. A weight/volume request (100 g, 250 ml) is NOT satisfied by
    a per-cup/per-piece seed value."""
    r, s = requested_unit, _normalize_unit(seed_unit)
    if r == s:
        return True
    if r in _MEASURE_UNITS:
        return False
    return r in _PORTION_UNITS  # both are countable portions - close enough


def resolve(name: str, unit_hint: str | None = None) -> NutritionEstimate:
    """Resolve one food's per-unit macros for the requested unit. Never raises -
    on total failure it returns a zeroed estimate with source 'failed'.

    A seed food wins on a name match ONLY when its portion unit is compatible
    with what was asked for; a weight/volume request falls through to a model
    estimate for that exact unit. Model estimates are cached per (name, unit).
    """
    key = normalize_name(name)
    unit = _normalize_unit(unit_hint)

    seeded = seed.lookup(key)
    if seeded and _seed_usable_for(unit, seeded["unit"]):
        repo.put_cached_nutrition(key, seeded, source="seed")
        return _from_macros(key, seeded, "seed", 1.0)

    cached = repo.get_cached_nutrition(key, unit)
    if cached:
        source = "seed" if cached["source"] == "seed" else "cache"
        return _from_macros(key, cached, source, 1.0 if source == "seed" else 0.7)

    estimate = _model_estimate(name, unit if unit != "serving" else None)
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
        unit=unit,
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
