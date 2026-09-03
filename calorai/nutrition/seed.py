"""Seed nutrition table.

A small set of common foods (with a bias toward the ones in the test set and
typical Indian meals) so the first lookup for these is instant and stable.
Anything not here falls through to a model estimate, which is then cached.

Values are per one unit of the food, taken from standard nutrition references
and rounded. Accuracy is deliberately "good enough" - the task is about the
agent, not the database.
"""

from __future__ import annotations

# name -> (unit, kcal, protein_g, carbs_g, fat_g)  [all per one unit]
_ROWS: dict[str, tuple[str, float, float, float, float]] = {
    # breads
    "roti": ("piece", 105, 3.0, 18.0, 2.5),
    "plain paratha": ("piece", 180, 4.0, 24.0, 8.0),
    "aloo paratha": ("piece", 240, 6.0, 32.0, 10.0),
    "paneer paratha": ("piece", 280, 11.0, 28.0, 14.0),
    "gobi paratha": ("piece", 220, 5.0, 30.0, 9.0),
    "naan": ("piece", 260, 8.0, 45.0, 5.0),
    "white bread": ("slice", 75, 2.5, 13.0, 1.0),
    "toast": ("slice", 80, 2.5, 13.0, 1.5),
    # rice / grains
    "white rice": ("cup", 200, 4.0, 44.0, 0.5),
    "jeera rice": ("cup", 240, 4.0, 45.0, 5.0),
    "chicken biryani": ("cup", 290, 14.0, 33.0, 11.0),
    "veg biryani": ("cup", 240, 6.0, 38.0, 8.0),
    "poha": ("bowl", 180, 3.0, 33.0, 4.0),
    "upma": ("bowl", 200, 4.0, 30.0, 7.0),
    # dals / curries
    "dal": ("bowl", 180, 9.0, 22.0, 6.0),
    "rajma": ("bowl", 210, 10.0, 30.0, 5.0),
    "chole": ("bowl", 230, 10.0, 32.0, 7.0),
    "paneer curry": ("bowl", 280, 12.0, 12.0, 20.0),
    "mixed veg sabzi": ("bowl", 140, 4.0, 14.0, 8.0),
    "chicken curry": ("bowl", 240, 20.0, 8.0, 15.0),
    # south indian
    "idli": ("piece", 58, 2.0, 12.0, 0.4),
    "plain dosa": ("piece", 165, 3.0, 25.0, 5.0),
    "masala dosa": ("piece", 250, 5.0, 37.0, 9.0),
    "sambar": ("bowl", 120, 6.0, 16.0, 3.0),
    # eggs
    "boiled egg": ("egg", 78, 6.0, 0.6, 5.0),
    "fried egg": ("egg", 90, 6.0, 0.4, 7.0),
    "omelette": ("serving", 220, 13.0, 2.0, 17.0),
    # drinks
    "chai": ("cup", 90, 2.5, 12.0, 3.5),
    "black coffee": ("cup", 5, 0.3, 0.8, 0.0),
    "coffee": ("cup", 60, 2.5, 8.0, 2.5),
    "milk": ("cup", 150, 8.0, 12.0, 8.0),
    "lassi": ("glass", 180, 6.0, 28.0, 5.0),
    "orange juice": ("glass", 110, 1.7, 26.0, 0.5),
    # snacks / sides / fruit
    "samosa": ("piece", 130, 3.0, 15.0, 7.0),
    "curd": ("bowl", 90, 5.0, 7.0, 5.0),
    "green salad": ("bowl", 40, 2.0, 8.0, 0.5),
    "banana": ("medium", 105, 1.3, 27.0, 0.4),
    "apple": ("medium", 95, 0.5, 25.0, 0.3),
    "butter": ("tsp", 34, 0.0, 0.0, 3.9),
}

# alias -> canonical name in _ROWS
_ALIASES: dict[str, str] = {
    "chapati": "roti",
    "phulka": "roti",
    "roti/chapati": "roti",
    "paratha": "plain paratha",
    "parantha": "plain paratha",
    "aloo parantha": "aloo paratha",
    "tea": "chai",
    "milk tea": "chai",
    "biryani": "chicken biryani",
    "rice": "white rice",
    "steamed rice": "white rice",
    "boiled rice": "white rice",
    "egg": "boiled egg",
    "eggs": "boiled egg",
    "dosa": "plain dosa",
    "yogurt": "curd",
    "dahi": "curd",
    "salad": "green salad",
    "bread": "white bread",
    "toasted bread": "toast",
}


def _canonical(name: str) -> str | None:
    """Map a normalized name to a key in _ROWS, trying an alias and a simple
    plural ('rotis' -> 'roti') before giving up."""
    if name in _ROWS:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    if name.endswith("s"):
        singular = name[:-1]
        if singular in _ROWS:
            return singular
        if singular in _ALIASES:
            return _ALIASES[singular]
    return None


def lookup(name: str) -> dict | None:
    """Return per-unit macros for a seed food, or None. `name` should already be
    normalized (lower-case, trimmed)."""
    key = _canonical(name)
    row = _ROWS.get(key) if key else None
    if row is None:
        return None
    unit, kcal, protein, carbs, fat = row
    return {
        "unit": unit,
        "kcal_per_unit": kcal,
        "protein_g_per_unit": protein,
        "carbs_g_per_unit": carbs,
        "fat_g_per_unit": fat,
    }


def all_names() -> list[str]:
    return sorted(set(_ROWS) | set(_ALIASES))
