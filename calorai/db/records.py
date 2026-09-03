"""Plain data classes for rows we pass around outside the DB layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from sqlite3 import Row


@dataclass
class User:
    id: str
    name: str
    timezone: str
    created_at: str

    @classmethod
    def from_row(cls, row: Row) -> "User":
        return cls(id=row["id"], name=row["name"], timezone=row["timezone"], created_at=row["created_at"])


@dataclass
class MealItem:
    id: str
    meal_id: str
    position: int
    name: str
    quantity: float
    unit: str
    kcal_per_unit: float
    protein_g_per_unit: float
    carbs_g_per_unit: float
    fat_g_per_unit: float
    confidence: float
    nutrition_source: str

    @classmethod
    def from_row(cls, row: Row) -> "MealItem":
        return cls(
            id=row["id"],
            meal_id=row["meal_id"],
            position=row["position"],
            name=row["name"],
            quantity=row["quantity"],
            unit=row["unit"],
            kcal_per_unit=row["kcal_per_unit"],
            protein_g_per_unit=row["protein_g_per_unit"],
            carbs_g_per_unit=row["carbs_g_per_unit"],
            fat_g_per_unit=row["fat_g_per_unit"],
            confidence=row["confidence"],
            nutrition_source=row["nutrition_source"],
        )

    @property
    def kcal(self) -> float:
        return self.quantity * self.kcal_per_unit

    @property
    def protein_g(self) -> float:
        return self.quantity * self.protein_g_per_unit

    @property
    def carbs_g(self) -> float:
        return self.quantity * self.carbs_g_per_unit

    @property
    def fat_g(self) -> float:
        return self.quantity * self.fat_g_per_unit


@dataclass
class Meal:
    id: str
    user_id: str
    eaten_at: str
    meal_date: str
    meal_type: str | None
    description: str
    source: str
    status: str
    created_at: str
    updated_at: str
    items: list[MealItem]

    @classmethod
    def from_row(cls, row: Row, items: list[MealItem]) -> "Meal":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            eaten_at=row["eaten_at"],
            meal_date=row["meal_date"],
            meal_type=row["meal_type"],
            description=row["description"],
            source=row["source"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            items=items,
        )

    @property
    def kcal(self) -> float:
        return sum(i.kcal for i in self.items)

    @property
    def protein_g(self) -> float:
        return sum(i.protein_g for i in self.items)

    @property
    def carbs_g(self) -> float:
        return sum(i.carbs_g for i in self.items)

    @property
    def fat_g(self) -> float:
        return sum(i.fat_g for i in self.items)


@dataclass
class MemoryRecord:
    id: str
    user_id: str
    type: str
    key: str
    content: str
    structured_value: dict | list | None
    meal_type: str | None
    status: str
    learned_via: str
    confidence: float
    source_turn_id: str | None
    use_count: int
    created_at: str
    updated_at: str
    last_used_at: str | None

    @classmethod
    def from_row(cls, row: Row) -> "MemoryRecord":
        raw = row["structured_value"]
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            type=row["type"],
            key=row["key"],
            content=row["content"],
            structured_value=json.loads(raw) if raw else None,
            meal_type=row["meal_type"],
            status=row["status"],
            learned_via=row["learned_via"],
            confidence=row["confidence"],
            source_turn_id=row["source_turn_id"],
            use_count=row["use_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_used_at=row["last_used_at"],
        )


@dataclass
class DailyTotals:
    date: str
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    meal_count: int

    def rounded(self) -> "DailyTotals":
        return DailyTotals(
            date=self.date,
            kcal=round(self.kcal),
            protein_g=round(self.protein_g, 1),
            carbs_g=round(self.carbs_g, 1),
            fat_g=round(self.fat_g, 1),
            meal_count=self.meal_count,
        )
