"""Assertions for eval cases.

"Correct" for this task = the six rules in the README:
  1. extraction   - the right items and quantities land in the DB
  2. totals       - daily totals are right after every edit
  3. no double-count on a correction
  4. memory       - stored when it should be, and used in the reply
  5. ambiguity    - asks only when it should; never invents a meal
  6. one meal per real meal

Each `expect` key below maps to one of those. Most checks read the DB directly;
a couple read the reply text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from calorai.agent.context import TurnContext
from calorai.db import repositories as repo
from calorai.db.records import User
from datetime import datetime, timezone


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GradeResult:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name, passed, detail))


def _today(user: User) -> str:
    ctx = TurnContext(user.id, "eval", user.timezone, datetime.now(timezone.utc))
    return ctx.local_date


def _todays_items(user: User) -> list:
    items = []
    for meal in repo.get_meals_for_date(user.id, _today(user)):
        items.extend(meal.items)
    return items


def grade(expect: dict, user: User, reply: str) -> GradeResult:
    result = GradeResult()
    reply_l = reply.lower()
    day = _today(user)
    todays_meals = repo.get_meals_for_date(user.id, day)
    todays_items = _todays_items(user)
    item_names = [i.name.lower() for i in todays_items]

    if "meals_today" in expect:
        want = expect["meals_today"]
        result.add(f"meals_today == {want}", len(todays_meals) == want, f"got {len(todays_meals)}")

    if "has_items" in expect:
        for name in expect["has_items"]:
            hit = any(name.lower() in n for n in item_names)
            result.add(f"logged an item matching '{name}'", hit, f"items={item_names}")

    if "no_item_matching" in expect:
        for name in expect["no_item_matching"]:
            absent = not any(name.lower() in n for n in item_names)
            result.add(f"no item matching '{name}'", absent, f"items={item_names}")

    if "item_quantity" in expect:
        for name, qty in expect["item_quantity"].items():
            match = next((i for i in todays_items if name.lower() in i.name.lower()), None)
            ok = match is not None and abs(match.quantity - qty) < 1e-6
            result.add(f"'{name}' quantity == {qty}", ok, f"got {match.quantity if match else 'missing'}")

    if "total_kcal_between" in expect:
        lo, hi = expect["total_kcal_between"]
        kcal = repo.daily_totals(user.id, day).kcal
        result.add(f"total kcal in [{lo}, {hi}]", lo <= kcal <= hi, f"got {round(kcal)}")

    if "meal_type" in expect:
        for _, mtype in expect["meal_type"].items():
            hit = any(m.meal_type == mtype for m in todays_meals)
            result.add(f"a meal is {mtype}", hit, f"types={[m.meal_type for m in todays_meals]}")

    if "memory_has" in expect:
        spec = expect["memory_has"]
        rows = repo.get_active_memory(user.id, types=[spec["type"]] if "type" in spec else None)
        hit = any(spec["contains"].lower() in r.content.lower() for r in rows)
        result.add(
            f"memory ({spec.get('type', 'any')}) contains '{spec['contains']}'",
            hit,
            f"memories={[r.content for r in rows]}",
        )

    if "reply_contains_any" in expect:
        hits = [s for s in expect["reply_contains_any"] if s.lower() in reply_l]
        result.add(f"reply contains one of {expect['reply_contains_any']}", bool(hits), f"reply={reply!r}")

    if expect.get("reply_asks"):
        offer_phrases = (
            "let me know", "want me to", "could you", "what did", "tell me", "if you can",
            "give me a", "rough idea", "rough estimate", "put a number", "skip it", "otherwise",
            "do you", "did you", "what do you", "would you like", "happy to",
        )
        asks = "?" in reply or any(p in reply_l for p in offer_phrases)
        result.add("reply asks / offers rather than assuming", asks, f"reply={reply!r}")

    return result
