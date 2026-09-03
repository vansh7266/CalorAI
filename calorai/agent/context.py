"""Per-turn context.

Tools only take arguments the model should fill. Everything else they need -
which user, which turn, what "today" means in the user's timezone - comes from
a context object set once at the start of each turn and read via `get_context()`.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_current: ContextVar["TurnContext | None"] = ContextVar("calorai_turn_context", default=None)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class TurnContext:
    user_id: str
    turn_id: str
    timezone_name: str
    now_utc: datetime

    def _tz(self) -> timezone | ZoneInfo:
        try:
            return ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return timezone.utc

    @property
    def local_now(self) -> datetime:
        return self.now_utc.astimezone(self._tz())

    @property
    def local_date(self) -> str:
        return self.local_now.date().isoformat()

    def resolve_date(self, reference: str | None) -> str:
        """Turn 'today' / 'yesterday' / a weekday / an ISO date into YYYY-MM-DD
        in the user's local timezone. Unknown input falls back to today."""
        if not reference:
            return self.local_date
        ref = reference.strip().lower()
        if ref in ("today", "now", "tonight", "this morning", "this afternoon", "this evening"):
            return self.local_date
        if ref in ("yesterday", "last night"):
            return (self.local_now.date() - timedelta(days=1)).isoformat()
        if ref in ("day before yesterday", "2 days ago"):
            return (self.local_now.date() - timedelta(days=2)).isoformat()
        if _ISO_DATE.match(ref):
            return ref
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        if ref in weekdays:
            today = self.local_now.date()
            delta = (today.weekday() - weekdays.index(ref)) % 7 or 7
            return (today - timedelta(days=delta)).isoformat()
        return self.local_date

    def meal_type_for_now(self) -> str:
        hour = self.local_now.hour
        if hour < 11:
            return "breakfast"
        if hour < 16:
            return "lunch"
        if hour < 22:
            return "dinner"
        return "snack"


def new_turn_id() -> str:
    import secrets

    return f"turn_{secrets.token_hex(4)}"


def set_context(ctx: TurnContext):
    return _current.set(ctx)


def reset_context(token) -> None:
    _current.reset(token)


def get_context() -> TurnContext:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("No TurnContext is set. A tool was called outside of an agent turn.")
    return ctx
