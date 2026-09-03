"""Persistent memory: what the agent remembers about a user across sessions.

Three pieces:

* the profile card - a short always-in-context block (diet, goals, a few
  preferences). Routines are NOT in the card; they are fetched on demand.
* two tools - `save_memory` (the agent writes a durable fact) and
  `recall_memory` (the agent looks one up, e.g. "my usual").
* the reflection pass - after the reply is sent, a cheap model call re-reads the
  exchange and saves anything the agent's explicit `save_memory` missed. It runs
  in a background thread so it adds nothing to response time.

Nothing here stores conversation history. Only durable facts and routines.
"""

from __future__ import annotations

import atexit
import os
import re
import threading
from datetime import date, timedelta

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from calorai.agent.context import get_context
from calorai.db import repositories as repo
from calorai.db.records import MemoryRecord
from calorai.models.gateway import as_structured, get_worker_model

_CARD_TYPES = ["diet", "goal", "preference"]
_CARD_LIMIT = 6



# --- profile card (tier 1) ---


def render_profile_card(user_id: str) -> str:
    lines: list[str] = []
    for r in repo.get_active_memory(user_id, types=_CARD_TYPES)[:_CARD_LIMIT]:
        prefix = "" if r.learned_via != "inferred" else "(unconfirmed) "
        lines.append(f"- {prefix}{r.content}")

    hint = _routine_suggestion(user_id)
    if hint:
        lines.append(hint)

    return "\n".join(lines)


def _routine_suggestion(user_id: str) -> str | None:
    """If the user keeps logging the same thing for a meal and has no routine
    saved for it, nudge the agent to offer to remember it."""
    ctx = get_context()
    start = (date.fromisoformat(ctx.local_date) - timedelta(days=7)).isoformat()
    recent = repo.get_meals_between(user_id, start, ctx.local_date)

    signatures_by_type: dict[str, list[frozenset[str]]] = {}
    for meal in recent:
        if not meal.meal_type or not meal.items:
            continue
        signature = frozenset(i.name for i in meal.items)
        signatures_by_type.setdefault(meal.meal_type, []).append(signature)

    have_routine_for = {r.meal_type for r in repo.get_active_memory(user_id, types=["routine"])}

    for meal_type, signatures in signatures_by_type.items():
        if meal_type in have_routine_for:
            continue
        for signature in set(signatures):
            if signatures.count(signature) >= 3:
                foods = ", ".join(sorted(signature))
                return (
                    f"Note: they've had the same {meal_type} ({foods}) 3+ times lately with no "
                    f"saved routine - you could offer to save it as their usual {meal_type}."
                )
    return None


# --- tools ---


class RoutineItem(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str = "serving"


def _derive_key(kind: str, content: str, meal_type: str | None) -> str:
    text = content.lower()
    if kind == "diet":
        return "diet"
    if kind == "goal":
        if "protein" in text:
            return "protein_target"
        if "calorie" in text or "kcal" in text or "cal " in text:
            return "calorie_target"
        return "goal"
    if kind == "routine":
        return f"usual_{meal_type}" if meal_type else "usual"
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return "_".join(slug.split("_")[:4]) or kind


@tool
def save_memory(
    kind: str,
    content: str,
    key: str | None = None,
    structured_value: dict | None = None,
    routine_items: list[RoutineItem] | None = None,
    meal_type: str | None = None,
) -> dict:
    """Remember a durable fact about the user, across sessions.

    kind: diet | goal | preference | routine | fact
    content: short natural language ("vegetarian, no eggs", "targets 140 g protein/day",
             "'my usual' = 2 eggs, 2 toast, black coffee")
    structured_value: for a goal, e.g. {"protein_g": 140}
    routine_items: for kind=routine, the foods that make up the routine
    meal_type: for a routine tied to a time of day (breakfast/lunch/dinner/snack)

    Save: stated diet or allergies, explicit targets, "my usual is ...", lasting
    likes/dislikes. Do NOT save one-off meals, passing states, or questions.
    """
    try:
        ctx = get_context()
        kind = kind.strip().lower()
        if kind not in {"diet", "goal", "preference", "routine", "fact"}:
            kind = "fact"
        payload = structured_value
        if routine_items:
            payload = {"items": [i.model_dump() for i in routine_items]}
        record = repo.upsert_memory(
            user_id=ctx.user_id,
            type=kind,
            key=key or _derive_key(kind, content, meal_type),
            content=content,
            structured_value=payload,
            meal_type=meal_type,
            learned_via="stated",
            confidence=1.0,
            source_turn_id=ctx.turn_id,
        )
        return {"saved": True, "memory_id": record.id, "kind": kind, "content": content}
    except Exception:  # pragma: no cover - defensive
        return {"error": "could not save that right now"}


@tool
def recall_memory(query: str) -> dict:
    """Look up something the user told you earlier - a routine ('my usual'), a
    preference, a goal, a fact. Call this whenever the user refers to something
    they may have mentioned before."""
    try:
        ctx = get_context()
        hits: list[MemoryRecord] = repo.search_memory(ctx.user_id, query)

        wants_routine = any(w in query.lower() for w in ("usual", "routine", "always", "normal", "same"))
        if wants_routine:
            known_ids = {h.id for h in hits}
            hits += [r for r in repo.get_active_memory(ctx.user_id, types=["routine"]) if r.id not in known_ids]

        for h in hits:
            repo.bump_memory_use(h.id)

        return {
            "found": len(hits),
            "memories": [
                {
                    "memory_id": h.id,
                    "kind": h.type,
                    "content": h.content,
                    "structured_value": h.structured_value,
                    "meal_type": h.meal_type,
                }
                for h in hits
            ],
        }
    except Exception:  # pragma: no cover - defensive
        return {"error": "could not recall that right now"}


MEMORY_TOOLS = [save_memory, recall_memory]


# --- reflection pass ---


class _Reflection(BaseModel):
    should_save: bool = Field(description="true only if the user revealed something durable and new")
    kind: str | None = Field(default=None, description="diet | goal | preference | routine | fact")
    key: str | None = None
    content: str | None = Field(default=None, description="the fact, in short natural language")


_REFLECT_PROMPT = """\
You maintain a small long-term memory about a nutrition-app user.

Already known about them:
{known}

Latest exchange:
User: {user_text}
Assistant: {agent_reply}

Did the USER reveal a DURABLE fact that is NOT already known - a diet or allergy,
a nutrition goal/target, a lasting preference, or a routine ("my usual is ...")?

Rules:
- Ignore one-off meals, passing states ("I'm full", "skipped lunch today"), and questions.
- Only report something genuinely new. If nothing, set should_save=false.
"""


def _run_reflection_model(prompt: str) -> _Reflection:
    """The one live call. Seam for tests to stub."""
    return as_structured(get_worker_model(), _Reflection).invoke(prompt)


def run_reflection(user_id: str, turn_id: str, user_text: str, agent_reply: str) -> MemoryRecord | None:
    known_rows = repo.get_active_memory(user_id)
    known = "\n".join(f"- ({r.type}) {r.content}" for r in known_rows) or "(nothing yet)"
    prompt = _REFLECT_PROMPT.format(known=known, user_text=user_text, agent_reply=agent_reply)

    try:
        result = _run_reflection_model(prompt)
    except Exception:
        return None

    if not result.should_save or not result.content:
        return None

    kind = (result.kind or "fact").strip().lower()
    if kind not in {"diet", "goal", "preference", "routine", "fact"}:
        kind = "fact"

    try:
        # Reflection is a model *inference* - store it as inferred so the profile
        # card marks it "(unconfirmed)" and the agent double-checks before relying
        # on it. An explicit save_memory tool call is what marks something "stated".
        return repo.upsert_memory(
            user_id=user_id,
            type=kind,
            key=result.key or _derive_key(kind, result.content, None),
            content=result.content,
            learned_via="inferred",
            confidence=0.7,
            source_turn_id=turn_id,
        )
    except Exception:
        return None


_inflight: set[threading.Thread] = set()


@atexit.register
def _drain_reflections() -> None:
    """Give in-flight reflection writes a short window to finish on exit, so a
    valid memory isn't lost when the user quits right after telling us something."""
    for t in list(_inflight):
        t.join(timeout=3.0)


def schedule_reflection(user_id: str, turn_id: str, user_text: str, agent_reply: str) -> None:
    """Run the reflection pass off the response path, on a daemon thread so it
    never delays the reply. Set CALORAI_SYNC_REFLECTION=1 to run it inline (used
    by tests and the eval harness)."""
    if os.getenv("CALORAI_SYNC_REFLECTION") in ("1", "true", "yes"):
        run_reflection(user_id, turn_id, user_text, agent_reply)
        return

    def _safe() -> None:
        try:
            run_reflection(user_id, turn_id, user_text, agent_reply)
        except Exception:
            pass
        finally:
            _inflight.discard(threading.current_thread())

    thread = threading.Thread(target=_safe, name="calorai-reflect", daemon=True)
    _inflight.add(thread)
    thread.start()
