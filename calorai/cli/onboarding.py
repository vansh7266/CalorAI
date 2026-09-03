"""Figuring out which user is talking.

No auth. Identity is a generated id (``usr_xxxxxxxx``) that the user keeps to
resume later. Resolution order:

    1. an explicit id (``--user`` / ``CALORAI_USER``)
    2. the last id used on this machine (``data/.session``)
    3. first run: ask for a name, create a user, show them the id
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from calorai.config import SESSION_FILE, ensure_data_dir
from calorai.db import repositories as repo
from calorai.db.records import User


def detect_timezone() -> str:
    """Best-effort local IANA timezone name, falling back to UTC."""
    override = os.getenv("CALORAI_TZ")
    if override:
        return override
    try:
        link = Path("/etc/localtime").resolve()
        parts = link.parts
        if "zoneinfo" in parts:
            return "/".join(parts[parts.index("zoneinfo") + 1 :])
    except OSError:
        pass
    return "UTC"


def _read_last_user_id() -> str | None:
    try:
        return SESSION_FILE.read_text().strip() or None
    except OSError:
        return None


def remember_user(user_id: str) -> None:
    ensure_data_dir()
    try:
        SESSION_FILE.write_text(user_id)
    except OSError:
        pass


def _greet_returning(console: Console, user: User) -> None:
    today = repo.daily_totals(user.id, date.today().isoformat()).rounded()
    if today.meal_count:
        line = f"Today so far: {today.kcal} kcal, {today.protein_g} g protein across {today.meal_count} meal(s)."
    else:
        line = "Nothing logged yet today."
    console.print(
        Panel(f"Welcome back, [bold]{user.name}[/bold].\n{line}", border_style="green", title="CalorAI")
    )


def _first_run(console: Console) -> User:
    console.print(
        Panel(
            "Hi! I'm CalorAI. Tell me what you eat - text or a photo - and I'll keep track.\n"
            "New here? Just tell me your name. Already have a CalorAI id? Paste it instead.",
            border_style="green",
            title="CalorAI",
        )
    )
    answer = Prompt.ask("[bold]Your name (or CalorAI id)[/bold]", default="guest", console=console).strip()

    if answer.startswith("usr_"):
        existing = repo.get_user(answer)
        if existing:
            remember_user(existing.id)
            _greet_returning(console, existing)
            return existing
        console.print(f"[yellow]No user with id '{answer}'. Creating a new one.[/yellow]")
        answer = "guest"

    name = answer or "guest"
    user = repo.create_user(name, detect_timezone())
    console.print(
        Panel(
            f"Nice to meet you, [bold]{name}[/bold].\n"
            f"Your CalorAI id is [bold cyan]{user.id}[/bold cyan] - save it to resume on another session\n"
            f"with [dim]python cli.py --user {user.id}[/dim].",
            border_style="cyan",
        )
    )
    return user


def resolve_user(console: Console, explicit_id: str | None = None) -> User:
    """Return the active user, running first-run onboarding if needed."""
    requested = explicit_id or os.getenv("CALORAI_USER")
    if requested:
        user = repo.get_user(requested.strip())
        if user:
            remember_user(user.id)
            _greet_returning(console, user)
            return user
        console.print(f"[yellow]No user found with id '{requested}'. Starting fresh.[/yellow]")

    last = _read_last_user_id()
    if last and not requested:
        user = repo.get_user(last)
        if user:
            _greet_returning(console, user)
            return user

    user = _first_run(console)
    remember_user(user.id)
    return user
