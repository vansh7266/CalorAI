"""Slash commands for the CLI.

Each handler takes (console, user, arg_string) and returns True to keep the REPL
running or False to exit. Anything not starting with '/' is a message for the
agent and never reaches here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from calorai.agent.context import TurnContext
from calorai.db import repositories as repo
from calorai.db.records import User

HELP_TEXT = """\
Just type what you ate and I'll log it. Commands:

  /totals [date]     calories + macros for today (or a given day)
  /history [date]    recent meals, or every meal on a day
  /memory            what I've remembered about you
  /forget <id>       make me forget one memory
  /whoami            your CalorAI id
  /help              this message
  /quit              exit

Dates accept 'today', 'yesterday', a weekday, or YYYY-MM-DD.\
"""


def _local_date(user: User, reference: str | None) -> str:
    ctx = TurnContext(user.id, "cli", user.timezone, datetime.now(timezone.utc))
    return ctx.resolve_date(reference)


def cmd_help(console: Console, user: User, arg: str) -> bool:
    console.print(HELP_TEXT)
    return True


def cmd_quit(console: Console, user: User, arg: str) -> bool:
    console.print("[dim]bye[/dim]")
    return False


def cmd_whoami(console: Console, user: User, arg: str) -> bool:
    console.print(f"[bold]{user.name}[/bold] - id [cyan]{user.id}[/cyan] - timezone {user.timezone}")
    return True


def cmd_totals(console: Console, user: User, arg: str) -> bool:
    day = _local_date(user, arg.strip() or None)
    t = repo.daily_totals(user.id, day).rounded()
    table = Table(title=f"Totals for {day}", show_header=False, border_style="dim")
    table.add_row("Calories", f"{t.kcal} kcal")
    table.add_row("Protein", f"{t.protein_g} g")
    table.add_row("Carbs", f"{t.carbs_g} g")
    table.add_row("Fat", f"{t.fat_g} g")
    table.add_row("Meals", str(t.meal_count))
    console.print(table)
    return True


def cmd_history(console: Console, user: User, arg: str) -> bool:
    if arg.strip():
        meals = repo.get_meals_for_date(user.id, _local_date(user, arg.strip()))
        heading = f"Meals on {_local_date(user, arg.strip())}"
    else:
        meals = repo.get_recent_meals(user.id, limit=10)
        heading = "Recent meals"

    if not meals:
        console.print("[dim]nothing logged[/dim]")
        return True

    table = Table(title=heading, border_style="dim")
    table.add_column("when")
    table.add_column("meal")
    table.add_column("items")
    table.add_column("kcal", justify="right")
    for m in meals:
        items = ", ".join(
            f"{int(i.quantity) if float(i.quantity).is_integer() else i.quantity} {i.name}" for i in m.items
        )
        table.add_row(m.meal_date, m.meal_type or "-", items or m.description, str(round(m.kcal)))
    console.print(table)
    return True


def cmd_memory(console: Console, user: User, arg: str) -> bool:
    rows = repo.get_active_memory(user.id)
    if not rows:
        console.print("[dim]I haven't remembered anything about you yet.[/dim]")
        return True
    table = Table(title="What I remember", border_style="dim")
    table.add_column("id")
    table.add_column("type")
    table.add_column("about")
    table.add_column("learned")
    for r in rows:
        table.add_row(r.id, r.type, r.content, r.learned_via)
    console.print(table)
    return True


def cmd_forget(console: Console, user: User, arg: str) -> bool:
    mem_id = arg.strip()
    if not mem_id:
        console.print("[yellow]usage: /forget <memory id>  (see /memory)[/yellow]")
        return True
    row = repo.get_memory(mem_id)
    if not row or row.user_id != user.id:
        console.print(f"[yellow]no memory with id {mem_id}[/yellow]")
        return True
    repo.deactivate_memory(mem_id)
    console.print(f"[green]forgotten:[/green] {row.content}")
    return True


COMMANDS: dict[str, Callable[[Console, User, str], bool]] = {
    "help": cmd_help,
    "quit": cmd_quit,
    "exit": cmd_quit,
    "whoami": cmd_whoami,
    "totals": cmd_totals,
    "history": cmd_history,
    "memory": cmd_memory,
    "forget": cmd_forget,
}


def dispatch(console: Console, user: User, line: str) -> bool:
    parts = line[1:].strip().split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    handler = COMMANDS.get(name)
    if handler is None:
        console.print(f"[yellow]unknown command '/{name}'. try /help[/yellow]")
        return True
    return handler(console, user, arg)
