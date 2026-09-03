"""The CalorAI chat CLI.

    python cli.py                       interactive chat
    python cli.py --user usr_ab12cd34   resume a specific user
    python cli.py --message "had 2 rotis"   one-shot (used by the eval harness)
    python cli.py --message "..." --image plate.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel

from calorai.cli.commands import dispatch
from calorai.cli.onboarding import resolve_user
from calorai.config import get_settings
from calorai.db.database import init_db
from calorai.db.records import User


def _banner(console: Console) -> None:
    settings = get_settings()
    lines = [
        "[bold]CalorAI[/bold]  —  meal logging that texts back",
        f"[dim]text model {settings.text_model.provider}/{settings.text_model.model}"
        f"  ·  vision {settings.vision_model.provider}/{settings.vision_model.model}[/dim]",
    ]
    if settings.langsmith_tracing:
        lines.append(
            f"[dim]monitoring (for the walkthrough): LangSmith project "
            f"'{settings.langsmith_project}'  ·  https://smith.langchain.com[/dim]"
        )
    console.print(Panel("\n".join(lines), border_style="magenta", expand=False))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="calorai", description="CalorAI meal-logging agent")
    parser.add_argument("--user", help="resume a specific CalorAI user id")
    parser.add_argument("--message", help="send one message and exit (non-interactive)")
    parser.add_argument("--image", help="path to a food photo (with --message or in a chat turn)")
    parser.add_argument("--no-stream", action="store_true", help="print the full reply at once")
    return parser.parse_args(argv)


def _agent_reply(console: Console, user: User, text: str, image_path: str | None, stream: bool) -> None:
    from calorai.agent.runner import run_turn, stream_turn

    console.print()
    try:
        if not stream:
            with console.status("[dim]thinking…[/dim]", spinner="dots"):
                reply = run_turn(text, user_id=user.id, thread_id=user.id,
                                 timezone_name=user.timezone, image_path=image_path)
            console.print(f"[bold green]CalorAI[/bold green]  {reply}")
            return

        # Hold the spinner until the first chunk arrives, THEN print the label +
        # stream. (Printing the label before console.status lets the spinner's
        # live display swallow it.)
        chunks = stream_turn(text, user_id=user.id, thread_id=user.id,
                             timezone_name=user.timezone, image_path=image_path)
        first = None
        with console.status("[dim]thinking…[/dim]", spinner="dots"):
            for chunk in chunks:
                if chunk:
                    first = chunk
                    break

        console.print("[bold green]CalorAI[/bold green]  ", end="")
        if first is None:
            console.print("[dim](no reply)[/dim]")
            return
        console.print(first, end="")
        for chunk in chunks:
            console.print(chunk, end="")
        console.print()
    except Exception as exc:  # keep the REPL alive on any agent failure
        console.print(f"\n[red]Something went wrong: {exc}[/red]")
        console.print("[dim]Your data is safe. Try rephrasing, or /quit.[/dim]")


def _repl(console: Console, user: User, stream: bool) -> None:
    console.print("[dim]Type what you ate, or /help. /quit to exit.[/dim]\n")
    while True:
        try:
            line = console.input("[bold cyan]you[/bold cyan]  ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not line:
            continue
        if line.startswith("/"):
            parts = line.split(maxsplit=2)
            if parts[0].lower() in ("/img", "/image", "/photo"):
                if len(parts) < 2:
                    console.print("[yellow]usage: /img <path> [caption][/yellow]")
                    continue
                path = os.path.expanduser(parts[1])
                caption = parts[2] if len(parts) > 2 else ""
                if not os.path.isfile(path):
                    console.print(f"[yellow]no file at {path}[/yellow]")
                    continue
                _agent_reply(console, user, caption, path, stream)
                console.print()
                continue
            if not dispatch(console, user, line):
                return
            continue

        _agent_reply(console, user, line, None, stream)
        console.print()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()
    init_db()

    one_shot = args.message is not None or args.image is not None
    if not one_shot:
        _banner(console)

    user = resolve_user(console, args.user)

    if one_shot:
        if args.image and not os.path.isfile(os.path.expanduser(args.image)):
            console.print(f"[red]no file at {args.image}[/red]")
            return 1
        image = os.path.expanduser(args.image) if args.image else None
        _agent_reply(console, user, args.message or "", image, stream=not args.no_stream)
        console.print()
        return 0

    _repl(console, user, stream=not args.no_stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
