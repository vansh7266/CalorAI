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
from rich.markup import escape
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


def _parse_img_command(line: str) -> tuple[str | None, str]:
    """`/img <path> [caption]`. Handles quoted paths and unquoted paths with
    spaces: try progressively shorter prefixes of the remainder until one is a
    real file; the rest is the caption."""
    import shlex

    rest = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else ""
    if not rest:
        return None, ""

    if rest[0] in "'\"":
        try:
            parts = shlex.split(rest)
        except ValueError:
            parts = [rest]
        path = os.path.expanduser(parts[0])
        caption = " ".join(parts[1:])
        return (path, caption) if os.path.isfile(path) else (None, "")

    words = rest.split(" ")
    for cut in range(len(words), 0, -1):
        candidate = os.path.expanduser(" ".join(words[:cut]))
        if os.path.isfile(candidate):
            return candidate, " ".join(words[cut:])
    return None, ""


def _agent_reply(console: Console, user: User, text: str, image_path: str | None, stream: bool) -> bool:
    """Run a turn and print the reply. Returns False if the agent reported failure."""
    from calorai.agent.runner import GENERIC_ERROR, run_turn, stream_turn

    console.print()
    try:
        if not stream:
            with console.status("[dim]thinking…[/dim]", spinner="dots"):
                reply = run_turn(text, user_id=user.id, timezone_name=user.timezone, image_path=image_path)
            console.print("[bold green]CalorAI[/bold green]  ", end="")
            console.print(reply, markup=False)
            return reply != GENERIC_ERROR

        # Hold the spinner until the first chunk arrives, THEN print the label +
        # stream. (Printing the label before console.status lets the spinner's
        # live display swallow it.)
        chunks = stream_turn(text, user_id=user.id, timezone_name=user.timezone, image_path=image_path)
        first = None
        with console.status("[dim]thinking…[/dim]", spinner="dots"):
            for chunk in chunks:
                if chunk:
                    first = chunk
                    break

        console.print("[bold green]CalorAI[/bold green]  ", end="")
        if first is None:
            console.print("[dim](no reply)[/dim]")
            return True
        console.print(first, end="", markup=False)
        collected = first
        for chunk in chunks:
            collected += chunk
            console.print(chunk, end="", markup=False)
        console.print()
        return collected.strip() != GENERIC_ERROR
    except Exception:
        console.print("\n[red]Something went wrong. Your data is safe - try rephrasing, or /quit.[/red]")
        return False


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
            if line[1:].split(maxsplit=1)[0].lower() in ("img", "image", "photo"):
                path, caption = _parse_img_command(line)
                if path is None:
                    console.print(
                        "usage: /img PATH  - add an optional caption after the path; "
                        "quote paths that contain spaces",
                        style="yellow",
                        markup=False,
                    )
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

    user = resolve_user(console, args.user, non_interactive=one_shot)

    if one_shot:
        if args.image and not os.path.isfile(os.path.expanduser(args.image)):
            console.print(f"[red]no file at {escape(args.image)}[/red]")
            return 1
        image = os.path.expanduser(args.image) if args.image else None
        ok = _agent_reply(console, user, args.message or "", image, stream=not args.no_stream)
        console.print()
        return 0 if ok else 1

    _repl(console, user, stream=not args.no_stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
