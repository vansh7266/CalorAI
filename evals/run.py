"""Run the eval set.

    python evals/run.py                # all cases
    python evals/run.py correction     # only cases whose id/category contains "correction"

Each case gets a fresh SQLite database. Reflection runs inline so results are
deterministic. Prints a pass/fail table and exits non-zero if anything failed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CALORAI_SYNC_REFLECTION", "1")

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calorai.db import database, repositories as repo  # noqa: E402
from evals.graders import grade  # noqa: E402

CASES_FILE = Path(__file__).with_name("cases.yaml")
console = Console()


def _fresh_db(tmp_dir: Path, case_id: str) -> None:
    db_path = tmp_dir / f"{case_id}.db"
    database.DB_PATH = db_path
    database._schema_applied = False
    database.init_db(db_path)


def _run_case(case: dict) -> tuple[bool, list, float, str]:
    from calorai.agent.runner import run_turn

    user = repo.create_user("eval", "UTC")
    thread = user.id

    start = time.time()
    for msg in case.get("setup", []):
        run_turn(msg, user_id=user.id, thread_id=thread, timezone_name="UTC")

    image = case.get("image")
    image_path = str(PROJECT_ROOT / image) if image else None
    message = case.get("caption", "") if image else case["message"]
    reply = run_turn(message, user_id=user.id, thread_id=thread, timezone_name="UTC", image_path=image_path)
    elapsed = time.time() - start

    result = grade(case["expect"], user, reply)
    return result.passed, result.checks, elapsed, reply


def main() -> int:
    cases = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))
    flt = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if flt:
        cases = [c for c in cases if flt in c["id"].lower() or flt in c["category"].lower()]

    tmp_dir = PROJECT_ROOT / "evals" / "results"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title=f"CalorAI evals ({len(cases)} cases)")
    table.add_column("case")
    table.add_column("category")
    table.add_column("result")
    table.add_column("time", justify="right")
    table.add_column("failing checks")

    passed_count = 0
    for case in cases:
        _fresh_db(tmp_dir, case["id"])
        try:
            ok, checks, elapsed, reply = _run_case(case)
        except Exception as exc:  # a case blowing up is itself a failure
            table.add_row(case["id"], case["category"], "[red]ERROR[/red]", "-", str(exc)[:60])
            continue

        fails = [c.name for c in checks if not c.passed]
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        passed_count += ok
        table.add_row(
            case["id"],
            case["category"],
            status,
            f"{elapsed:.1f}s",
            "" if ok else "; ".join(f"{n}" for n in fails)[:80],
        )
        if not ok:
            console.print(f"  [dim]{case['id']} reply:[/dim] {reply}")

    console.print(table)
    console.print(f"\n{passed_count}/{len(cases)} cases passed")
    return 0 if passed_count == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
