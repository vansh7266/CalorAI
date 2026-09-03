"""Compare model configs on the task's own test conversation.

    python benchmarks/model_compare.py            # run all configs, print a table
    python benchmarks/model_compare.py sarvam     # one config, prints JSON (worker mode)

Each config runs in its own subprocess with its own env and database, so there is
no state bleed between them. Reports latency, token usage, and whether the data
landed correctly for the 11 messages from the brief (plus 3 setup lines so "same
as yesterday" / "my usual" / the correction have something to act on).

Configs:
  1. sarvam       - GLM-5.2 brain + GLM-5.2 worker            (current default)
  2. groq-worker  - GLM-5.2 brain + Groq gpt-oss-120b worker  (reflection/nutrition on Groq)
  3. groq-brain   - Groq gpt-oss-120b brain + Sarvam Gemma 4 vision

Needs GROQ_API_KEY in .env for configs 2 and 3. (Groq's own vision models are
skipped - their names churn and the free tier rate-limits a full conversation.)
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROQ_TEXT_MODEL = "openai/gpt-oss-120b"

SETUP = [
    "had rajma and rice for lunch yesterday",
    "my usual breakfast is 2 idlis and a filter coffee",
    "had 2 rotis as an evening snack",
]

SCRIPT = [
    {"msg": "had 2 parathas and chai for breakfast"},
    {"msg": "leftover biryani, maybe two thirds of the box"},
    {"msg": "skipped lunch but grazed all afternoon"},
    {"msg": "same as yesterday for lunch"},
    {"msg": "actually that was 3 rotis not 2"},
    {"msg": "how much protein have I had today?"},
    {"msg": "how am I doing on calories?"},
    {"msg": "", "image": "samples/thali.jpeg"},
    {"msg": "half of this was my brother's", "image": "samples/pork_plate.jpg"},
    {"msg": "my usual"},
    {"msg": "i'm vegetarian btw"},
]

CONFIGS = {
    "sarvam": {"MODEL_PROFILE": "sarvam"},
    "groq-worker": {"MODEL_PROFILE": "sarvam", "WORKER_PROVIDER": "groq", "WORKER_MODEL": GROQ_TEXT_MODEL},
    "groq-brain": {
        "MODEL_PROFILE": "sarvam",           # keep Sarvam Gemma 4 for vision
        "TEXT_PROVIDER": "groq",
        "TEXT_MODEL": GROQ_TEXT_MODEL,
        "WORKER_PROVIDER": "groq",
        "WORKER_MODEL": GROQ_TEXT_MODEL,
    },
}


# --------------------------------------------------------------------------- #
# worker mode: run one config, print JSON
# --------------------------------------------------------------------------- #

def _worker(config_name: str) -> None:
    from datetime import datetime, timezone

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    for extra in ("MODEL_PROFILE", "TEXT_MODEL", "VISION_MODEL", "TEXT_PROVIDER",
                  "VISION_PROVIDER", "WORKER_PROVIDER", "WORKER_MODEL"):
        os.environ.pop(extra, None)
    os.environ.update(CONFIGS[config_name])
    os.environ["CALORAI_SYNC_REFLECTION"] = "1"

    db = PROJECT_ROOT / "benchmarks" / "results" / f"cmp_{config_name}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        p = db.with_name(db.name + suffix)
        if p.exists():
            p.unlink()
    os.environ["CALORAI_DB_PATH"] = str(db)

    sys.path.insert(0, str(PROJECT_ROOT))
    from langchain_core.callbacks import BaseCallbackHandler

    from calorai.agent.context import TurnContext, new_turn_id, reset_context, set_context
    from calorai.agent.graph import build_app
    from calorai.agent.runner import _initial_state
    from calorai.db import repositories as repo

    class Tokens(BaseCallbackHandler):
        def __init__(self) -> None:
            self.i = self.o = self.n = 0

        def on_llm_end(self, response, **kwargs):  # type: ignore[override]
            self.n += 1
            try:
                usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
                self.i += usage.get("input_tokens", 0)
                self.o += usage.get("output_tokens", 0)
            except Exception:
                tu = (response.llm_output or {}).get("token_usage", {})
                self.i += tu.get("prompt_tokens", 0)
                self.o += tu.get("completion_tokens", 0)

    app = build_app()
    user = repo.create_user("cmp", "Asia/Kolkata")
    tokens = Tokens()
    cfg = {"configurable": {"thread_id": user.id}, "callbacks": [tokens]}

    def turn(text: str, image: str | None) -> float:
        ctx = TurnContext(user.id, new_turn_id(), "Asia/Kolkata", datetime.now(timezone.utc))
        tok = set_context(ctx)
        start = time.time()
        try:
            app.invoke(_initial_state(text, user_id=user.id, turn_id=ctx.turn_id,
                                      timezone_name="Asia/Kolkata", image_path=image), cfg)
            return time.time() - start
        finally:
            reset_context(tok)

    for msg in SETUP:
        turn(msg, None)
    tokens.i = tokens.o = tokens.n = 0

    per_turn, errors = [], 0
    for step in SCRIPT:
        image = str(PROJECT_ROOT / step["image"]) if step.get("image") else None
        try:
            per_turn.append(turn(step["msg"], image))
        except Exception:
            errors += 1
            per_turn.append(0.0)

    day = TurnContext(user.id, "x", "Asia/Kolkata", datetime.now(timezone.utc)).local_date
    meals = repo.get_meals_for_date(user.id, day)
    yday = repo.get_meals_between(
        user.id, (datetime.now(timezone.utc).astimezone().date().replace(day=1)).isoformat(), day
    )
    totals = repo.daily_totals(user.id, day).rounded()

    print(json.dumps({
        "config": config_name,
        "errors": errors,
        "p50": statistics.median(per_turn),
        "p95": sorted(per_turn)[int(len(per_turn) * 0.95)],
        "total_time": sum(per_turn),
        "llm_calls": tokens.n,
        "in_tokens": tokens.i,
        "out_tokens": tokens.o,
        "meals_today": len(meals),
        "meals_all": len(yday),
        "kcal": totals.kcal,
    }))


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #

def main() -> int:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    have_groq = bool(os.getenv("GROQ_API_KEY"))

    rows = []
    for name in CONFIGS:
        if name != "sarvam" and not have_groq:
            console.print(f"[yellow]skipping '{name}' - no GROQ_API_KEY in .env[/yellow]")
            continue
        console.print(f"[dim]running {name} ...[/dim]")
        proc = subprocess.run(
            [sys.executable, __file__, name], capture_output=True, text=True, timeout=900
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            r = json.loads(line)
            rows.append(r)
            console.print(
                f"  errors={r['errors']} p50={r['p50']:.1f}s p95={r['p95']:.1f}s "
                f"in={r['in_tokens']:,} out={r['out_tokens']:,} meals(today)={r['meals_today']} kcal={r['kcal']}"
            )
        except Exception:
            console.print(f"[red]{name} produced no result[/red]\n{proc.stderr[-500:]}")

    table = Table(title="Model comparison - the brief's 11-message conversation")
    for col in ("config", "errors", "p50", "p95", "total", "LLM calls", "in tok", "out tok", "meals", "kcal"):
        table.add_column(col, justify="left" if col == "config" else "right")
    for r in rows:
        table.add_row(
            r["config"], str(r["errors"]), f"{r['p50']:.1f}s", f"{r['p95']:.1f}s", f"{r['total_time']:.0f}s",
            str(r["llm_calls"]), f"{r['in_tokens']:,}", f"{r['out_tokens']:,}",
            str(r["meals_today"]), str(r["kcal"]),
        )
    console.print(table)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in CONFIGS:
        _worker(sys.argv[1])
    else:
        sys.exit(main())
