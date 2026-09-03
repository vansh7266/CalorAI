"""Measure response latency for the text path and the image path.

    python benchmarks/latency.py             # default: text x30, image x8
    python benchmarks/latency.py 40 16       # text x40, image x16

Reports p50 / p95 / mean per path. Every text turn is streamed so total time and
time-to-first-token come from the same sample. The component table (raw text
call, full vision extraction, cached nutrition lookup) lets the numbers be
reasoned about rather than just quoted. Larger `image` counts give a steadier
p95 but cost more.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CALORAI_SYNC_REFLECTION", "0")  # reflection is off the response path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from calorai.db import database, repositories as repo  # noqa: E402

console = Console()

TEXT_MESSAGES = [
    "had 2 rotis and dal for lunch",
    "how am I doing today?",
    "actually that was 3 rotis not 2",
    "how much protein have I had?",
    "had a bowl of rice too",
    "skipped dinner",
]

IMAGE_CASES = [
    ("samples/paratha_plate.png", ""),
    ("samples/thali.jpeg", "half of this was my brother's"),
]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _fresh_db(tag: str) -> None:
    path = PROJECT_ROOT / "benchmarks" / "results" / f"{tag}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    database.DB_PATH = path
    database._schema_applied = False
    database.init_db(path)


def bench_text(target_turns: int) -> tuple[list[float], list[float]]:
    """Measure exactly `target_turns` text turns. Every turn is streamed and its
    time-to-first-token recorded, so the two samples are the same size."""
    from calorai.agent.runner import stream_turn

    totals: list[float] = []
    ttfts: list[float] = []
    user = None
    for n in range(target_turns):
        if n % len(TEXT_MESSAGES) == 0:
            _fresh_db(f"text_{n}")
            user = repo.create_user("bench", "UTC")
        msg = TEXT_MESSAGES[n % len(TEXT_MESSAGES)]
        start = time.time()
        first = None
        for chunk in stream_turn(msg, user_id=user.id, thread_id=user.id):
            if first is None and chunk.strip():
                first = time.time() - start
        totals.append(time.time() - start)
        ttfts.append(first if first is not None else time.time() - start)
    return totals, ttfts


def bench_image(runs: int) -> list[float]:
    from calorai.agent.runner import run_turn

    totals: list[float] = []
    for r in range(runs):
        _fresh_db(f"img_{r}")
        user = repo.create_user("bench", "UTC")
        image, caption = IMAGE_CASES[r % len(IMAGE_CASES)]
        start = time.time()
        run_turn(caption, user_id=user.id, thread_id=user.id, image_path=str(PROJECT_ROOT / image))
        totals.append(time.time() - start)
    return totals


def bench_components() -> dict[str, float]:
    from calorai.models.gateway import get_text_model
    from calorai.nutrition.resolver import resolve
    from calorai.vision.extract import extract_food_from_image

    _fresh_db("components")

    t = time.time()
    get_text_model().invoke("Reply with: ok")
    text_call = time.time() - t

    resolve("roti")  # warm the cache
    t = time.time()
    resolve("roti")
    cached_nutrition = time.time() - t

    # a real vision-path call goes through image load + resize + the model
    t = time.time()
    extract_food_from_image(str(PROJECT_ROOT / "samples" / "thali.jpeg"))
    vision_extract = time.time() - t

    return {
        "raw text-model call": text_call,
        "full vision extraction (load + resize + model)": vision_extract,
        "cached nutrition lookup": cached_nutrition,
    }


def _stats_row(name: str, values: list[float]) -> tuple[str, ...]:
    return (
        name,
        str(len(values)),
        f"{_pct(values, 0.5):.1f}s",
        f"{_pct(values, 0.95):.1f}s",
        f"{statistics.mean(values):.1f}s" if values else "-",
        f"{min(values):.1f}s" if values else "-",
        f"{max(values):.1f}s" if values else "-",
    )


def main() -> int:
    text_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    image_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    console.print(f"[dim]benchmarking: text x{text_runs} turns, image x{image_runs} turns...[/dim]")

    text_totals, ttfts = bench_text(text_runs)
    image_totals = bench_image(image_runs)
    components = bench_components()

    table = Table(title="Response latency")
    for col in ("path", "n", "p50", "p95", "mean", "min", "max"):
        table.add_column(col, justify="right" if col != "path" else "left")
    table.add_row(*_stats_row("text path (total)", text_totals))
    table.add_row(*_stats_row("text path (time to first token)", ttfts))
    table.add_row(*_stats_row("image path (total)", image_totals))
    console.print(table)

    comp = Table(title="Component timings (single call)")
    comp.add_column("component")
    comp.add_column("time", justify="right")
    for name, secs in components.items():
        comp.add_row(name, f"{secs * 1000:.0f} ms" if secs < 1 else f"{secs:.1f}s")
    console.print(comp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
