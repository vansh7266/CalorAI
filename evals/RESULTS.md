# Eval results

Run: `python evals/run.py` — 22 cases, GLM-5.2 (text) + Gemma 4 (vision), fresh SQLite per case,
reflection run inline for determinism.

## Latest run: 22 / 22 passed

| category | cases | result |
|---|---|---|
| extraction | 3 | 3/3 |
| correction | 4 | 4/4 |
| totals | 3 | 3/3 |
| memory-write | 2 | 2/2 |
| memory-use | 2 | 2/2 |
| routine | 2 | 2/2 |
| episodic | 1 | 1/1 |
| ambiguity (don't invent) | 3 | 3/3 |
| multimodal | 2 | 2/2 |

Per-case time: most 2–9 s; one multimodal case hit ~47 s (a transient Sarvam API slowdown —
the next case was back to ~7 s). See the latency section of the README.

## What "correct" means

Defined in `evals/graders.py`, one rule per the README:

1. **extraction** — the right items and quantities are in the DB (`has_items`, `item_quantity`)
2. **totals** — daily calorie total is in range after the turn (`total_kcal_between`)
3. **no double-count** — a correction leaves `meals_today: 1`, not 2
4. **memory** — the fact is stored (`memory_has`) and/or shows up in the reply (`reply_contains_any`)
5. **ambiguity** — `meals_today: 0` plus the reply asks/offers rather than assuming (`reply_asks`)
6. **one meal per real meal** — photo + caption ⇒ `meals_today: 1`

Most checks read the database directly; a few read the reply text.
