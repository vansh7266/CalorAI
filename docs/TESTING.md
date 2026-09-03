# Testing

What was tested, how, and the results. Everything here is reproducible — the
commands and the exact messages are included.

## Summary

| Layer | What | Count | Command | Result |
|---|---|---:|---|---|
| Unit | database, tools, memory, vision parsing, schema migrations, tool-call recovery, graph, CLI | 106 | `pytest` | 106 / 106 pass (offline, no API key) |
| Eval | scenario tests with a defined grader | 23 | `python evals/run.py` | pass (needs an API key) |
| End-to-end | the brief's test conversation | 1 | see Test 1 | pass |
| End-to-end | hard run — multimodal, multi-part correction, restart | 1 | see Test 2 | pass |
| End-to-end | hard run — adversarial input, multi-user, schema migration | 1 | see Test 3 | pass |

Environment for the runs below: macOS 26.6 (arm64), Python 3.13, `MODEL_PROFILE=sarvam`
(GLM-5.2 text + Gemma 4 vision), commit `4aa489b`. Each run uses a fresh database.

---

## Automated checks

### Unit tests — `pytest`

```
106 passed
```

Run with the provider key removed and no `.env` — the suite is fully offline. Covers:

- `test_db`, `test_migrations` — schema, computed totals, soft-delete, upgrade from the previous schema
- `test_tools` — every tool, the read/write split, invalid-argument rejection
- `test_nutrition` — seed / cache / model resolution, unit compatibility
- `test_memory` — profile card, save/recall, the reflection pass
- `test_vision` — image parsing, size and pixel guards, error paths
- `test_recovery`, `test_formatting` — GLM tool-call quirks, markdown stripping
- `test_graph`, `test_robustness` — node behaviour, degraded-context handling, "never raises"
- `test_cli` — onboarding, slash commands, Rich-markup escaping

### Eval set — `python evals/run.py`

23 scenario cases with `evals/graders.py` defining "correct" per the brief:

| category | cases | checks |
|---|---:|---|
| extraction | 3 | right items and quantities land in the DB |
| totals | 3 | daily total is right after the turn |
| correction | 4 | quantity updates in place, `meals_today` stays 1 |
| memory-write | 2 | the fact is stored |
| memory-use | 2 | the fact shows up in the reply |
| routine | 2 | "my usual" logs the saved items |
| episodic | 1 | "same as yesterday" resolves via lookup |
| ambiguity | 3 | `meals_today: 0` and the reply asks, never invents a meal |
| multimodal | 2 | photo + caption → one meal, correct source, portions scaled by the caption |
| sustained | 1 | tools stay available across 7 tool-using turns |

---

## Test 1 — The brief's test conversation

The eleven messages from the task brief, run in order against a fresh database.
Setup: one "yesterday" meal is logged first so the memory-dependent turns have
something to act on (the brief calls these "memory, not parsing").

**Setup:** `had rajma and rice for lunch yesterday`
→ logged on yesterday's date, ~410 cal.

| # | Message | Expected | What happened |
|---|---|---|---|
| 1 | had 2 parathas and chai for breakfast | log, state the assumption | Logged 2 parathas + chai, ~450 cal; asked "aloo or paneer?" |
| 2 | leftover biryani, maybe two thirds of the box | log an estimated portion | Logged ~1.5 plates biryani as lunch, ~435 cal; asked "chicken or veg?" |
| 3 | skipped lunch but grazed all afternoon | **do not invent a meal**; offer a rough number | No meal logged. "If you can remember roughly what you snacked on, I can log it… otherwise no stress." |
| 4 | same as yesterday | resolve via memory, not parsing | Looked up yesterday's lunch, logged rajma + rice, ~410 cal |
| 5 | actually that was 3 rotis not 2 | **update, do not double-count** | Breakfast quantity 2 → 3 in place; day total moved +180 cal, not +630 |
| 6 | how much protein have I had today? | answer from context, no tool call | "49.5 g protein so far today across your 3 meals." |
| 7 | how am I doing on calories? | answer from context | "You're at 1475 cal today across 3 meals." Offered to save a goal. |
| 8 | [photo of a plate] | route to the vision model, log as one meal | Gemma 4 read the thali; logged meat curry, dal, 3 roti, rice, raita, salad as dinner (`source = image`), ~1330 cal; asked "chicken or mutton?" |
| 9 | [photo] "half of this was my brother's" | **one meal, portions halved** | Half pork chop, 1.5 potatoes, half salad — one `log_meal` (`source = image+text`), ~300 cal |
| 10 | my usual | if nothing saved, ask and offer to remember | Recalled the routine auto-saved from turn 4, noted it was already had today, asked which usual they meant |
| 11 | i'm vegetarian btw | save it; from now on question meat | Saved as `diet: vegetarian`; flagged the meat curry and pork chop already logged today |

**Database check after the run:**

- Computed day total = **3106 cal / 127.3 g protein** — exactly what the agent reported on the last turn. Totals stayed exact through the correction.
- `meal_edits`: one row — `item:paratha:quantity  2.0 → 3.0`. The correction did not create a second meal.
- `memory` (active): `diet: vegetarian` (`stated`), `routine: usual afternoon snack = rajma and rice` (`inferred` — from the reflection pass).
- Photo meals carry `source = image` and `source = image+text`; the captioned photo is a single meal.

The three the brief calls out as separating good from average: **correction without
double-count** (turn 5), **memory not parsing** (turns 4 and 10), **photo + caption
→ one meal** (turn 9) — all pass.

---

## Test 2 — Hard end-to-end: photo, multi-part correction, restart

A fresh clone and empty database. Ten steps, following an independent reviewer's
"ready to record" protocol.

| Step | Action | Result |
|---|---|---|
| 1–2 | Fresh `pip install -e .`, `pip check`, `pytest` in the clone; CLI onboarding | install clean, 106 tests pass, user created, `user_version = 1` |
| 3 | Photo (`samples/thali.jpeg`) + caption "I ate half the rice and two rotis" | one meal, `source = image+text`, 8 items, caption applied (roti → 2, rice → 0.5) |
| 4 | Verify the DB | reply total 1091 cal / 52 g protein = computed DB total; seed nutrition for known foods, model for the rest |
| 5 | "remove the rice, change rotis to 1, and add 200 g chicken" — one message | atomic: one meal, three edits recorded; **200 g chicken resolved per-gram (1.65 kcal/g = 330 cal)**, not as "200 bowls" |
| 6 | "what are my totals today?" | answered from context, no tool call, = DB |
| 7 | 8 more tool-using turns (add eggs to breakfast, log lunch, correct to 2×, "what did I eat", save vegetarian, delete an item) — 11+ tool turns in the thread | no tool-budget exhaustion; every running total matched the DB exactly; "I'm vegetarian" saved and the meat already logged was flagged |
| 8 | Restart the process | meals, corrections, memory, and conversation history all persisted |
| 9 | Invalid inputs (see Test 3 for the full set) | impossible date → asks; missing image → error + exit 1; negative quantity → rejected before write |
| 10 | Open a database built by the previous schema | migrates to `user_version = 1`, all rows preserved, live turn works |

**Key result:** the correction in step 5 is the exact failure an earlier audit found
(`200 g chicken` becoming a per-cup value ≈ 48,000 cal). It now resolves to a
per-gram estimate and the day total stays sane.

---

## Test 3 — Hard end-to-end: adversarial input, multi-user, schema migration

Designed to be re-run by hand on any machine (including Windows). Three parts.

### 3A — Correction chains and adversarial input

Fresh database, onboard as any name, then:

| # | Message | Result |
|---|---|---|
| 1 | had 3 eggs for breakfast | logged, 234 cal |
| 2 | actually make that 5 eggs | quantity → 5, 390 cal |
| 3 | actually make that 4 | quantity → 4, **312 cal — not stacked to 12** |
| 4 | add 2 slices of toast to breakfast | item added, 472 cal |
| 5 | remove the eggs from breakfast | item removed, breakfast is now just toast, 160 cal |
| 6 | how am I doing today? | "160 cal… just that breakfast toast." = DB |
| 7 | log -3 rotis for lunch | **refused** — "I can't log a negative amount"; asks what was meant |
| 8 | I had a dosa on the 45th of Maytember | **refused** — "that isn't a month I've heard of"; asks for a real date |
| 9 | how many calories in 2 gulab jamun? | answered (~300 cal), **not logged** |
| 10 | what's the capital of France? | "Paris — but I'm better with calories"; nothing logged |
| 11 | qwerty asdfgh zxcvb | no meal invented; asks what they meant |
| 12 | had idli, sambar, and filter coffee | one meal, three items |
| 13 | `/img docs/agent-graph.png` (a non-food image) | "That looks like a diagram, not food. What did you actually eat?" |
| 14 | `/img samples/pork_plate.jpg` | logged — pork chop, potatoes, salad, ~650 cal |
| 15 | actually only 2/3 of that plate was mine | **every item scaled to 2/3 in one atomic edit** — pork 1→0.67, potato 3→2, salad 1→0.67; reply total = DB total. (Regression guard: this multi-item correction in a long-running session used to hit "database is locked".) |
| 16 | `/img "/tmp/no such file.jpg" my dinner` | "no file at /tmp/no such file.jpg"; exit code 1 |

**DB check:** after the adversarial turns (7–11), the only meals are the real ones
(breakfast toast, the idli meal). Nothing garbage was written. The correction chain
audit trail is clean: `egg 3→5`, `egg 5→4`, `add toast`, `remove egg`.

### 3B — Multi-user isolation

```
onboard as "Ravi"   → log "had 2 rotis and dal for lunch"        (390 cal)
delete .session, onboard as "Priya"  → log "a big chicken biryani"  (290 cal)
```

- Ravi asks "what did I eat today?" → 2 rotis and dal only. No biryani.
- Priya asks "what did I eat today?" → chicken biryani only. No rotis.
- DB: two users, completely separate meals and totals. The LangGraph thread id is
  the user id, so the checkpointed conversation is per-user too.

### 3C — Schema migration from the previous release

A database is built with the schema shipped by the previous commit (no
`user_version`, old `nutrition_cache` primary key, loose numeric checks), with a
user, a meal, two items, two memories, a cache row, and an edit-trail row. Then it
is opened with the current code:

```
[built] previous-release DB: user_version=0
[open with current code — migrations run inside init_db]
  user_version now       : 1
  nutrition_cache PK     : ['name', 'unit']
  user preserved         : {'name': 'Meera', 'timezone': 'Asia/Kolkata'}
  meal preserved         : {'description': '3 roti, paneer curry', 'meal_type': 'dinner'}
  items preserved        : [('roti', 3.0), ('paneer curry', 1.0)]
  memory preserved       : ['targets 130 g protein per day', 'vegetarian']
  edit trail preserved   : 1 row(s)

[live turn on the upgraded DB]
  "I just had 100g paneer and 2 rotis"
  → Logged 100g paneer and 2 rotis — about 507 cal and 24g protein.
    You're at 24g of your 130g protein target.   (recalled the migrated goal)
```

The same path is asserted in `tests/test_migrations.py` (`pytest tests/test_migrations.py`).

### Reproducing Test 3 on your machine

Part 3A and 3B are just messages — type them into `calorai` in order (delete
`data/.session` between the two users in 3B, or pass a fresh name). Part 3C:

```bash
pytest tests/test_migrations.py -v
```

---

## Latency

Measured with `python benchmarks/latency.py 30 10` — 30 text turns, 10 image turns,
GLM-5.2 + Gemma 4 on Sarvam. Every text turn is streamed, so "total" and
"time to first token" are measured on the same turns. Fresh database; a new user
every 6 turns so the agent↔tools loop is exercised, not just cache hits.

| Path | n | p50 | p95 | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|
| Text (total) | 30 | **2.2 s** | **3.2 s** | 2.2 s | 1.2 s | 3.7 s |
| Text (time to first token) | 30 | 2.1 s | 3.1 s | 2.1 s | 1.2 s | 3.7 s |
| Image (total) | 10 | **8.7 s** | **9.7 s** | 8.6 s | 7.3 s | 9.8 s |

Component calls: raw text-model call ~1.1 s · full vision extraction ~1.3 s ·
cached nutrition lookup ~4 ms.

### Why the text path is ~2 s

A logging turn is two sequential model calls — the agent reads the message and
emits a tool call, the tool runs (single-digit ms), then the agent gets the result
back and writes the reply. `2 × ~1.1 s` is the 2.2 s p50. A question answered from
the context block is one call and sits near the 1.2 s minimum.

Time-to-first-token tracks total almost exactly because GLM-5.2 does not stream
while it reasons, then sends the reply in one or two chunks — "first token" arrives
just before "last token" on this provider.

### What was done to keep it there

- `log_meal` resolves nutrition itself — no separate lookup round-trip.
- Today's totals are injected into the prompt — "how am I doing today?" needs zero
  tool calls.
- Nutrition cache — first mention of a food costs a model call; every mention after
  is a ~4 ms DB read.
- Vision runs in parallel with context loading, so extraction overlaps the DB reads.
- The reflection pass runs after the reply is sent, on a background thread.
- Prompt history is capped at the last 8 turns; the full thread stays in the
  checkpointer.

### What is slow and could not be fixed

- GLM-5.2 always reasons and Sarvam rejects the parameter to disable it — a fixed
  ~1 s floor per call. Deliberate quality trade-off; `MODEL_PROFILE=groq` is faster.
- A logging text turn is two model calls. One is possible with a templated
  confirmation, at the cost of reply quality.
- The image path is ~6 s over the text path: the vision items go into the agent's
  prompt, so its calls then reason over more, and a multi-item plate resolves
  several foods.
- Sarvam has occasional latency spikes — this run was clean, but roughly one image
  call in ~20 has returned in 30–50 s with no error. Retries do not help a slow
  success.

---

## Green flags from the brief — where each is covered

| Green flag | Where |
|---|---|
| Tool boundaries that make sense and don't overlap | README "Tools"; read/write split, `lookup_nutrition` vs `log_meal` |
| Memory that's actually selective | 3-tier design; only a capped card is always in context |
| Corrections without double-counting | Test 1 turn 5, Test 2 step 5, Test 3A; totals are computed, never stored |
| Vision uncertainty surfaced, not guessed | confidence bands in the prompt; Test 1 turn 8 ("chicken or mutton?") |
| Measured latency with real tradeoff reasoning | this file + README "Latency" |
| Evals set up | `evals/` — 23 cases with a grader |
| Clean commit history | one commit per phase, descriptive messages |
| Honest README about what's incomplete | README "Assumptions and trade-offs" |
| Smart, documented AI-tool use | README "AI-tool usage" |
