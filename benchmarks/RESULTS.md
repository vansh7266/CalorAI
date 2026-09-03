# Latency

Run: `python benchmarks/latency.py 30 10` — 36 text turns, 10 image turns, GLM-5.2 (text) +
Gemma 4 (vision) on Sarvam. Fresh SQLite per run. Reflection is off the response path.

## Numbers

| path | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| **text path** (total) | 36 | **2.1 s** | **3.0 s** | 2.1 s | 1.2 s | 3.1 s |
| text path — time to first token | 6 | 2.7 s | 3.1 s | 2.7 s | 2.2 s | 3.1 s |
| **image path** (total) | 10 | **7.8 s** | **31.1 s** | 11.9 s | 7.1 s | 49.2 s |

Component timings (single call):

| component | time |
|---|---|
| raw text-model call (GLM-5.2) | ~1.1 s |
| raw vision-model call (Gemma 4) | ~0.2 s |
| full vision extraction (load + resize + call + parse) | ~1.2 s |
| cached nutrition lookup | ~3 ms |

## Why the text path is ~2 s

A text turn is **two sequential model calls**: the agent decides and emits a tool call, the
tool runs, then the agent writes the reply. 2 × ~1.1 s ≈ the 2.1 s p50. The tool step itself
is negligible (SQLite + cached nutrition are single-digit ms).

## What we did to keep it there

- **`log_meal` resolves nutrition itself** — no separate `lookup_nutrition` round-trip.
- **Today's totals are injected into context**, so "how am I doing today?" is answered with
  **zero tool calls** (verified: the graph visits `ingest → load_context → agent` only).
- **Nutrition cache** — the first mention of a food costs one worker-model call (~1.5 s), every
  mention after is a 3 ms DB read, and the number never drifts.
- **Vision runs in parallel with context loading** on the image path (`load_context` and
  `vision_extract` fan out from `ingest`), so the vision call overlaps the DB reads instead
  of adding to them.
- **The reflection pass runs after the reply is sent**, on a background thread — it adds
  nothing to response time.
- **No embeddings on the hot path** — memory recall is a keyword/`LIKE` query.
- **Streaming** — the reply streams token by token once the model starts producing it.

## What's slow, and what we couldn't fix

- **GLM-5.2 reasons on every call and it can't be turned off.** Sarvam rejects the
  `reasoning` / `enable_thinking` parameters with HTTP 422. Each call spends ~80–120 output
  tokens thinking before it answers, which is why **time-to-first-token (~2.7 s) is close to
  the full turn time** on tool-calling turns — the reasoning blocks the first visible token.
  This is the single biggest cost. It is a deliberate quality trade-off (GLM-5.2 scores far
  higher than the fast alternatives on the judgment calls this task is really about), and it
  is one env var to change: set `MODEL_PROFILE=groq` (or any non-reasoning model) for a
  faster path at some cost to the harder cases.
- **The image path p95 (31 s) and max (49 s) are transient Sarvam API slowdowns.** Roughly
  one call in ~20 returns in 40–50 s instead of 2–5 s, with no error. `max_retries=2` covers
  hard failures but not a slow success. The p50 (7.8 s) is the honest typical number; the
  p95 is dominated by one outlier in a 10-sample set. A larger sample would pull p95 toward
  the low teens.
- **A text turn is two model calls.** It could be collapsed to one with a templated
  confirmation string, but that trades away the conversational quality the task grades. We
  kept the slower, better path.
- **The image path adds ~5 s over the text path**, not the ~1.2 s of the raw vision call:
  the vision result puts 6–8 items with confidences into the agent's prompt, so the agent's
  own calls reason over more, and logging a multi-item plate resolves several foods. The raw
  vision call is cheap; making sense of a full plate is not.

## If we had more time

- Cache vision results by image hash (re-sending the same photo is free).
- A one-call fast path for the unambiguous "had 2 rotis" case (skip the second model call
  when the first turn's tool result is high-confidence and needs no narration).
- Prompt caching for the static system prompt (Sarvam bills a cached-input rate; wiring the
  cache markers would cut input processing on every call).
