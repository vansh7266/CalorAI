# Requirements coverage

Every item from the brief, mapped to where it lives and how it was verified.

## Core features (must have)

| # | Requirement | Where | Verified |
|---|---|---|---|
| 1 | Conversational agent + tool calling (LangChain/LangGraph); tools for log meal, retrieve past meals, look up nutrition, current totals | `calorai/agent/graph.py` (LangGraph), `calorai/agent/tools.py` (8 tools) + `memory.py` (2 memory tools) | `tests/test_tools.py`, live |
| 2 | Persistent DB; meals persist across turns **and sessions** | `calorai/db/` (SQLite, `data/calorai.db`) + LangGraph SqliteSaver checkpointer | `tests/test_db.py`; live: quit + `--user <id>` resumes, cross-session correction works |
| 3 | Running daily totals (calories + macros), correct through edits & deletions; "how am I doing today?" anytime | `repositories.daily_totals` = SQL SUM over active items (computed, never stored); totals injected into the prompt each turn | `tests/test_db.py::test_correction_updates_not_doubles`, `test_soft_delete_excludes_from_totals`; eval `totals_*` |
| 4 | Image on a **separate** model; document why + handoff + ambiguity handling | `calorai/vision/extract.py` (Gemma 4), `graph.py` `vision_extract` node (parallel with `load_context`); per-item confidence; low confidence / not-food → agent asks | `tests/test_vision.py`; live on 3 real photos |
| 5 | Persistent memory — selective, survives sessions, retrieved without bloating every prompt | `calorai/agent/memory.py`: profile card (Tier 1, ~200 tokens, always) + `save_memory`/`recall_memory` (Tier 2, model-driven) + reflection pass (post-reply, daemon thread) + routine suggestions; upsert-supersedes; `/memory` `/forget` | `tests/test_memory.py`; live: vegetarian → meat questioned, "my usual" across sessions, 140g goal stored |
| 6 | Multi-turn ambiguity — log vs ask | system prompt policy (`prompts.py`), `interrupt`-free (agent just asks in text) | eval `ambiguity_*`; live: "grazed all afternoon" offers not invents |

## The three that separate good from average

| Case | Handled by | Verified |
|---|---|---|
| "actually that was 3 rotis not 2" — update, no double count | `update_meal` sets quantity in place; totals recompute; `meal_edits` audit | structural (never INSERTs a 2nd row); eval `correction_rotis_no_double_count`; live correction chains |
| "same as yesterday" / "my usual" — memory, not parsing | "same as yesterday" → `get_meals(date="yesterday")` then `log_meal`; "my usual" → `recall_memory` routine → `log_meal` | eval `episodic_same_as_yesterday`, `routine_my_usual_after_saving`; live |
| photo + caption → one meal | `vision_extract` result + caption both go to the agent, which makes a single `log_meal` call | eval `multimodal_caption_half`; live: pork plate + "half my brother's" → one meal, halved |

## Latency

- Measured p50 / p95 for text and image paths — `benchmarks/latency.py`, results in `benchmarks/RESULTS.md`.
- Optimizations + honest limits documented there (GLM reasoning can't be disabled; Sarvam transient spikes).

## Bonus

| Bonus | Status |
|---|---|
| LangSmith tracing wired up | ✅ `LANGSMITH_TRACING=true` in `.env`; CLI banner shows the project link |
| Eval set with a defined "correct" | ✅ `evals/` — 22 cases, `graders.py` defines correctness, 22/22 pass |
| Streaming responses | ✅ `runner.stream_turn`; GLM emits 1–2 chunks after its reasoning pass (documented) |
| Multiple users / session isolation | ✅ checkpointer thread = user id; every query filtered by user_id; `tests/test_robustness.py::test_multi_user_isolation` |

## Definition of done

- ✅ Runs from a clean clone with documented setup — verified (fresh clone + venv + `pip install`); cross-platform hardened (Windows/Linux)
- ✅ Accepts image input — `/img <path> [caption]` in the REPL, or `python cli.py --image <path>`
- ✅ No authentication
- ✅ No frontend polish
- ✅ Nutrition data choice documented — seed table + cache + model fallback (`calorai/nutrition/`, README)
- ⏳ README covers all required sections — **pending (Phase 7)**
- ⏳ Walkthrough video — pending (user records)

## Red flags — all avoided

doesn't run clean ✗ (verified it does) · missing core features ✗ (all 6) · everything through one model ✗ (separate vision model) · memory = chat history in prompt ✗ (structured table + selective retrieval) · totals break on correction ✗ (structural guarantee) · no latency measurement ✗ (measured) · one giant commit ✗ (17+ iterative commits) · no docs ✗ (`docs/`, PROJECT_LOG, README pending) · overbuilt ✗ (working agent, no k8s)

## Green flags — hit

non-overlapping tool boundaries · selective memory · corrections without double-count · vision uncertainty surfaced · measured latency + reasoning · evals · clean iterative commits · honest docs · documented AI-tool usage
