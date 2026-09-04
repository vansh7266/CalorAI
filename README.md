# CalorAI Logging Agent

A conversational agent that logs meals the way people actually talk about food — in
plain language, by text or photo, no forms. It keeps an accurate running daily
total, remembers what matters about you between sessions, and replies at messaging
speed.

Built with LangGraph. SQLite for persistence. Runs locally as a CLI.

The default setup runs text on **GLM-5.2** and photos on a separate vision model,
**Gemma 4** (both via Sarvam). Both models are configurable — you can point them at
OpenAI, Anthropic, Google, or Groq instead. See
[Using another provider](#using-another-provider).

---

## What it does

- **Logs meals from text** — "had 2 rotis and dal for lunch" becomes structured log
  entries with calories and macros.
- **Logs meals from a photo** — send a picture of your plate; a vision model reads
  the items and the text agent logs them.
- **Keeps a running daily total** — ask "how am I doing today?" any time and get an
  accurate answer, including after edits and deletions.
- **Handles corrections without double-counting** — "actually 3 rotis not 2" updates
  the meal in place; the total never drifts.
- **Remembers across sessions** — that you're vegetarian, that you target 140 g
  protein, that "my usual" is a specific breakfast.
- **Asks only when it needs to** — logs with a sensible assumption and says what it
  assumed; asks a single short question when the food or quantity is genuinely
  unknowable.

---

## Setup

Requires Python 3.10 or newer and one model-provider API key (Sarvam by default).

```bash
git clone https://github.com/vansh7266/CalorAI.git
cd CalorAI

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # or: pip install -r requirements.txt (pinned)

cp .env.example .env                  # Windows: copy .env.example .env
```

Open `.env` and set your key:

```
MODEL_PROFILE=sarvam
SARVAM_API_KEY=your_key_here
```

Then start it:

```bash
python cli.py
```

On first run it asks your name and prints a user id (`usr_...`). Keep that id — it
is how you resume the same history later.

### Using another provider

The gateway is provider-agnostic. Two ways to configure it, both in `.env`:

- **One provider for everything** — set `MODEL_PROFILE` (`openai` / `anthropic` /
  `google` / `groq` / `sarvam`) and its API key. For any provider except Sarvam,
  also set `TEXT_MODEL` and `VISION_MODEL` (hosted names change often, so there is
  no built-in default).
- **A different provider per role** — e.g. text on OpenAI, vision on Google: set
  `TEXT_PROVIDER` / `TEXT_MODEL` and `VISION_PROVIDER` / `VISION_MODEL` separately,
  and include the API key for each. The worker model follows the text model unless
  you set `WORKER_PROVIDER` / `WORKER_MODEL`.

`.env.example` has a worked example of each. Anthropic and Google need an extra
install:

```bash
pip install -e ".[anthropic]"      # or ".[google]"
```

---

## Using it

### Text

Just type what you ate:

```
you  had a paratha and chai for breakfast
you  actually that was 2 parathas
you  how much protein have I had today?
```

### Photos

In the chat, use `/img`:

```
/img <path> [optional caption]
```

Examples:

```
/img samples/thali.jpeg
/img ~/Downloads/lunch.jpg had half of this
/img "~/My Photos/dinner plate.jpg"          # quote paths that contain spaces
```

- The path can be relative to where you started the CLI, or a full/absolute path
  pasted as-is (`~` expands; on Windows `C:\Users\...\lunch.jpg` is fine). Wrap it
  in quotes only if it contains spaces.
- Supported formats: `.jpg` `.jpeg` `.png` `.webp` `.gif` `.bmp`, up to 20 MB.
- A photo plus a caption is treated as **one meal** — the caption adjusts the
  vision items (e.g. "half was my brother's" halves the portions).
- The image is sent only to the vision model, never to the text model.

### One-shot (non-interactive)

Send a single message and exit — useful for scripting or a quick check:

```bash
python cli.py --message "had 2 rotis and dal"
python cli.py --user usr_xxxxxxxx --message "how am I doing today?"
python cli.py --user usr_xxxxxxxx --image ~/Downloads/lunch.jpg --message "half was my brother's"
```

### Slash commands

| Command | What it does |
|---|---|
| `/totals [date]` | calories and macros for today, or a given day |
| `/history [date]` | recent meals, or every meal on a day |
| `/memory` | everything the agent has remembered about you |
| `/forget <id>` | make it forget one memory (id from `/memory`) |
| `/whoami` | your CalorAI user id |
| `/img <path> [caption]` | log a meal from a photo |
| `/help` | command list |
| `/quit` | exit |

Dates accept `today`, `yesterday`, a weekday, or `YYYY-MM-DD`.

### Resuming

```bash
python cli.py --user usr_xxxxxxxx
```

The last-used id is also saved locally, so a plain `python cli.py` resumes it
automatically. If you ran `pip install -e .`, the `calorai` command works
everywhere `python cli.py` does.

---

## Architecture

One LangGraph state machine per turn. State is checkpointed to SQLite per user, so
a conversation survives a restart and each user's thread is isolated.

Two database files under `data/`: `calorai.db` holds the application tables
(meals, items, memory, edits, nutrition cache); `checkpoints.db` holds LangGraph's
per-turn graph state. Keeping them apart means graph-state writes never lock out a
meal or memory write.

```
              ingest
             /      \
    load_context   vision_extract      (parallel; vision only when there is a photo)
             \      /
              agent  <----->  tools
                |
               end
```

- **ingest** — classify the turn as text, image, or image + text; clear any stale
  photo result from the previous turn.
- **load_context** — read today's totals, the most recent meal, and the memory
  profile card. Failures here are logged and flagged, not swallowed.
- **vision_extract** — a separate vision model turns the photo into items, each with
  a confidence. Runs in parallel with `load_context`.
- **agent** — GLM-5.2 with the tools bound. Decides whether to call a tool or reply.
- **tools** — run the tool calls and loop back to the agent (capped per turn).

Full diagram: [`docs/agent-graph.md`](docs/agent-graph.md).

---

## Models and the vision handoff

| | Text / conversation | Vision / photo → items |
|---|---|---|
| Model | GLM-5.2 | Gemma 4 |
| Provider | Sarvam | Sarvam |
| Role | reads the message, calls tools, writes the reply, applies memory | extracts food items and portions from an image only |
| Cost / latency | ~1.1 s per call; a logging turn is two calls | ~1.3 s per extraction (file read + resize + call) |

**Why two separate models.** The brief requires it, and it is the right split:
extraction and conversation are different jobs. Gemma 4 is fast and cheap and good
enough at "what is on this plate". GLM-5.2 is stronger on the judgment calls this
task is really about — is this ambiguous, should I ask or assume, does this contradict
what I know about the user. Running everything through one model would either slow
the common text turn down or make the vision step overkill.

**Why GLM-5.2 for text.** It has reliable tool-calling and it reasons before
answering, which is what keeps the ambiguity handling sane. The cost is that it
always spends ~80–120 tokens reasoning and Sarvam does not allow turning that off —
a fixed ~1 s floor per call. That is a deliberate quality-over-speed choice; a
faster non-reasoning model is one env var away (`MODEL_PROFILE=groq`).

**The handoff.** `vision_extract` returns a list of items, a confidence (0–1) for
each, and a free-text note about anything that made the photo hard to read. The text
agent gets that as a context block and decides what to do with it:

- confidence ≥ 0.7 — log it normally.
- confidence 0.4–0.7 — log it, but say it is a guess and invite a correction.
- confidence < 0.4, or a name like "unknown fried item" — ask the user about that
  item, log the confident ones.
- `is_food` false — say it does not look like food and ask what they had.
- a caption is applied to the items (identity and portion), and the photo + caption
  is always one `log_meal` call, never two meals.
- memory still wins — if the user is vegetarian and the vision model reports chicken
  curry, the agent flags it rather than logging it silently.

The vision model never writes to the database. It only ever proposes.

---

## Memory

Three tiers, each with a different job.

**1. Profile card — always in context, capped.**
A short block (diet, goals, a few lasting preferences — at most six lines) is built
from the memory table and injected into every system prompt. It is size-capped so it
cannot grow without bound. Routines ("my usual") are deliberately **not** in the card
— they are fetched on demand.

**2. Tools — the agent reads and writes explicitly.**
- `save_memory` — the agent stores a durable fact (a diet, an allergy, a target, a
  routine) when the user states one.
- `recall_memory` — the agent looks something up when the user refers to it ("my
  usual", "like I said last time").

**3. Reflection — a safety net after the reply.**
Once the reply is sent, a background thread re-reads the exchange with a cheap model
call and saves anything the explicit `save_memory` missed. Reflection writes are
marked `inferred` and shown as "(unconfirmed)" until the user confirms them, so a
guess never silently becomes fact.

**How it stays out of the way.** Only the capped card is always present. Everything
else is pulled in only when the agent asks for it. Recall is a keyword / `LIKE`
query — no embeddings on the response path.

---

## Tools

Read and write are split. Reads never mutate. Every tool returns a plain dict and
never raises — an error comes back as `{"error": "..."}` so the agent can recover or
ask the user.

| Tool | Type | Purpose |
|---|---|---|
| `log_meal` | write | log a new meal; resolves nutrition itself (no extra round-trip) |
| `update_meal` | write | correct an existing meal — quantity, add/remove an item, meal type, date — applied atomically |
| `delete_meal` | write | soft-delete a meal logged by mistake |
| `get_daily_totals` | read | totals for a past day (today's are already in context) |
| `get_meals` | read | recent meals or a day's meals; powers "same as yesterday" and finding a meal to fix |
| `lookup_nutrition` | read | estimate calories for a food **without** logging ("how many calories in a samosa?") |
| `save_memory` | write | store a durable user fact |
| `recall_memory` | read | look up something the user mentioned earlier |

A correction always goes through `update_meal` or `delete_meal`, never a second
`log_meal`. That is enforced in the prompt and is why double-counting cannot happen.

---

## Nutrition data

Resolution order for any food:

1. **Seed table** — about 40 common foods (Indian staples plus basics), per-unit
   macros. This is the one piece the brief explicitly allows to be hardcoded.
2. **Cache** — `nutrition_cache`, keyed by `(food, unit)`. Every resolved food is
   stored here.
3. **Model estimate** — if seed and cache miss, one worker-model call estimates the
   macros, and the result is written to the cache.

**Why a cache instead of asking the model every time.** The models already know food
macros — the problem is that a fresh estimate drifts between calls, which would make
the daily total wobble for no reason. Resolving each food once and reusing the number
keeps totals stable.

**Unit safety.** A seed value is only used when its unit is compatible with what was
asked. A per-cup seed number does not answer "100 g rice" — that falls through to a
per-gram model estimate.

**Trade-off.** Seed + cache is instant and stable but approximate. A real API (USDA
FoodData Central) would be more accurate per portion, at the cost of another key,
network latency on the hot path, and rate limits. The resolver is structured so a
real source can be dropped in behind one env flag; it is not built here.

---

## Daily totals stay correct

The daily total is **never stored**. It is computed every time as the sum of
`quantity × per-unit macros` over the active items for that day.

- A correction is a quantity update on one item.
- A deletion flips a meal's status to `deleted`.

Both simply change what the sum sees. There is no running counter, so there is
nothing to double-add to — correcting "2 rotis" to "3" can only ever produce the
right number.

---

## Latency

Measured with `python benchmarks/latency.py 30 10` (30 text turns, 10 image turns),
GLM-5.2 + Gemma 4 on Sarvam. Every text turn is streamed, so total time and
time-to-first-token are measured on the same turns.

| Path | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| Text (total) | 30 | **2.2 s** | **3.2 s** | 2.2 s | 1.2 s | 3.7 s |
| Text (time to first token) | 30 | 2.1 s | 3.1 s | 2.1 s | 1.2 s | 3.7 s |
| Image (total) | 10 | **8.7 s** | **9.7 s** | 8.6 s | 7.3 s | 9.8 s |

Component calls: raw text-model call ~1.1 s, full vision extraction ~1.3 s, cached
nutrition lookup ~4 ms.

### Why the text path is ~2 s

A logging turn is two sequential model calls — the agent reads the message and emits
a tool call, the tool runs (single-digit ms), then the agent gets the result back
and writes the reply. `2 × ~1.1 s` is the 2.2 s p50. A question that needs no tool
(answered from the context block) is one call and sits near the 1.2 s minimum.

Time-to-first-token tracks total almost exactly because GLM-5.2 does not emit tokens
while reasoning and then sends the reply in one or two chunks — on this provider,
"first token" arrives just before "last token".

### What was done to keep it there

- `log_meal` resolves nutrition itself — no separate lookup round-trip on the log
  path.
- Today's totals are injected into the prompt, so "how am I doing today?" needs
  **zero tool calls**.
- Nutrition cache — first mention of a food costs a model call, every mention after
  is a ~4 ms DB read.
- Vision runs in parallel with context loading, so the ~1.3 s extraction overlaps
  the DB reads instead of stacking on them.
- The reflection pass runs after the reply is sent, on a background thread.
- Conversation history sent to the model is capped at the last 8 turns; the full
  thread stays in the checkpointer.

### What is slow and could not be fixed

- GLM-5.2 always reasons and Sarvam rejects the parameter to disable it — a fixed
  ~1 s floor per call. Deliberate quality trade-off; `MODEL_PROFILE=groq` is faster.
- A logging text turn is two model calls. It could be one with a templated
  confirmation, but that trades away reply quality.
- The image path is ~6 s over the text path, not the ~1.3 s of the raw vision call:
  the vision items go into the agent's prompt, so its calls then reason over more,
  and a multi-item plate resolves several foods.
- Sarvam has occasional latency spikes — this run was clean, but roughly one image
  call in ~20 has returned in 30–50 s with no error. `max_retries` does not help a
  slow success.

Full run metadata and reasoning: [`docs/TESTING.md`](docs/TESTING.md).

---

## Testing

- **104 unit tests** — `pytest` — cover the database, tools, memory, vision parsing,
  schema migrations, GLM tool-call recovery, the graph, and the CLI. They run
  offline with no API key.
- **23-case eval set** — `python evals/run.py` — scenario tests with a defined
  grader (`evals/graders.py`): extraction, totals, corrections, memory write and
  use, routines, ambiguity, and multimodal.
- **The brief's own test conversation**, walked through end to end.
- **Two hard end-to-end runs** — one for multimodal + multi-part corrections +
  restart persistence, one for adversarial input + schema upgrade.

Full transcripts, expected behaviour, and how to reproduce each:
[`docs/TESTING.md`](docs/TESTING.md).

```bash
pytest                 # unit tests
python evals/run.py    # eval set (needs an API key)
```

---

## Assumptions and trade-offs

- **No authentication** (per the brief). Identity is a `usr_...` id the user keeps.
  It is not a security boundary and must not be treated as one if this is ever
  exposed over a network.
- **Single machine, local SQLite.** Fine for this task; not a multi-node design.
- **A logging text turn is two model calls.** Kept for reply quality over a
  templated one-call path.
- **Nutrition is estimated, not authoritative.** Stable-but-approximate was chosen
  over accurate-but-drifting.
- **The vision model only proposes.** A wrong or ambiguous result is caught by the
  text agent, the confidence bands, and the user — never logged silently.
- **Reasoning latency floor.** GLM-5.2 spends ~1 s reasoning on every call and that
  cannot be turned off on Sarvam.

---

## What I would do next

- Cache vision results by image hash so re-sending the same photo is free.
- A one-call fast path for unambiguous logs ("had 2 rotis") to halve that turn.
- Prompt caching for the static system prompt (Sarvam bills a cached-input rate).
- A real nutrition API (USDA) behind the existing env flag.
- A thin web / WhatsApp adapter over the same `runner` interface.

---

## Bonus features implemented

| Bonus | Status | Where |
|---|---|---|
| LangSmith tracing | Done | in `.env`, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` (free at smith.langchain.com) |
| Eval set with a defined "correct" | Done | `evals/` — 23 cases, `graders.py` |
| Streaming responses | Done | default in the CLI; `--no-stream` to disable |
| Multiple users / session isolation | Done | LangGraph thread = user id; per-user checkpoint |

Public LangSmith trace: https://smith.langchain.com/public/b72a0407-be4d-4793-8767-764646aff886/r/01a0695d-2b45-7ed2-90f2-82432593284b

---

## Time breakdown

_Approximately 13 hours, across more than one sitting._

| Area | Hours |
|---|---:|
| Understanding the brief, research, design discussion, locking decisions | ~3 |
| Implementation — graph, tools, database, memory, image path, CLI | ~4 |
| Review cycles, testing, and fixes | ~3 |
| README, testing docs, and the walkthrough video | ~3 |
| **Total** | **~13** |

---

## AI-tool usage

Two tools, two distinct roles.

**Claude (Claude Code) — research and build partner.**
- Used first to understand the brief, research the options (LangGraph, model
  choices, memory design, nutrition sources), and build the plan.
- Every design point was discussed in plain language, doubts were cleared, and
  decisions were locked one part at a time before any code was written.
- Then used to implement the whole thing end to end and to test and fix as it went.

**Codex (GPT-5.6) — independent reviewer.**
- Reviewed every part of the implementation, ran its own tests, and produced a
  written report.
- Each finding was handed back, verified by hand, and only the valid ones were
  fixed. This loop ran twice.

**Then** the unit tests, the brief's test conversation, and two hard end-to-end runs
— all passing — followed by this README and the testing guide. Work was pushed to
GitHub throughout; the walkthrough video is the last step.

All architectural decisions and trade-offs are the author's. The tools accelerated
the work; they did not make the calls.

---

## Project layout

```
calorai/
  agent/        graph, tools, memory, prompts, per-turn context, runner
  models/       provider-agnostic model gateway
  nutrition/    seed table + resolver (seed → cache → model)
  vision/       image → structured items (separate model)
  db/           schema, migrations, repositories, write serialisation
  cli/          chat loop, onboarding, slash commands
evals/          eval set + grader
benchmarks/     latency harness
docs/           architecture diagram, testing guide
samples/        example food photos
tests/          unit tests
```
