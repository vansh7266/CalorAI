# Testing CalorAI by hand

## Setup (once)

```bash
cd /path/to/CalorAI                 # the repo root (where cli.py is)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # paste your key, keep MODEL_PROFILE=sarvam
python cli.py
```

Every command below assumes you are in the repo root with the venv activated:

```bash
cd /path/to/CalorAI && source .venv/bin/activate
```

First run asks your name and prints your `usr_...` id. Resume later with
`python cli.py --user usr_xxxxxxxx`. To test as a brand-new user, delete
`data/.session` (or pass a name it hasn't seen).

## The task's test conversation

Run these in one session, in order. Expected behaviour in brackets.

1. `had 2 parathas and chai for breakfast`
   [logs 2 parathas + chai as breakfast, states it assumed plain parathas]
2. `leftover biryani, maybe two thirds of the box`
   [logs an estimated portion of biryani, says it estimated]
3. `skipped lunch but grazed all afternoon`
   [does NOT log a meal; acknowledges and offers to put a rough number on it]
4. `same as yesterday`  (first log something "yesterday" — see below)
5. `actually that was 3 rotis not 2`
   [updates the roti quantity; total goes up, not double-counted]
6. `how much protein have I had today?`
   [answers with a number, no tool call needed for today]
7. `how am I doing on calories?`
   [answers from context]
8. `/img samples/paratha_plate.png`
   [vision model reads the plate, logs it as one meal]
9. `/img samples/pork_plate.jpg half of this was my brother's`
   [ONE meal, all portions halved]
10. `my usual`
    [if nothing saved: asks what your usual is and offers to remember it]
11. `i'm vegetarian btw`
    [acknowledges; from now on it questions any meat]

For step 4, first send `had rajma and rice for lunch yesterday`, then
`for lunch today, same as yesterday`.

## The three that matter most

- **Correction (step 5):** run `/history` before and after — the roti row's
  quantity changes, no second meal appears, `/totals` updates.
- **Memory (steps 10-11):** save a usual (`my usual breakfast is 2 eggs and toast`),
  then in a NEW session (`python cli.py --user <your id>`) say `log my usual` —
  it logs the saved items. `/memory` shows what it knows.
- **Photo + caption (step 9):** `/history` shows one meal with halved portions.

## Edge cases worth trying

- `actually make that 4` right after a correction to 3 (updates, doesn't stack)
- `remove the chai from breakfast`
- `that biryani was lunch not dinner`
- `how many calories in a samosa?` (answers, does NOT log)
- `what's the weather?` (friendly redirect, no log)
- `asdfkj` (treats it as an accidental text)
- `had eggs for breakfast, salad for lunch, pasta for dinner` (logs three meals)
- send `/img` on a non-food photo (says it doesn't look like food, asks)
- `im vegetarian` then `had a chicken sandwich` (questions it, does not log)
- start a second user (`python cli.py --user usr_new` or a fresh name) and confirm
  totals are independent

## Useful commands

```
/totals [date]     /history [date]     /memory     /forget <id>
/whoami            /img <path> [cap]    /help       /quit
```

## Automated checks

```bash
cd /path/to/CalorAI && source .venv/bin/activate
pytest -q                    # 69 unit tests
python evals/run.py          # 22 scenario evals
python benchmarks/latency.py # p50/p95 for both paths
```
