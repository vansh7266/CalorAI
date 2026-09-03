# CalorAI Logging Agent

A conversational agent that logs meals the way people actually talk about food —
text or photo, in plain language, no forms.

> 🚧 **Work in progress.** This README is written last; see `PROJECT_LOG.md` for the
> full design record and decision rationale while the build is underway.

## Status

| Phase | State |
|-------|-------|
| 0. Scaffold | ✅ |
| 1. Text core (tools, agent, CLI) | ⏳ |
| 2. Memory | ⏳ |
| 3. Image path | ⏳ |
| 4. Onboarding + polish | ⏳ |
| 5. Evals | ⏳ |
| 6. Latency benchmarks | ⏳ |
| 7. README (final) | ⏳ |

## Quick start (will be finalised in Phase 7)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your API key into .env
python cli.py
```
