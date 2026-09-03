"""Test-suite setup.

This runs before any test module - and therefore before ``calorai.config`` is
imported and calls ``load_dotenv()``. We force LangSmith / LangChain tracing off
here so a developer whose local ``.env`` enables tracing does not have the unit
suite make network calls (and print redacted credential metadata) on every run.
``load_dotenv()`` does not override variables that are already set, so setting
them now wins. A test that genuinely exercises tracing can re-enable it locally.
"""

from __future__ import annotations

import os

for _var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_var] = "false"
