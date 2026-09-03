"""Render the compiled agent graph.

    python scripts/render_graph.py

Writes docs/agent-graph.md (mermaid + ASCII) and tries docs/agent-graph.png
(via LangGraph's draw_mermaid_png, which needs internet). Everything is
generated directly from the compiled LangGraph, so it always matches the real
control flow.
"""

from __future__ import annotations

import os
from pathlib import Path

# A key only needs to exist for config to resolve; no model call is made here.
os.environ.setdefault("SARVAM_API_KEY", "not-used-for-rendering")

from calorai.agent.graph import build_app  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    graph = build_app().get_graph()
    ascii_art = graph.draw_ascii()
    mermaid = graph.draw_mermaid()

    print(ascii_art)
    print()
    print(mermaid)

    DOCS.mkdir(exist_ok=True)
    md = DOCS / "agent-graph.md"
    md.write_text(
        "# CalorAI agent graph\n\n"
        "Generated from the compiled LangGraph with `python scripts/render_graph.py`.\n\n"
        "```mermaid\n" + mermaid + "\n```\n\n"
        "```text\n" + ascii_art + "\n```\n"
    )
    print(f"\nwrote {md}")

    try:
        png = graph.draw_mermaid_png()
        (DOCS / "agent-graph.png").write_bytes(png)
        print(f"wrote {DOCS / 'agent-graph.png'}")
    except Exception as exc:
        print(f"(png skipped: {exc})")


if __name__ == "__main__":
    main()
