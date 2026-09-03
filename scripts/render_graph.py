"""Render the compiled agent graph.

    python scripts/render_graph.py            # print ASCII + mermaid
    python scripts/render_graph.py docs/agent-graph.md   # also write a file

The output is generated directly from the compiled LangGraph, so it always
matches the real control flow.
"""

from __future__ import annotations

import os
import sys

# A key only needs to exist for config to resolve; no model call is made here.
os.environ.setdefault("SARVAM_API_KEY", "not-used-for-rendering")

from calorai.agent.graph import build_app  # noqa: E402


def main() -> None:
    graph = build_app().get_graph()
    ascii_art = graph.draw_ascii()
    mermaid = graph.draw_mermaid()

    print(ascii_art)
    print()
    print(mermaid)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        content = (
            "# CalorAI agent graph\n\n"
            "Generated from the compiled LangGraph (`scripts/render_graph.py`).\n\n"
            "```mermaid\n" + mermaid + "\n```\n\n"
            "```\n" + ascii_art + "\n```\n"
        )
        with open(path, "w") as fh:
            fh.write(content)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
