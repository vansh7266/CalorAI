"""Keep replies as plain messaging-app text.

The system prompt already tells the model not to use markdown; this is the
belt-and-braces pass so a stray ``**`` never reaches the user.
"""

from __future__ import annotations

import re

_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)([^*\n]+?)\*(?![*\w])|(?<![_\w])_(?!\s)([^_\n]+?)_(?![_\w])")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s{0,3}\d+\.\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_BACKTICKS = re.compile(r"`+")
_EXTRA_BLANKS = re.compile(r"\n{3,}")


def clean_reply(text: str) -> str:
    """Strip markdown formatting from a complete reply."""
    if not text:
        return text
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _NUMBERED.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _BACKTICKS.sub("", text)
    text = _EXTRA_BLANKS.sub("\n\n", text)
    return text.strip()


def strip_markdown_chunk(chunk: str) -> str:
    """Cheap per-chunk pass for streaming - handles the common inline markers.
    Cross-chunk splits are covered by the prompt rule."""
    return chunk.replace("**", "").replace("__", "").replace("`", "")
