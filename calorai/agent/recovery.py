"""Recover tool calls that the model emitted as plain text.

GLM-5.2 occasionally writes a tool call into the message content instead of
returning it as a structured tool call, e.g.

    <tool_call>log_meal<arg_key>items</arg_key><arg_value>[...]</arg_value>
    <arg_key>meal_type</arg_key><arg_value>lunch</arg_value></tool_call>

When that happens the tool never runs and the user sees the raw markup. We
detect it, parse it back into a real tool call, and let the normal tool node
execute it.
"""

from __future__ import annotations

import ast
import json
import re
import secrets

_LEAK_MARKERS = ("<tool_call", "<arg_key>", "<arg_value>", "<function=", "<invoke", "‹tool_call")

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*([A-Za-z_][A-Za-z0-9_]*)(.*?)</tool_call>", re.DOTALL)
_ARG_PAIR = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def looks_like_leaked_tool_call(content: str) -> bool:
    return bool(content) and any(marker in content for marker in _LEAK_MARKERS)


def _coerce(value: str):
    value = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            continue
    return value


def recover_tool_calls(content: str) -> list[dict]:
    """Return recovered tool calls (LangChain tool-call dicts), or []."""
    calls: list[dict] = []

    for name, body in _TOOL_CALL_BLOCK.findall(content):
        args: dict = {}
        pairs = _ARG_PAIR.findall(body)
        if pairs:
            for key, raw in pairs:
                args[key.strip()] = _coerce(raw)
        else:
            match = _JSON_OBJ.search(body)
            if match:
                parsed = _coerce(match.group(0))
                if isinstance(parsed, dict):
                    args = parsed.get("arguments") or parsed.get("parameters") or parsed
        calls.append({"name": name, "args": args, "id": f"recovered_{secrets.token_hex(3)}", "type": "tool_call"})

    return calls


def strip_leaked_markup(content: str) -> str:
    """Remove any leaked tool-call markup so it never reaches the user."""
    cleaned = _TOOL_CALL_BLOCK.sub("", content)
    cleaned = re.sub(r"<arg_key>.*?</arg_value>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"</?(tool_call|arg_key|arg_value|invoke|function)[^>]*>", "", cleaned)
    return cleaned.strip()
