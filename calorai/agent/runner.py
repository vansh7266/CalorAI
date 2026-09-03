"""Entry point for one conversation turn.

Sets the per-turn context and runs the graph. The LangGraph thread is always the
user id, so each user gets an isolated, persisted conversation and a caller can't
attach one user's context to another user's thread.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from calorai.agent.context import TurnContext, new_turn_id, reset_context, set_context
from calorai.agent.formatting import clean_reply, strip_markdown_chunk
from calorai.agent.graph import build_app
from calorai.agent.memory import schedule_reflection
from calorai.agent.recovery import looks_like_leaked_tool_call

logger = logging.getLogger("calorai.agent")

GENERIC_ERROR = "Something went wrong on my end - your logged meals are safe. Please try again."


def _source_of(user_text: str, image_path: str | None) -> str:
    if image_path and user_text.strip():
        return "image+text"
    if image_path:
        return "image"
    return "text"


def _initial_state(user_text: str, *, user_id: str, turn_id: str, timezone_name: str, image_path: str | None) -> dict:
    if image_path:
        shown = f"[photo] {user_text}".strip() if user_text else "[photo]"
    else:
        shown = user_text
    return {
        "messages": [HumanMessage(content=shown)],
        "user_id": user_id,
        "turn_id": turn_id,
        "timezone_name": timezone_name,
        "user_text": user_text,
        "image_path": image_path,
        "vision_result": None,
        "context_degraded": None,
        "user_name": None,
    }


def _turn_context(user_id: str, timezone_name: str, *, image_path: str | None = None, user_text: str = "") -> TurnContext:
    return TurnContext(
        user_id=user_id,
        turn_id=new_turn_id(),
        timezone_name=timezone_name,
        now_utc=datetime.now(timezone.utc),
        source=_source_of(user_text, image_path),
    )


def run_turn(
    user_text: str,
    *,
    user_id: str,
    thread_id: str | None = None,  # accepted for API symmetry; the thread is always the user
    timezone_name: str = "UTC",
    image_path: str | None = None,
) -> str:
    """Run one turn and return the agent's final reply as a string. Never raises.
    Returns `GENERIC_ERROR` verbatim on failure so callers can detect it."""
    ctx = _turn_context(user_id, timezone_name, image_path=image_path, user_text=user_text)
    token = set_context(ctx)
    try:
        app = build_app()
        config = {"configurable": {"thread_id": user_id}}
        state = _initial_state(
            user_text, user_id=user_id, turn_id=ctx.turn_id, timezone_name=timezone_name, image_path=image_path
        )
        result = app.invoke(state, config)
        reply = "(no reply)"
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                raw = message.content if isinstance(message.content, str) else str(message.content)
                reply = clean_reply(raw)
                break
        schedule_reflection(user_id, ctx.turn_id, user_text, reply)
        return reply
    except Exception:
        logger.exception("run_turn failed for user %s", user_id)
        return GENERIC_ERROR
    finally:
        reset_context(token)


def stream_turn(
    user_text: str,
    *,
    user_id: str,
    thread_id: str | None = None,
    timezone_name: str = "UTC",
    image_path: str | None = None,
) -> Iterator[str]:
    """Run one turn, yielding chunks of the agent's final reply as they arrive.

    Only tokens from a final agent message (no tool calls) are yielded; tool-call
    planning turns are held back so the user sees one clean answer.
    """
    ctx = _turn_context(user_id, timezone_name, image_path=image_path, user_text=user_text)
    token = set_context(ctx)
    try:
        app = build_app()
        config = {"configurable": {"thread_id": user_id}}
        state = _initial_state(
            user_text, user_id=user_id, turn_id=ctx.turn_id, timezone_name=timezone_name, image_path=image_path
        )

        buffer = ""       # not yet yielded - lets us detect a leaked tool call early
        streamed = ""     # what the user has actually seen
        flushing = False  # once true, this message is confirmed clean; pass chunks straight through
        try:
            for chunk, meta in app.stream(state, config, stream_mode="messages"):
                if meta.get("langgraph_node") != "agent":
                    continue
                if getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None):
                    buffer, streamed, flushing = "", "", False  # tool-call turn, not the answer
                    continue
                text = chunk.content if isinstance(chunk.content, str) else ""
                if not text:
                    continue

                if flushing:
                    streamed += text
                    yield strip_markdown_chunk(text)
                    continue

                buffer += text
                if looks_like_leaked_tool_call(buffer):
                    buffer = ""  # the agent node will recover it; stream nothing here
                    continue
                if len(buffer) > 16:  # enough to have seen a "<tool_call" marker
                    streamed += buffer
                    yield strip_markdown_chunk(buffer)
                    buffer, flushing = "", True

            if buffer and not looks_like_leaked_tool_call(buffer):
                streamed += buffer
                yield strip_markdown_chunk(buffer)

            raw = streamed
            if not raw:
                # Nothing streamed (provider returned the final message whole, or the
                # only agent output was a recovered/leaked tool call). Read the final
                # reply from the checkpointed state rather than re-running the turn.
                snapshot = app.get_state(config)
                for message in reversed(snapshot.values.get("messages", [])):
                    if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                        raw = message.content if isinstance(message.content, str) else str(message.content)
                        yield clean_reply(raw)
                        break
        except Exception:
            logger.exception("stream_turn failed for user %s", user_id)
            if not streamed:
                yield GENERIC_ERROR
            raw = streamed

        schedule_reflection(user_id, ctx.turn_id, user_text, clean_reply(raw))
    finally:
        reset_context(token)
