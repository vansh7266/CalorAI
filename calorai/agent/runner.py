"""Entry point for one conversation turn.

Sets the per-turn context, runs the graph for the given thread, and returns the
agent's reply. `thread_id` is the user id, so each user gets an isolated,
persisted conversation.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from calorai.agent.context import TurnContext, new_turn_id, reset_context, set_context
from calorai.agent.graph import build_app


def _initial_state(user_text: str, *, user_id: str, turn_id: str, timezone_name: str, image_path: str | None) -> dict:
    return {
        "messages": [HumanMessage(content=user_text)],
        "user_id": user_id,
        "turn_id": turn_id,
        "timezone_name": timezone_name,
        "user_text": user_text,
        "image_path": image_path,
    }


def _turn_context(user_id: str, timezone_name: str) -> TurnContext:
    return TurnContext(
        user_id=user_id,
        turn_id=new_turn_id(),
        timezone_name=timezone_name,
        now_utc=datetime.now(timezone.utc),
    )


def run_turn(
    user_text: str,
    *,
    user_id: str,
    thread_id: str | None = None,
    timezone_name: str = "UTC",
    image_path: str | None = None,
) -> str:
    """Run one turn and return the agent's final reply as a string."""
    ctx = _turn_context(user_id, timezone_name)
    token = set_context(ctx)
    try:
        app = build_app()
        config = {"configurable": {"thread_id": thread_id or user_id}}
        state = _initial_state(
            user_text, user_id=user_id, turn_id=ctx.turn_id, timezone_name=timezone_name, image_path=image_path
        )
        result = app.invoke(state, config)
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                return message.content if isinstance(message.content, str) else str(message.content)
        return "(no reply)"
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
    ctx = _turn_context(user_id, timezone_name)
    token = set_context(ctx)
    try:
        app = build_app()
        config = {"configurable": {"thread_id": thread_id or user_id}}
        state = _initial_state(
            user_text, user_id=user_id, turn_id=ctx.turn_id, timezone_name=timezone_name, image_path=image_path
        )

        pending: list[str] = []
        for chunk, meta in app.stream(state, config, stream_mode="messages"):
            if meta.get("langgraph_node") != "agent":
                continue
            if getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None):
                pending.clear()  # this agent turn is a tool call, not the answer
                continue
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                pending.append(text)
                yield text

        if not pending:
            # Nothing streamed (e.g. the provider returned the final message whole).
            # Read it from the checkpointed state rather than re-running the turn.
            snapshot = app.get_state(config)
            for message in reversed(snapshot.values.get("messages", [])):
                if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                    yield message.content if isinstance(message.content, str) else str(message.content)
                    break
    finally:
        reset_context(token)
