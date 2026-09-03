"""The LangGraph state passed between nodes for one conversation thread."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Conversation. `add_messages` appends/merges rather than replacing.
    messages: Annotated[list[AnyMessage], add_messages]

    # Who / which turn (constant for the turn).
    user_id: str
    turn_id: str
    timezone_name: str

    # Set by the ingest node.
    input_type: str          # "text" | "image" | "image+text"
    user_text: str
    image_path: str | None

    # Set by the context-loading nodes (run before the agent node).
    user_name: str | None    # the name given at onboarding (None for a guest)
    memory_card: str         # compact profile block, always injected (Phase 2)
    today_totals: dict | None
    last_meal: dict | None   # {"id", "description", "items": [{"name", "quantity"}]}
    vision_result: dict | None  # extracted items + confidence (Phase 3)
    context_degraded: list[str] | None  # parts of the context that failed to load this turn

    # Control.
    awaiting_user: bool      # the agent asked a question and is waiting
