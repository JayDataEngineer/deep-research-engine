"""Base State Contract - The I/O Contract for all modes.

All 4 app modes must implement this contract for Vercel AI SDK compatibility.
Each mode can extend this with app-specific fields, but these fields MUST exist.
"""

from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage


def add_messages_reducer(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer for messages - appends new messages to existing list.

    This is the standard LangGraph message reducer used in state definitions.
    It appends new messages (right) to the existing conversation history (left).

    Args:
        left: Existing message list
        right: New messages to append

    Returns:
        Combined message list with new messages appended
    """
    # If left is empty or this is the first turn, start with right
    if not left:
        return right
    # If right has new messages, append them
    return left + right


class BaseState(TypedDict):
    """The minimum contract for all apps.

    All 4 modes must implement this. They can extend it with
    app-specific fields, but these fields MUST exist.
    """
    # Core messages (LangChain format) - with reducer
    messages: Annotated[List[BaseMessage], add_messages_reducer]

    # User input for this turn
    user_input: str

    # Thread ID for persistence
    thread_id: str

    # Model information for tracking and persistence
    model: Optional[str]  # Model name (e.g., "glm-4.7", "gpt-4")
    provider: Optional[str]  # Provider name (e.g., "zai", "openai", "local")

    # Timestamps
    start_time: float
    processing_time: float | None
