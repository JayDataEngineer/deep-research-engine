"""Message edit operations.

When a message is edited, all subsequent messages in the conversation
are removed to maintain history consistency. The user can then
regenerate the response from the edited message.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.base import CheckpointMetadata

logger = logging.getLogger(__name__)


async def edit_message(
    checkpointer,
    message_id: str,
    thread_id: str,
    content: str,
) -> dict:
    """Edit a message and update conversation history.

    Args:
        checkpointer: The shared checkpointer instance
        message_id: The ID of the message to edit
        thread_id: The thread ID containing the message
        content: New content for the message

    Returns:
        Dict with success status, message_id, content, and removed_count

    Raises:
        HTTPException: If thread or message not found, or validation fails
    """
    # Validate inputs
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    if content is None:
        raise HTTPException(status_code=400, detail="content is required")

    # Get the current thread state
    state = checkpointer.get_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    messages = state.get("messages", [])

    # Find the message to edit
    message_index = -1
    for i, msg in enumerate(messages):
        if msg.id == message_id:
            message_index = i
            break

    if message_index == -1:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found in thread")

    # Get the original message
    original_message = messages[message_index]

    # Create a new message with updated content
    # LangChain messages are often immutable, so we need to create a new one
    if isinstance(original_message, HumanMessage):
        updated_message = HumanMessage(content=content, id=original_message.id)
    elif isinstance(original_message, AIMessage):
        updated_message = AIMessage(content=content, id=original_message.id)
    else:
        # Fallback for other message types
        updated_message = type(original_message)(content=content, id=original_message.id)

    # Remove all messages after the edited one to maintain history consistency
    # This is necessary because subsequent responses were based on the original content
    updated_messages = messages[:message_index] + [updated_message]

    # We need to update the state properly in LangGraph
    # Since messages have a reducer that appends, we need to create a new checkpoint
    # with a new checkpoint_id to force a fresh state
    new_checkpoint_id = str(uuid.uuid4())

    # Get the current checkpoint to preserve ALL state fields
    current_config = {"configurable": {"thread_id": thread_id}}
    current_tuple = checkpointer.get_tuple(current_config)

    # Start with the current checkpoint state and update only messages
    if current_tuple and current_tuple.checkpoint:
        # Preserve all existing fields, update messages
        current_state = current_tuple.checkpoint
        new_checkpoint = {
            **current_state,
            "messages": updated_messages
        }
    else:
        # Fallback: create minimal valid state
        new_checkpoint = {
            "messages": updated_messages,
            "brain_state": {},
            "thread_id": thread_id,
            "user_id": "default",
            "mode": "companion",
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "config_overrides": {},
        }

    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": new_checkpoint_id}}

    # Create metadata
    metadata = CheckpointMetadata(
        source="edit",
        step=1,
        writes={"edited": message_id},
        parents={},
    )

    # Save using the inner checkpointer's put method
    inner_checkpointer = checkpointer.graph.checkpointer
    inner_checkpointer.put(config, new_checkpoint, metadata)

    logger.info(f"[MESSAGE EDIT] Edited message {message_id} in thread {thread_id}")
    logger.info(f"[MESSAGE EDIT] Removed {len(messages) - len(updated_messages)} subsequent messages")

    return {
        "success": True,
        "message_id": message_id,
        "content": content,
        "removed_count": len(messages) - len(updated_messages),
    }
