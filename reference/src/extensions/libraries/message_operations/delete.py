"""Message delete operations.

Removes messages from conversation threads, with proper handling for:
- Assistant messages (also removes preceding user message for coherence)
- Fallback message IDs (like "msg-0", "msg-1")
- Thread deletion when no messages remain
- Message versions cleanup
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import CheckpointMetadata

logger = logging.getLogger(__name__)


async def delete_message(
    checkpointer,
    message_id: str,
    thread_id: str,
) -> dict:
    """Delete a message from a thread.

    Removes the specified message from the conversation.
    If the message is an assistant message, also removes the preceding user message
    to maintain conversation coherence.

    Args:
        checkpointer: The shared checkpointer instance
        message_id: The ID of the message to delete (can be fallback ID like "msg-2")
        thread_id: The thread ID containing the message

    Returns:
        Dict with success status, message_id, removed_count, remaining_messages,
        and thread_deleted flag

    Raises:
        HTTPException: If thread or message not found, or validation fails
    """
    # Validate inputs
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    # Get the current thread state
    state = checkpointer.get_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    messages = state.get("messages", [])

    # Handle fallback IDs like "msg-0", "msg-1", etc.
    if message_id.startswith("msg-"):
        try:
            message_index = int(message_id.split("-")[1])
            if message_index >= len(messages):
                raise HTTPException(
                    status_code=404,
                    detail=f"Message {message_id} not found (index {message_index} >= {len(messages)} messages)"
                )
            message_to_delete = messages[message_index]
        except (ValueError, IndexError):
            raise HTTPException(status_code=404, detail=f"Invalid message ID format: {message_id}")
    else:
        # Find the message by actual ID
        message_index = -1
        message_to_delete = None
        for i, msg in enumerate(messages):
            if msg.id == message_id:
                message_index = i
                message_to_delete = msg
                break

        if message_index == -1:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found in thread")

    # Determine which messages to remove
    indices_to_remove = [message_index]

    # If it's an assistant message, also remove the preceding user message
    if isinstance(message_to_delete, AIMessage) and message_index > 0:
        prev_message = messages[message_index - 1]
        if isinstance(prev_message, HumanMessage):
            indices_to_remove.append(message_index - 1)

    # Remove the messages (in reverse order to preserve indices)
    updated_messages = [
        msg for i, msg in enumerate(messages)
        if i not in indices_to_remove
    ]

    # If no messages remain, delete the entire thread
    if not updated_messages:
        logger.info(f"[MESSAGE DELETE] No messages remaining after deletion, deleting thread {thread_id}")
        # Delete message versions for this thread
        inner_checkpointer = checkpointer.graph.checkpointer
        try:
            conn = inner_checkpointer.conn
            conn.execute(
                "DELETE FROM message_versions WHERE thread_id = ?",
                (thread_id,)
            )
            conn.commit()
            logger.info(f"[MESSAGE DELETE] Deleted all message versions for thread {thread_id}")
        except Exception as e:
            logger.warning(f"[MESSAGE DELETE] Failed to delete message versions for thread: {e}")

        # Delete the entire thread
        inner_checkpointer.delete_thread(thread_id)
        logger.info(f"[MESSAGE DELETE] Deleted thread {thread_id} after removing last message")

        return {
            "success": True,
            "message_id": message_id,
            "removed_count": len(indices_to_remove),
            "remaining_messages": 0,
            "thread_deleted": True,
        }

    # We need to update the state properly in LangGraph
    # Since messages have a reducer that appends, we need to create a new checkpoint
    # with a new checkpoint_id to force a fresh state
    import uuid
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
        source="delete",
        step=1,
        writes={"deleted": message_id},
        parents={},
    )

    # Save using the inner checkpointer's put method
    inner_checkpointer = checkpointer.graph.checkpointer
    inner_checkpointer.put(config, new_checkpoint, metadata)

    # Delete message versions from the message_versions table
    # Only do this if the message_id is a real UUID (not a fallback ID)
    if not message_id.startswith("msg-"):
        try:
            conn = inner_checkpointer.conn
            conn.execute(
                "DELETE FROM message_versions WHERE message_id = ?",
                (message_id,)
            )
            conn.commit()
            logger.info(f"[MESSAGE DELETE] Deleted versions for message {message_id}")
        except Exception as e:
            logger.warning(f"[MESSAGE DELETE] Failed to delete message versions: {e}")

    logger.info(f"[MESSAGE DELETE] Deleted message {message_id} from thread {thread_id}")
    logger.info(f"[MESSAGE DELETE] Removed {len(indices_to_remove)} message(s) total")

    return {
        "success": True,
        "message_id": message_id,
        "removed_count": len(indices_to_remove),
        "remaining_messages": len(updated_messages),
        "thread_deleted": False,
    }
