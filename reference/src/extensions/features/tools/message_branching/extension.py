"""Message Branching Extension

Handles message versioning, editing, regeneration, and deletion.
Provides support for multiple response versions per message.
"""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from src.extensions.core import LogicExtension, ExtensionContext
from src.config import settings

logger = logging.getLogger(__name__)


class MessageBranchingExtension(LogicExtension):
    """Manage message versions and branching.

    Provides endpoints for:
    - Regenerating messages with new versions
    - Editing messages
    - Switching between message versions
    - Deleting messages
    - Getting message version history
    """

    name = "message_branching"

    async def initialize(self, context: ExtensionContext):
        """Initialize the extension."""
        self.context = context
        logger.info("[MESSAGE_BRANCHING] Initialized")

    async def shutdown(self):
        """Clean up resources."""
        logger.info("[MESSAGE_BRANCHING] Shutdown")

    async def process(self, input_data: dict, **kwargs) -> dict:
        """Process method - not used for this API-driven extension."""
        return {}

    @property
    def router(self) -> Optional[APIRouter]:
        """Get the FastAPI router for this extension."""
        router = APIRouter(prefix="/api/v1/messages", tags=["messages"])

        @router.get("/{message_id}/versions")
        async def get_message_versions(message_id: str):
            """Get all versions of a message.

            Returns all available versions/branches for a specific message.
            Returns empty list if no versions exist (not a 404).
            """
            from src.shared.checkpointer import get_shared_checkpointer

            try:
                checkpointer = await get_shared_checkpointer()
                versions = checkpointer.get_message_versions(message_id) or []

                # Find active version
                active_version_id = None
                for v in versions:
                    if v.get("is_active"):
                        active_version_id = v["version_id"]
                        break

                return {
                    "message_id": message_id,
                    "versions": versions,
                    "active_version_id": active_version_id,
                    "total_count": len(versions)
                }
            except Exception as e:
                logger.error(f"Error getting message versions: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get versions: {str(e)}")

        @router.post("/switch-version")
        async def switch_message_version(request: dict):
            """Switch to a different version of a message.

            Changes which version of a message is currently active/displayed.
            """
            from src.shared.checkpointer import get_shared_checkpointer

            message_id = request.get("message_id")
            version_id = request.get("version_id")

            if not message_id:
                raise HTTPException(status_code=400, detail="message_id is required")
            if not version_id:
                raise HTTPException(status_code=400, detail="version_id is required")

            try:
                checkpointer = await get_shared_checkpointer()
                success = checkpointer.switch_active_version(message_id, version_id)

                if not success:
                    raise HTTPException(status_code=404, detail="Version not found")

                logger.info(f"[MESSAGE VERSION] Switched to version {version_id} for message {message_id}")

                return {
                    "success": True,
                    "message_id": message_id,
                    "active_version_id": version_id
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error switching version: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to switch version: {str(e)}")

        @router.post("/{message_id}/edit")
        async def edit_message(message_id: str, request: dict):
            """Edit a message and update conversation history.

            When a message is edited, all subsequent messages in the conversation
            are removed to maintain history consistency. The user can then
            regenerate the response from the edited message.
            """
            from src.shared.checkpointer import get_shared_checkpointer
            from langgraph.checkpoint.base import CheckpointMetadata
            from datetime import datetime, timezone

            thread_id = request.get("thread_id")
            content = request.get("content")

            if not thread_id:
                raise HTTPException(status_code=400, detail="thread_id is required")
            if content is None:
                raise HTTPException(status_code=400, detail="content is required")

            try:
                checkpointer = await get_shared_checkpointer()

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
                if isinstance(original_message, HumanMessage):
                    updated_message = HumanMessage(content=content, id=original_message.id)
                elif isinstance(original_message, AIMessage):
                    updated_message = AIMessage(content=content, id=original_message.id)
                else:
                    updated_message = type(original_message)(content=content, id=original_message.id)

                # Remove all messages after the edited one
                updated_messages = messages[:message_index] + [updated_message]

                # Create new checkpoint
                new_checkpoint_id = str(uuid.uuid4())

                # Get the current checkpoint to preserve ALL state fields
                current_config = {"configurable": {"thread_id": thread_id}}
                current_tuple = checkpointer.get_tuple(current_config)

                # Start with the current checkpoint state and update only messages
                if current_tuple and current_tuple.checkpoint:
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

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error editing message: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to edit message: {str(e)}")

        @router.delete("/{message_id}")
        async def delete_message(message_id: str, thread_id: str = None):
            """Delete a message from a thread.

            Removes the specified message from the conversation.
            If the message is an assistant message, also removes the preceding user message
            to maintain conversation coherence.
            """
            from src.shared.checkpointer import get_shared_checkpointer
            from langchain_core.messages import AIMessage, HumanMessage
            from langgraph.checkpoint.base import CheckpointMetadata
            from datetime import datetime, timezone

            if not thread_id:
                raise HTTPException(status_code=400, detail="thread_id is required")

            try:
                checkpointer = await get_shared_checkpointer()

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
                            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
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
                    inner_checkpointer = checkpointer.graph.checkpointer
                    try:
                        conn = inner_checkpointer.conn
                        conn.execute(
                            "DELETE FROM message_versions WHERE thread_id = ?",
                            (thread_id,)
                        )
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"[MESSAGE DELETE] Failed to delete message versions for thread: {e}")

                    inner_checkpointer.delete_thread(thread_id)
                    logger.info(f"[MESSAGE DELETE] Deleted thread {thread_id} after removing last message")

                    return {
                        "success": True,
                        "message_id": message_id,
                        "removed_count": len(indices_to_remove),
                        "remaining_messages": 0,
                        "thread_deleted": True,
                    }

                # Update the state
                new_checkpoint_id = str(uuid.uuid4())

                current_config = {"configurable": {"thread_id": thread_id}}
                current_tuple = checkpointer.get_tuple(current_config)

                if current_tuple and current_tuple.checkpoint:
                    current_state = current_tuple.checkpoint
                    new_checkpoint = {
                        **current_state,
                        "messages": updated_messages
                    }
                else:
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

                metadata = CheckpointMetadata(
                    source="delete",
                    step=1,
                    writes={"deleted": message_id},
                    parents={},
                )

                inner_checkpointer = checkpointer.graph.checkpointer
                inner_checkpointer.put(config, new_checkpoint, metadata)

                # Delete message versions from the message_versions table
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

                return {
                    "success": True,
                    "message_id": message_id,
                    "removed_count": len(indices_to_remove),
                    "remaining_messages": len(updated_messages),
                    "thread_deleted": False,
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting message: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete message: {str(e)}")

        @router.post("/regenerate")
        async def regenerate_message(request: dict):
            """Regenerate a message with a new version.

            Creates a new version of an assistant message by re-running the LLM
            with the same conversation context up to that point.
            """
            from src.shared.checkpointer import get_shared_checkpointer

            message_id = request.get("message_id")
            thread_id = request.get("thread_id")
            model = request.get("model")
            temperature = request.get("temperature")

            if not message_id:
                raise HTTPException(status_code=400, detail="message_id is required")
            if not thread_id:
                raise HTTPException(status_code=400, detail="thread_id is required")

            try:
                checkpointer = await get_shared_checkpointer()

                # Get existing versions to determine version number
                existing_versions = checkpointer.get_message_versions(message_id)
                next_version_number = len(existing_versions) + 1

                # Get thread state to understand the mode and context
                state = checkpointer.get_state(thread_id)
                if not state:
                    raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

                messages = state.get("messages", [])

                if not messages:
                    raise HTTPException(status_code=400, detail="No messages in thread")

                # Find the message to regenerate by ID
                message_index = -1
                for i, msg in enumerate(messages):
                    if msg.id == message_id:
                        message_index = i
                        break

                if message_index == -1:
                    raise HTTPException(status_code=404, detail=f"Message {message_id} not found in thread")

                # Build messages up to (but not including) the message to regenerate
                messages_for_regeneration = messages[:message_index]

                if not messages_for_regeneration:
                    raise HTTPException(status_code=400, detail="Cannot regenerate first message")

                # Update checkpoint to remove all messages after the regenerated one
                from langgraph.checkpoint.base import CheckpointMetadata
                from datetime import datetime, timezone

                updated_messages = messages[:message_index]

                # Get current checkpoint to preserve other state
                current_config = {"configurable": {"thread_id": thread_id}}
                current_tuple = checkpointer.get_tuple(current_config)

                new_checkpoint_id = str(uuid.uuid4())

                if current_tuple and current_tuple.checkpoint:
                    current_state = current_tuple.checkpoint
                    new_checkpoint = {
                        **current_state,
                        "messages": updated_messages
                    }
                else:
                    new_checkpoint = {
                        "messages": updated_messages,
                        "brain_state": {},
                        "thread_id": thread_id,
                        "user_id": "default",
                        "mode": state.get("mode", "direct"),
                        "last_activity": datetime.now(timezone.utc).isoformat(),
                        "config_overrides": {},
                    }

                config = {"configurable": {"thread_id": thread_id, "checkpoint_id": new_checkpoint_id}}

                metadata = CheckpointMetadata(
                    source="regenerate",
                    step=1,
                    writes={"regenerated": message_id},
                    parents={},
                )

                # Save updated checkpoint (removes messages after regenerated one)
                inner_checkpointer = checkpointer.graph.checkpointer
                inner_checkpointer.put(config, new_checkpoint, metadata)

                logger.info(f"[REGENERATE] Removed {len(messages) - len(updated_messages)} messages after {message_id}")

                # Filter out messages with empty content (e.g., tool call messages)
                messages_for_regeneration = [
                    msg for msg in messages_for_regeneration
                    if (msg.content if hasattr(msg, 'content') else msg.get("content", ""))
                ]

                # Get the extension for this mode and use its build_state method
                backend_mode = state.get("mode", "companion")
                extension = self.context.get_extension(f"chat_{backend_mode}") or self.context.get_extension(backend_mode)
                if not extension:
                    raise HTTPException(status_code=400, detail=f"Mode {backend_mode} not found")

                # Create a fake request object for build_state
                class FakeRequest:
                    def __init__(self, messages, model):
                        self.messages = messages
                        self.model = model
                        self.session_id = thread_id
                        self.personality = state.get("personality", "friendly")
                        self.personality_enabled = state.get("personality_enabled", True)

                # Use the extension's build_state method
                from src.shared.models.api_schema import Message

                # Convert LangChain messages back to API format
                api_messages = []
                for msg in messages_for_regeneration:
                    if hasattr(msg, 'type'):
                        msg_type = msg.type
                    else:
                        msg_type = msg.get("type", "user")

                    content = msg.content if hasattr(msg, 'content') else msg.get("content", "")

                    if msg_type == "human":
                        api_messages.append(Message(role="user", content=content))
                    elif msg_type == "ai":
                        api_messages.append(Message(role="assistant", content=content))
                    elif msg_type == "system":
                        api_messages.append(Message(role="system", content=content))

                fake_request = FakeRequest(api_messages, model)

                # Build initial state using the extension's method
                initial_state = extension.build_state(fake_request)
                initial_state["thread_id"] = thread_id

                # Create a unique branch_thread_id for this branch BEFORE streaming
                # This thread will contain the messages for this specific branch
                branch_thread_id = f"{thread_id}_branch_{message_id}_{next_version_number}"
                logger.info(f"[MESSAGE_REGEN] Creating branch {next_version_number} with branch_thread_id={branch_thread_id}")

                # Stream the regenerated response
                async def stream_regeneration():
                    # Use the extension's stream_chat method which emits simple SSE events
                    async for chunk in extension.stream_chat(initial_state, config):
                        yield chunk

                    # Save the new version after streaming completes
                    full_content = "".join(content_parts)
                    full_reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

                    # Parse content to extract JSON metadata if present
                    actual_response = full_content
                    extracted_reasoning = None

                    stripped = full_content.strip()
                    if (stripped.startswith('{') or stripped.startswith('[')):
                        if '"tools"' in stripped[:500] and '"reasoning"' in stripped[:500]:
                            try:
                                json_end = 0
                                bracket_count = 0
                                for i, char in enumerate(stripped):
                                    if char == '{':
                                        bracket_count += 1
                                    elif char == '}':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            json_end = i + 1
                                            break

                                if json_end > 0:
                                    metadata_json = json.loads(stripped[:json_end])
                                    if isinstance(metadata_json, dict) and "reasoning" in metadata_json:
                                        extracted_reasoning = metadata_json["reasoning"]
                                    actual_response = stripped[json_end:].strip()
                            except (json.JSONDecodeError, KeyError):
                                logger.warning("[MESSAGE_REGEN] Failed to parse JSON metadata from content")

                    # Use extracted reasoning if available
                    final_reasoning = extracted_reasoning if extracted_reasoning else full_reasoning

                    # Build parts array
                    parts_list = []
                    if final_reasoning:
                        parts_list.append({"type": "reasoning", "text": final_reasoning})
                    parts_list.append({"type": "text", "text": actual_response})

                    # Build tool_calls metadata
                    tool_calls_metadata = []
                    for tool_part in tool_parts:
                        tool_calls_metadata.append({
                            "type": tool_part.get("type", "tool"),
                            "toolName": tool_part.get("toolName", ""),
                            "toolCallId": tool_part.get("toolCallId", ""),
                            "state": tool_part.get("state", "output-available"),
                            "input": tool_part.get("input"),
                            "output": tool_part.get("output"),
                            "errorText": tool_part.get("errorText"),
                        })

                    # The regeneration was streamed to the branch thread above
                    # Now save the version with the branch_thread_id reference
                    # branch_thread_id was created earlier in the function
                    await checkpointer.asave_message_version(
                        message_id=message_id,
                        thread_id=thread_id,
                        version_id=str(uuid.uuid4()),
                        version_number=next_version_number,
                        role="assistant",
                        content=actual_response,
                        parts=parts_list,
                        reasoning_content=final_reasoning,
                        model=model,
                        temperature=temperature,
                        is_active=True,
                        metadata={
                            "tool_calls": tool_calls_metadata,
                            "branch_thread_id": branch_thread_id,  # Links this version to its branch thread
                        } if tool_calls_metadata else {
                            "branch_thread_id": branch_thread_id,
                        },
                    )

                    logger.info(f"[MESSAGE_REGEN] Saved version {next_version_number} of message {message_id} with branch_thread_id={branch_thread_id}")

                return StreamingResponse(
                    stream_regeneration(),
                    media_type="text/event-stream",
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error regenerating message: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to regenerate message: {str(e)}")

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(MessageBranchingExtension)
