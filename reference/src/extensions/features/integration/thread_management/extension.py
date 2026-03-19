"""Thread Management Extension

Handles conversation thread management including listing, retrieving, deleting,
and managing thread titles and chain of thought data.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.extensions.core import LogicExtension, ExtensionContext

logger = logging.getLogger(__name__)


class ThreadManagementExtension(LogicExtension):
    """Manage conversation threads.

    Provides endpoints for listing threads, getting thread details,
    deleting threads, and managing thread metadata like titles.
    """

    name = "thread_management"

    async def initialize(self, context: ExtensionContext):
        """Initialize the extension (abstract method)."""
        self.context = context

    async def shutdown(self):
        """Shutdown the extension (abstract method)."""
        pass

    async def process(self, input_data: dict, **kwargs) -> dict:
        """Process method - not used for this API-driven extension."""
        return {}

    async def activate(self, context: ExtensionContext):
        """Initialize the extension."""
        await super().activate(context)
        logger.info("[THREAD_MGMT] Activated")

    async def deactivate(self):
        """Clean up resources."""
        await super().deactivate()
        logger.info("[THREAD_MGMT] Deactivated")

    @property
    def router(self) -> Optional[APIRouter]:
        """Get the FastAPI router for this extension."""
        router = APIRouter(prefix="/api/v1/threads", tags=["threads"])

        @router.get("")
        async def list_threads(limit: int = 50, before: Optional[str] = None):
            """List all conversation threads with pagination.

            Args:
                limit: Maximum number of threads to return (default: 50, max: 100)
                before: Return threads created before this timestamp (ISO 8601 format) for pagination

            Returns:
                Paginated list of threads sorted by created_at descending (newest first)
            """
            from src.shared.checkpointer import get_shared_checkpointer

            try:
                # Cap limit to prevent excessive responses
                limit = min(limit, 100)

                logger.info(f"[THREADS] Listing threads with limit={limit}, before={before}...")
                checkpointer = await get_shared_checkpointer()
                threads = checkpointer.list_all_threads(limit=limit, before=before)
                logger.info(f"[THREADS] Got {len(threads)} threads")

                return {
                    "threads": threads,
                    "limit": limit,
                    "has_more": len(threads) == limit  # If we got exactly limit, there might be more
                }
            except Exception as e:
                logger.error(f"Error listing threads: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list threads: {str(e)}")

        @router.get("/{thread_id}")
        async def get_thread(thread_id: str, branch_thread_id: Optional[str] = None):
            """Get details of a specific thread.

            Returns full message data including parts array for Vercel AI SDK compatibility.
            Merges messages from checkpoint (human) and message_versions (AI) tables.

            IMPORTANT: AI messages from streaming are saved to message_versions table,
            not to checkpoints. We must merge both sources to get complete conversation.

            Args:
                thread_id: The thread ID to fetch
                branch_thread_id: Optional. If provided, fetches messages from this branch thread
                                 instead of the main thread. Used when viewing a specific branch.
            """
            from src.shared.checkpointer import get_shared_checkpointer
            from langchain_core.messages import HumanMessage, AIMessage
            import json

            try:
                checkpointer = await get_shared_checkpointer()

                # If branch_thread_id is provided, fetch from that branch instead
                actual_thread_id = branch_thread_id if branch_thread_id else thread_id

                if branch_thread_id:
                    logger.info(f"[THREADS] Fetching branch thread: branch_thread_id={branch_thread_id}, original_thread_id={thread_id}")

                state = checkpointer.get_state(actual_thread_id)
                stats = checkpointer.get_thread_stats(actual_thread_id)

                # Get messages with versions from the checkpointer (includes AI messages)
                # Use actual_thread_id to fetch from the correct thread (branch or main)
                messages_with_versions = checkpointer.get_thread_messages_with_versions(actual_thread_id)

                # Build a map for quick version lookup
                message_id_to_versions = {}
                for msg_with_versions in messages_with_versions:
                    msg_id = msg_with_versions["message_id"]
                    message_id_to_versions[msg_id] = msg_with_versions

                logger.info(f"[THREADS] Found {len(message_id_to_versions)} messages in versions table")

                # Convert to frontend-compatible format
                formatted_messages = []
                assistant_message_count = 0

                # First, iterate through checkpoint messages (usually human messages)
                checkpoint_messages = state.get("messages", []) if state else []

                for msg in checkpoint_messages:
                    msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, 'type', 'unknown')

                    # Skip ToolMessages - they are internal to LangGraph execution
                    if msg_type == "tool":
                        continue

                    # Try to get message_id from message object
                    msg_id = None
                    try:
                        msg_id = msg.id if hasattr(msg, 'id') else msg.get('id') if isinstance(msg, dict) else None
                    except Exception:
                        pass

                    # Extract parts from message (Vercel AI SDK format)
                    parts = []
                    try:
                        if hasattr(msg, 'content') and isinstance(msg.content, list):
                            parts = msg.content
                        elif isinstance(msg, dict) and isinstance(msg.get('content'), list):
                            parts = msg['content']
                        # If content is a string (not a list), convert to text part
                        elif hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                            parts = [{"type": "text", "text": msg.content}]
                        elif isinstance(msg, dict) and isinstance(msg.get('content'), str) and msg['content'].strip():
                            parts = [{"type": "text", "text": msg['content']}]
                    except Exception:
                        pass

                    # Extract reasoning_content from additional_kwargs
                    reasoning_content = None
                    try:
                        if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                            reasoning_content = msg.additional_kwargs.get('reasoning_content')
                        elif isinstance(msg, dict) and isinstance(msg.get('additional_kwargs'), dict):
                            reasoning_content = msg['additional_kwargs'].get('reasoning_content')
                    except Exception:
                        pass

                    # If we have reasoning_content but no reasoning part, create one
                    if reasoning_content and not any(p.get('type') == 'reasoning' for p in parts if isinstance(p, dict)):
                        parts.append({
                            "type": "reasoning",
                            "text": reasoning_content
                        })

                    # Filter parts to include SDK-compatible parts:
                    # - text, reasoning parts (must have 'text' field with content)
                    # - tool-* parts (toolCallId, toolName, input/output)
                    # - data-* parts (custom data parts)
                    # Skip parts with empty 'text' field
                    filtered_parts = []
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get('type', '')

                        # Include text/reasoning parts with non-empty text
                        if part_type in ('text', 'reasoning'):
                            text_content = part.get('text', '')
                            if text_content and isinstance(text_content, str) and text_content.strip():
                                filtered_parts.append(part)

                        # Include tool parts (tool-input-available, tool-output-available, etc.)
                        elif part_type and (part_type.startswith('tool-') or part_type.startswith('data-')):
                            filtered_parts.append(part)

                    # Check for tool_calls in metadata and convert to parts BEFORE building message
                    # This must be done before building formatted_message so tool parts are included
                    metadata = None
                    if msg_id and msg_id in message_id_to_versions:
                        versions_data = message_id_to_versions[msg_id]
                        versions = versions_data.get("versions")
                        if versions:
                            active_version = None
                            for v in versions:
                                if v.get("is_active"):
                                    active_version = v
                                    break
                            if active_version:
                                metadata = active_version.get("metadata", {})
                                tool_calls = metadata.get("tool_calls", []) if metadata else []
                                if tool_calls:
                                    # Convert tool_calls to SDK tool parts format
                                    # SDK format: {type: "tool-{tool_name}", state: "...", toolCallId, ...}
                                    for tc in tool_calls:
                                        if isinstance(tc, dict):
                                            tool_name = tc.get("toolName", "tool")
                                            tc_state = tc.get("state", "output-available")

                                            # Map our states to SDK states
                                            # Our: "executing" → SDK: "input-available"
                                            # Our: "output-available" → SDK: "output-available"
                                            # Our: "output-error" → SDK: "output-error"
                                            if tc_state == "executing":
                                                sdk_state = "input-available"
                                            elif tc_state == "output-error":
                                                sdk_state = "output-error"
                                            else:
                                                sdk_state = "output-available"

                                            # SDK format: type = "tool-{tool_name}", separate state property
                                            tool_part = {
                                                "type": f"tool-{tool_name}",
                                                "state": sdk_state,
                                                "toolCallId": tc.get("toolCallId", ""),
                                            }

                                            # Add input or output depending on state
                                            if tc_state == "executing":
                                                tool_part["input"] = tc.get("input", {})
                                            else:
                                                # output-available or output-error
                                                if tc.get("errorText"):
                                                    tool_part["errorText"] = tc["errorText"]
                                                tool_part["output"] = tc.get("output", "")

                                            filtered_parts.append(tool_part)

                    # Build formatted message (SDK format: parts array only, no content field)
                    formatted_message = {
                        "id": msg_id or f"msg-{len(formatted_messages)}",
                        "type": msg_type,
                        "parts": filtered_parts if filtered_parts else None,
                    }

                    if reasoning_content:
                        formatted_message["reasoning_content"] = reasoning_content

                    # Keep metadata for compatibility (frontend may still read it)
                    if metadata:
                        formatted_message["metadata"] = metadata

                    formatted_messages.append(formatted_message)

                    if msg_type == "ai":
                        assistant_message_count += 1

                # Second, add AI messages from versions table that aren't in checkpoint
                # This is critical because streaming AI messages are saved to message_versions
                # but may not be in the checkpoint state
                # IMPORTANT: Only add assistant messages from versions table!
                # Human messages are already in checkpoint, so we skip them to avoid duplicates.
                used_message_ids = set()
                for msg in formatted_messages:
                    if msg["id"] and not msg["id"].startswith("msg-"):
                        used_message_ids.add(msg["id"])

                for msg_with_versions in messages_with_versions:
                    msg_id = msg_with_versions["message_id"]
                    versions = msg_with_versions.get("versions", [])
                    if not versions:
                        continue

                    # Skip if we already added this message from checkpoint
                    if msg_id in used_message_ids:
                        continue

                    # Find active version
                    active_version = None
                    for v in versions:
                        if v.get("is_active"):
                            active_version = v
                            break

                    if not active_version:
                        continue

                    # CRITICAL: Only add assistant (AI) messages from versions table
                    # Human messages are already in checkpoint - adding them here creates duplicates!
                    if active_version.get("role") != "assistant":
                        continue

                    # Extract parts from active version
                    parts = active_version.get("parts", [])
                    if not parts or not isinstance(parts, list):
                        parts = []

                    # Get reasoning_content
                    reasoning_content = active_version.get("reasoning_content")

                    # If we have reasoning_content but no reasoning part, create one
                    if reasoning_content and not any(p.get('type') == 'reasoning' for p in parts if isinstance(p, dict)):
                        parts.append({
                            "type": "reasoning",
                            "text": reasoning_content
                        })

                    # Filter parts to include SDK-compatible parts:
                    # - text, reasoning parts (must have 'text' field with content)
                    # - tool-* parts (toolCallId, toolName, input/output)
                    # - data-* parts (custom data parts)
                    # Skip parts with empty 'text' field
                    filtered_parts = []
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get('type', '')

                        # Include text/reasoning parts with non-empty text
                        if part_type in ('text', 'reasoning'):
                            text_content = part.get('text', '')
                            if text_content and isinstance(text_content, str) and text_content.strip():
                                filtered_parts.append(part)

                        # Include tool parts (tool-input-available, tool-output-available, etc.)
                        elif part_type and (part_type.startswith('tool-') or part_type.startswith('data-')):
                            filtered_parts.append(part)

                    # Convert metadata.tool_calls to SDK tool parts BEFORE building message
                    # SDK format: {type: "tool-{tool_name}", state: "...", toolCallId, ...}
                    metadata = active_version.get("metadata", {})
                    tool_calls = metadata.get("tool_calls", []) if metadata else []
                    if tool_calls:
                        # Convert tool_calls to SDK tool parts format
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tool_name = tc.get("toolName", "tool")
                                tc_state = tc.get("state", "output-available")

                                # Map our states to SDK states
                                # Our: "executing" → SDK: "input-available"
                                # Our: "output-available" → SDK: "output-available"
                                # Our: "output-error" → SDK: "output-error"
                                if tc_state == "executing":
                                    sdk_state = "input-available"
                                elif tc_state == "output-error":
                                    sdk_state = "output-error"
                                else:
                                    sdk_state = "output-available"

                                # SDK format: type = "tool-{tool_name}", separate state property
                                tool_part = {
                                    "type": f"tool-{tool_name}",
                                    "state": sdk_state,
                                    "toolCallId": tc.get("toolCallId", ""),
                                }

                                # Add input or output depending on state
                                if tc_state == "executing":
                                    tool_part["input"] = tc.get("input", {})
                                else:
                                    # output-available or output-error
                                    if tc.get("errorText"):
                                        tool_part["errorText"] = tc["errorText"]
                                    tool_part["output"] = tc.get("output", "")

                                filtered_parts.append(tool_part)

                    # Build formatted message (SDK format: parts array only, no content field)
                    formatted_message = {
                        "id": msg_id or f"msg-{len(formatted_messages)}",
                        "type": "ai",
                        "parts": filtered_parts if filtered_parts else None,
                    }

                    if reasoning_content:
                        formatted_message["reasoning_content"] = reasoning_content

                    # Keep metadata for compatibility (frontend may still read it)
                    if metadata:
                        formatted_message["metadata"] = metadata

                    formatted_messages.append(formatted_message)

                # Sort messages by version order to maintain correct conversation flow
                # Messages are already in correct order from checkpoint, and appended versions
                # are in order by first_version from get_thread_messages_with_versions

                logger.info(f"[THREADS] Returning {len(formatted_messages)} messages (checkpoint={len(checkpoint_messages)}, versions={len(message_id_to_versions)})")

                return {
                    "thread_id": thread_id,  # Original thread ID
                    "actual_thread_id": actual_thread_id,  # The thread we actually fetched from (could be a branch)
                    "branch_thread_id": branch_thread_id,  # Included for clarity
                    "stats": stats,
                    "messages": formatted_messages,
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting thread {thread_id}: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to get thread: {str(e)}")

        @router.delete("/{thread_id}")
        async def delete_thread(thread_id: str):
            """Delete a specific thread.

            Idempotent: returns success even if thread doesn't exist (already deleted).
            """
            from src.shared.checkpointer import get_shared_checkpointer

            try:
                checkpointer = await get_shared_checkpointer()
                # Delete is idempotent - success even if thread doesn't exist
                checkpointer.delete_thread(thread_id)
                return {"success": True, "thread_id": thread_id, "existed": True}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting thread {thread_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete thread: {str(e)}")

        @router.post("/generate-title")
        async def generate_thread_title(request: dict, background_tasks: BackgroundTasks):
            """Generate a title for a thread based on the first user message.

            Runs in parallel with chat response - triggered immediately on first user message.
            Non-blocking, fast generation optimized for speed (no CoT, max 20 tokens).
            Uses the centralized LLM factory for hot-swappable backend support.

            IMPORTANT: Title generation only happens ONCE per thread. If a title already
            exists, this endpoint returns early without calling the LLM.
            """
            from src.shared.checkpointer import get_shared_checkpointer
            from src.extensions.libraries.title_generator import get_title_generator

            thread_id = request.get("thread_id")
            first_message = request.get("first_message", "")
            ai_response = request.get("ai_response", "")
            fallback_title = request.get("fallbackTitle", "")

            if not thread_id:
                raise HTTPException(status_code=400, detail="thread_id is required")
            if not first_message:
                raise HTTPException(status_code=400, detail="first_message is required")

            try:
                checkpointer = await get_shared_checkpointer()

                # Check if title already exists in database - skip LLM generation if it does
                if checkpointer.has_thread_title(thread_id):
                    existing_title = checkpointer.get_thread_title(thread_id)
                    logger.info(f"[TITLE] Title already exists for thread {thread_id}: {existing_title}, skipping generation")
                    return {"thread_id": thread_id, "title": existing_title}

                title_gen = get_title_generator()

                if not title_gen.is_available():
                    logger.warning("[TITLE] Model not available, using fallback")
                    title = fallback_title if fallback_title else (first_message[:40] + "..." if len(first_message) > 40 else first_message)
                else:
                    title = await title_gen.generate_title_async(
                        user_message=first_message,
                        ai_response=ai_response,
                        fallback_title=fallback_title,
                    )

                checkpointer.set_thread_title_auto_generated(thread_id, title)

                logger.info(f"[TITLE] Generated title for thread {thread_id}: {title}")

                return {"thread_id": thread_id, "title": title}

            except Exception as e:
                logger.error(f"Error generating title for thread {thread_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate title: {str(e)}")

        @router.patch("/{thread_id}/title")
        async def update_thread_title(thread_id: str, request: dict):
            """Update the title of a thread (manual edit by user)."""
            from src.shared.checkpointer import get_shared_checkpointer

            title = request.get("title")

            if not title:
                raise HTTPException(status_code=400, detail="title is required")
            if len(title) > 100:
                raise HTTPException(status_code=400, detail="title must be 100 characters or less")

            try:
                checkpointer = await get_shared_checkpointer()
                success = checkpointer.update_thread_title(thread_id, title)

                if not success:
                    raise HTTPException(status_code=500, detail="Failed to update title")

                logger.info(f"[TITLE] Updated title for thread {thread_id}: {title}")

                return {"thread_id": thread_id, "title": title}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating title for thread {thread_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")

        @router.post("/{thread_id}/cot")
        async def save_chain_of_thought(thread_id: str, request: dict):
            """Save chain of thought (CoT) reasoning steps for a thread.

            Filters out tool router events, only storing actual reasoning steps.
            Automatically called after each message completes.
            """
            from src.shared.checkpointer import get_shared_checkpointer
            import time

            checkpoint_id = request.get("checkpoint_id")
            steps = request.get("steps", [])

            if not steps:
                raise HTTPException(status_code=400, detail="steps is required")

            # Generate checkpoint_id if not provided (use current timestamp)
            if not checkpoint_id:
                checkpoint_id = str(int(time.time() * 1000))

            try:
                checkpointer = await get_shared_checkpointer()
                success = await checkpointer.asave_chain_of_thought(thread_id, checkpoint_id, steps)

                if not success:
                    raise HTTPException(status_code=500, detail="Failed to save chain of thought")

                logger.info(f"[CoT] Saved {len(steps)} steps for thread {thread_id}")

                return {
                    "success": True,
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "steps_saved": len(steps)
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error saving CoT for thread {thread_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to save chain of thought: {str(e)}")

        @router.get("/{thread_id}/cot")
        async def get_chain_of_thought(thread_id: str, checkpoint_id: str = None):
            """Get chain of thought (CoT) reasoning steps for a thread.

            Returns all reasoning steps for the thread, optionally filtered by checkpoint_id.
            """
            from src.shared.checkpointer import get_shared_checkpointer

            try:
                checkpointer = await get_shared_checkpointer()
                steps = await checkpointer.aget_chain_of_thought(thread_id, checkpoint_id)

                logger.info(f"[CoT] Retrieved {len(steps)} steps for thread {thread_id}")

                return {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "steps": steps
                }

            except Exception as e:
                logger.error(f"Error getting CoT for thread {thread_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get chain of thought: {str(e)}")

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(ThreadManagementExtension)
