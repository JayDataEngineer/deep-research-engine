"""Deep Research Extension Routes.

API endpoints for managing deep research configuration and state.
These endpoints were moved from universal.py to follow the plugin architecture.

ARCHITECTURE:
- Config storage: Uses context_engine S3Backend via registry tools
- Local fallback: Filesystem for dev/offline mode
- Session isolation: User configs in /configs/{user_id}/
"""

from pathlib import Path
import logging
import sys
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent))
from .logging_utils import get_logger

import yaml
import json

from fastapi import APIRouter, HTTPException

from .constants import RESEARCH_EXTENSIONS_DIR

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level

# Create routers
deep_research_router = APIRouter(prefix="/api/v1/deep-research", tags=["deep-research"])
research_router = APIRouter(prefix="/api/v1/research", tags=["research"])


# =============================================================================
# CONTEXT ENGINE STORAGE HELPERS
# =============================================================================

def _get_config_backend():
    """Get MinioBackend for config storage.

    Returns:
        MinioBackend instance for configs
    """
    from .storage.minio_helpers import get_minio_backend
    return get_minio_backend(prefix="configs")


def _save_config_to_storage(path: str, content: str) -> str:
    """Save config content to context_engine or local file.

    Args:
        path: Storage path (e.g., "/user_123/research_variants/rag/custom.yaml")
        content: YAML config content

    Returns:
        Path where config was stored
    """
    # Try context_engine tool first
    try:
        from src.shared.tools.registry import get_tool
        save_raw_file = get_tool("save_raw_file")
        if save_raw_file:
            try:
                full_path = f"/configs{path}"
                save_raw_file.invoke({"path": full_path, "content": content})
                return path
            except Exception as e:
                logger.debug(f"Context engine save failed, using local: {e}")
    except ImportError:
        pass

    # Fallback: MinioBackend
    try:
        backend = _get_config_backend()
        result = backend.write(path, content)
        if not result.error:
            return path
    except Exception as e:
        logger.debug(f"MinioBackend save failed, using local: {e}")

    # Last resort: local file
    from src.constants import USER_CONFIG_DIR
    local_path = USER_CONFIG_DIR / path.lstrip("/")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(content)
    return str(local_path)


def _read_config_from_storage(path: str) -> Optional[str]:
    """Read config content from context_engine or local file.

    Args:
        path: Storage path

    Returns:
        Config content or None if not found
    """
    # Try MinioBackend first
    try:
        backend = _get_config_backend()
        content = backend.read(path)
        # read() returns formatted with line numbers, get raw content
        if content and not content.startswith("Error:"):
            # Extract just the content (strip line numbers)
            lines = []
            for line in content.split("\n"):
                if "\t" in line:
                    lines.append(line.split("\t", 1)[1])
                else:
                    lines.append(line)
            return "\n".join(lines)
    except Exception:
        pass

    # Fallback: local file
    from src.constants import USER_CONFIG_DIR
    local_path = USER_CONFIG_DIR / path.lstrip("/")
    if local_path.exists():
        return local_path.read_text()

    return None


def _delete_config_from_storage(path: str) -> bool:
    """Delete config from context_engine or local file.

    Args:
        path: Storage path

    Returns:
        True if deleted, False if not found
    """
    # Try MinioBackend first
    try:
        backend = _get_config_backend()
        return backend.s3_backend.delete(f"/configs{path}")
    except Exception:
        pass

    # Fallback: local file
    from src.constants import USER_CONFIG_DIR
    local_path = USER_CONFIG_DIR / path.lstrip("/")
    if local_path.exists():
        local_path.unlink()
        return True

    return False


# =============================================================================
# Deep Research Config Management Endpoints
# =============================================================================

@deep_research_router.get("/use-cases")
async def list_deep_research_use_cases():
    """List all available deep research use cases and their variants from research extensions."""
    try:
        use_cases = []

        if not RESEARCH_EXTENSIONS_DIR.exists():
            return {"use_cases": []}

        for ext_dir in sorted(RESEARCH_EXTENSIONS_DIR.iterdir()):
            if ext_dir.is_dir() and not ext_dir.name.startswith('_'):
                variants = ["default"]  # All extensions have at least default

                # Check for variants subdirectory
                variants_dir = ext_dir / "variants"
                if variants_dir.exists():
                    for variant_file in sorted(variants_dir.glob("*.yaml")):
                        variant_name = variant_file.stem
                        if variant_name != "default":
                            variants.append(variant_name)

                use_cases.append({
                    "name": ext_dir.name,
                    "variants": variants
                })

        return {"use_cases": use_cases}
    except Exception as e:
        logger.error(f"Error listing deep research use cases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list use cases: {str(e)}")


@deep_research_router.get("/config")
async def get_deep_research_config(use_case: str = "rag", variant: str = "default"):
    """Get deep research config for a specific use case and variant."""
    from src.extensions.content.deep_research import load_research_config

    try:
        config = load_research_config(use_case=use_case, variant=variant)
        return {
            "use_case": use_case,
            "variant": variant,
            "config": config
        }
    except Exception as e:
        logger.error(f"Error loading deep research config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")


@deep_research_router.post("/config")
async def save_deep_research_config(request: dict):
    """Save deep research config for a specific use case and variant.

    Creates or updates a YAML config file for the specified use_case/variant.
    User variants are saved to context_engine storage or local fallback.
    """
    use_case = request.get("use_case")
    variant = request.get("variant")
    config = request.get("config")
    user_id = request.get("user_id", "default")

    if not use_case:
        raise HTTPException(status_code=400, detail="use_case is required")
    if not variant:
        raise HTTPException(status_code=400, detail="variant is required")
    if not config:
        raise HTTPException(status_code=400, detail="config is required")

    try:
        from src.constants import USER_CONFIG_DIR

        # Build storage path
        if variant != "default":
            # User variants go to /configs/{user_id}/research_variants/{use_case}/{variant}.yaml
            storage_path = f"/configs/{user_id}/research_variants/{use_case}/{variant}.yaml"
            local_dir = USER_CONFIG_DIR / user_id / "research_variants" / use_case
        else:
            # Default variant (override base)
            storage_path = f"/configs/{user_id}/research_variants/{use_case}/config.yaml"
            local_dir = RESEARCH_EXTENSIONS_DIR / use_case

        # Ensure local directory exists for fallback
        local_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata to indicate user-created
        if "_metadata" not in config:
            config["_metadata"] = {}
        config["_metadata"]["user_created"] = True

        # Serialize to YAML
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)

        # Save via context_engine
        stored_path = _save_config_to_storage(storage_path, content)

        logger.info(f"Saved deep research config: {use_case}/{variant} -> {stored_path}")

        return {
            "success": True,
            "use_case": use_case,
            "variant": variant,
            "path": stored_path
        }
    except Exception as e:
        logger.error(f"Error saving deep research config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@deep_research_router.delete("/config")
async def delete_deep_research_config(use_case: str, variant: str, user_id: str = "default"):
    """Delete a deep research config variant.

    Cannot delete the default variant for a use case.
    Only user-created variants can be deleted.
    """
    if variant == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default variant")

    try:
        from src.constants import USER_CONFIG_DIR

        # Build storage path
        storage_path = f"/configs/{user_id}/research_variants/{use_case}/{variant}.yaml"

        # Read config to check if user-created
        config_content = _read_config_from_storage(storage_path)

        if config_content is None:
            # Check local fallback
            local_path = USER_CONFIG_DIR / user_id / "research_variants" / use_case / f"{variant}.yaml"
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"Config variant {variant} not found")
            config_content = local_path.read_text()

        # Parse and check metadata
        config_data = yaml.safe_load(config_content)
        if not config_data.get("_metadata", {}).get("user_created", False):
            raise HTTPException(
                status_code=400,
                detail="Cannot delete built-in variants. Only user-created variants can be deleted."
            )

        # Delete via context_engine or local
        deleted = _delete_config_from_storage(storage_path)

        # Also delete local fallback if exists
        local_path = USER_CONFIG_DIR / user_id / "research_variants" / use_case / f"{variant}.yaml"
        if local_path.exists():
            local_path.unlink()
            deleted = True

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Config variant {variant} not found")

        logger.info(f"Deleted deep research config: {use_case}/{variant}")

        return {
            "success": True,
            "use_case": use_case,
            "variant": variant
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting deep research config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete config: {str(e)}")


# =============================================================================
# Research State Management - Navigation Resume Capability
# =============================================================================

@research_router.get("/state/{thread_id}")
async def get_research_state(thread_id: str):
    """Get current research state for a thread.

    Returns the in-progress research state if it exists, allowing users
    to resume watching research progress after navigating away.
    """
    from src.extensions.content.deep_research.storage import load_research_state

    try:
        state = load_research_state(thread_id)

        if not state:
            raise HTTPException(status_code=404, detail=f"No research state found for thread {thread_id}")

        logger.info(f"[RESEARCH_STATE] Retrieved state for thread_id={thread_id}")

        return {
            "thread_id": thread_id,
            "state": state
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get research state: {str(e)}")


@research_router.delete("/state/{thread_id}")
async def delete_research_state(thread_id: str):
    """Delete research state for a thread.

    Clears the persisted research state, typically after research completes
    or user explicitly dismisses the results.
    """
    from src.extensions.content.deep_research.storage import delete_research_state

    try:
        success = delete_research_state(thread_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"No research state found for thread {thread_id}")

        logger.info(f"[RESEARCH_STATE] Deleted state for thread_id={thread_id}")

        return {
            "success": True,
            "message": f"Research state deleted for thread {thread_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting research state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete research state: {str(e)}")


@research_router.get("/states")
async def list_research_states():
    """List all active (in-progress) research states.

    Returns metadata for all currently stored research sessions.
    """
    from src.extensions.content.deep_research.storage import list_active_states

    try:
        states = list_active_states()

        logger.info(f"[RESEARCH_STATE] Listed {len(states)} active states")

        return {
            "states": states,
            "total_count": len(states)
        }
    except Exception as e:
        logger.error(f"Error listing research states: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list research states: {str(e)}")


# =============================================================================
# Streaming Research Execution - Real-time Progress Updates
# =============================================================================

@research_router.post("/stream")
async def stream_research(request: dict):
    """Execute deep research with streaming progress updates.

    Uses Server-Sent Events (SSE) to send real-time progress updates to the client.
    The client receives updates as each node completes, enabling live progress tracking.

    Request body:
        query: str - Research query
        use_case: str - Use case name (default: "base")
        variant: str - Config variant (default: "default")
        overdrive: bool - Enable max quality mode (default: False)
        thread_id: str - Thread ID for checkpointing (default: auto-generated)

    Returns:
        SSE stream with progress updates (simple JSON format)
    """
    from fastapi.responses import StreamingResponse
    from asyncio import sleep
    import uuid
    from src.shared.sse_events import SSEvent

    query = request.get("query")
    use_case = request.get("use_case", "base")
    variant = request.get("variant", "default")
    overdrive = request.get("overdrive", False)
    thread_id = request.get("thread_id")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    async def event_generator():
        """Generate SSE events for research progress."""
        from src.extensions.content.deep_research.graph import run_v4_spawner
        from src.extensions.content.deep_research.storage import save_research_state

        # Generate thread_id if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())
        research_thread_id = thread_id or f"research_{uuid.uuid4().hex[:16]}"

        # Send initial event
        yield SSEvent.start(research_thread_id)

        try:
            # Create initial state
            initial_state = {
                "raw_query": query,
                "chat_history": [],
            }

            # Get LangFuse callback for observability (if configured)
            from src.extensions.content.deep_research.observability.langfuse_integration import (
                get_langfuse_callback_for_research,
            )
            langfuse_callback = get_langfuse_callback_for_research(
                session_id=research_thread_id,
                use_case=use_case,
                variant=variant,
                query=query,
            )

            # Build callbacks list
            callbacks = []
            if langfuse_callback:
                callbacks.append(langfuse_callback)
                logger.info(f"[STREAMING] LangFuse callback attached for session={research_thread_id}")

            # Create RunLogger for observability (required by graph nodes)
            from src.extensions.content.deep_research.observability import RunLogger
            from pathlib import Path

            run_logger_config = {
                "use_case_config": {
                    "use_case": use_case,
                    "variant": variant,
                },
            }
            run_logger = RunLogger(
                run_name=f"research_{research_thread_id}",
                config=run_logger_config,
                base_dir=Path("data/outputs/logs/deep_research"),
                session_id=research_thread_id,
            )

            # Create config with callbacks for LangGraph tracing
            config = {
                "configurable": {
                    "thread_id": research_thread_id,
                    "run_logger": run_logger,  # Required by graph nodes
                },
                "recursion_limit": 150,  # Prevent infinite loops
                "callbacks": callbacks if callbacks else None,  # LangFuse tracing
            }

            # Import graph building
            from src.extensions.content.deep_research.graph import build_v4_graph

            # Build graph with checkpointer and store (using factory for environment-aware selection)
            from src.shared.checkpointer.factory import get_checkpointer
            checkpointer = await get_checkpointer()
            graph = build_v4_graph(checkpointer=checkpointer)

            # Stream graph execution with updates
            phase_mapping = {
                "worker": "gathering_information",
                "boss": "planning_research",
                "lab_architect": "orchestrating_research",
                "critic": "evaluating_quality",
                "synthesis": "generating_output",
            }

            last_node = None
            worker_count = 0

            # Use propagate_attributes context manager for proper session grouping
            # This ensures all nested LLM calls are grouped under the same session
            try:
                from langfuse import propagate_attributes
                session_context = propagate_attributes(session_id=research_thread_id)
                session_context.__enter__()
            except ImportError:
                # Fallback for older SDK versions
                session_context = None

            try:
                async for chunk, metadata in graph.astream(
                    initial_state,
                    config,
                    stream_mode="updates",
                ):
                    # Extract node name from metadata
                    node_name = metadata.get("langgraph_step", "unknown")

                    # Map to human-readable phase
                    phase = phase_mapping.get(node_name, node_name)

                    # Extract progress info
                    progress_data = {
                        "node": node_name,
                        "phase": phase,
                        "thread_id": research_thread_id,
                    }

                    # Add phase-specific info
                    if node_name == "worker" and "research_brief" in chunk:
                        brief = chunk.get("research_brief", {})
                        if brief:
                            progress_data["research_brief"] = {
                                "topic": brief.get("topic", ""),
                                "key_questions": len(brief.get("key_questions", [])),
                            }

                    elif node_name == "boss":
                        if "workers_to_spawn" in chunk:
                            workers = chunk.get("workers_to_spawn", [])
                            worker_count = len(workers)
                            progress_data["workers_to_spawn"] = worker_count

                    elif node_name == "lab_architect":
                        # Lab Architect orchestrates Flash Explorers
                        if "architect_findings" in chunk:
                            progress_data["lab_completed"] = True
                            findings = chunk.get("architect_findings")
                            if findings:
                                progress_data["avenues_explored"] = len(findings.get("avenues_summary", {}))

                    elif node_name == "critic":
                        if "critique_result" in chunk:
                            critique = chunk.get("critique_result", {})
                            if critique:
                                progress_data["quality_score"] = critique.get("quality_score", 0)
                                progress_data["passed"] = critique.get("passed", False)

                    elif node_name == "synthesis":
                        if "final_response" in chunk:
                            progress_data["response_ready"] = True
                            progress_data["response_length"] = len(chunk.get("final_response", ""))

                    # Skip if no change
                    if node_name != last_node:
                        yield SSEvent.progress(**progress_data)
                        last_node = node_name

                        # Small delay to prevent overwhelming client
                        await sleep(0.05)

                # Send completion event
                yield SSEvent.custom("complete", thread_id=research_thread_id, status="complete")

            finally:
                # Exit session context
                if session_context:
                    session_context.__exit__(None, None, None)

                # Flush LangFuse events to ensure all traces are sent
                if langfuse_callback:
                    try:
                        from src.extensions.integrations.langfuse import flush_langfuse_client
                        flush_langfuse_client()
                        logger.debug(f"[STREAMING] LangFuse events flushed for session={research_thread_id}")
                    except Exception as flush_error:
                        logger.warning(f"[STREAMING] Failed to flush LangFuse: {flush_error}")

        except Exception as e:
            logger.error(f"[STREAMING] Error during research streaming: {e}")
            yield SSEvent.error(str(e), recoverable=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
