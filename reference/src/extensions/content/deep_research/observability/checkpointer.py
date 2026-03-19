"""Checkpointer API routes for LangGraph checkpoint database.

This module provides endpoints for querying LangGraph checkpoint data
from PostgreSQL for observability and run history.

All queries use session_id for linking with Context Engine artifacts.

CRASH-ONLY: Raises explicit errors if Postgres is unavailable.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# LangGraph imports
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool
except ImportError as e:
    raise ImportError(
        "╔═══════════════════════════════════════════════════════════════════╗\n"
        "║           LANGGRAPH POSTGRES CHECKPOINTER REQUIRED                    ║\n"
        "╚═══════════════════════════════════════════════════════════════════╝\n"
        "\n"
        f"Error: {e}\n"
        "\n"
        "Install with: uv add langgraph-checkpoint-postgres\n"
    ) from e

from ..logging_utils import safe_get_logger

logger = safe_get_logger(__name__, "checkpointer_api")
router = APIRouter(prefix="/checkpointer", tags=["checkpointer"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RunInfo(BaseModel):
    """Basic information about a research run."""
    run_id: str = Field(..., description="Unique run identifier (session_id)")
    thread_id: str = Field(..., description="LangGraph thread identifier")
    timestamp: str = Field(..., description="Run timestamp (ISO format)")
    use_case: Optional[str] = Field(None, description="Use case type")
    variant: Optional[str] = Field(None, description="Configuration variant")
    status: str = Field(default="unknown", description="Run status")
    checkpoint_count: int = Field(default=0, description="Number of checkpoints")


class CheckpointInfo(BaseModel):
    """Information about a checkpoint."""
    checkpoint_id: str = Field(..., description="Checkpoint identifier")
    checkpoint_ns: str = Field(default="", description="Checkpoint namespace")
    step: int = Field(default=0, description="Step number")
    timestamp: str = Field(..., description="Checkpoint timestamp (ISO format)")
    is_configured: bool = Field(default=False, description="Whether state is configured")


class RunDetails(BaseModel):
    """Detailed information about a run."""
    run_info: RunInfo
    checkpoints: List[CheckpointInfo]
    total_checkpoints: int


# =============================================================================
# CONNECTION POOL (SINGLETON)
# =============================================================================
# NOTE: The connection pool is intentionally global - it's designed for sharing
# across concurrent requests. Session isolation is handled by thread_id/session_id
# in the LangGraph config, not by having separate connections.
# The pool manages connection reuse efficiently.

_pool: Optional[AsyncConnectionPool] = None
_checkpointer: Optional[AsyncPostgresSaver] = None


async def _get_pool() -> AsyncConnectionPool:
    """Get or create the connection pool.

    The pool is shared across all requests for efficiency.
    Session isolation is handled by LangGraph's thread_id mechanism.

    Returns:
        AsyncConnectionPool instance

    Raises:
        HTTPException: If Postgres connection fails
    """
    global _pool, _checkpointer

    if _pool is not None:
        return _pool

    # Get Postgres connection string
    db_url = (
        os.getenv("POSTGRES_URL") or
        os.getenv("CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING") or
        "postgresql://ulli:ulli_password@localhost:5432/ulli_db"
    )

    if not db_url:
        raise HTTPException(
            status_code=500,
            detail="Postgres connection string not configured. Set POSTGRES_URL or CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING."
        )

    try:
        # Create connection pool
        _pool = AsyncConnectionPool(
            conninfo=db_url,
            min_size=1,
            max_size=10,
            open=False,
        )

        # Open the pool
        await _pool.open()

        # Create AsyncPostgresSaver
        _checkpointer = AsyncPostgresSaver(_pool)
        await _checkpointer.setup()

        logger.info("[CHECKPOINTER API] Connection pool initialized")
        return _pool

    except Exception as e:
        logger.error(f"[CHECKPOINTER API] Failed to initialize: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Postgres: {str(e)}"
        ) from e


async def _get_checkpointer() -> AsyncPostgresSaver:
    """Get or create the checkpointer.

    The checkpointer is shared across all requests.
    Session isolation is handled by LangGraph's thread_id mechanism.

    Returns:
        AsyncPostgresSaver instance
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    # Initialize pool (which also initializes checkpointer)
    await _get_pool()
    return _checkpointer  # type: ignore


def _extract_run_info_from_thread_id(thread_id: str) -> Dict[str, Any]:
    """Extract run information from thread_id.

    Thread ID format: deep_research_{use_case}_{variant}_{run_name}_{timestamp}

    Args:
        thread_id: LangGraph thread identifier

    Returns:
        Dictionary with extracted info
    """
    parts = thread_id.split("_")

    info = {
        "thread_id": thread_id,
        "use_case": None,
        "variant": None,
        "run_name": None,
        "timestamp": None,
    }

    # Parse thread_id if it follows our convention
    if len(parts) >= 5 and parts[0] == "deep":
        info["use_case"] = parts[2]
        info["variant"] = parts[3]
        info["run_name"] = "_".join(parts[4:-1])  # Everything except timestamp
        info["timestamp"] = parts[-1]

    return info


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/runs", response_model=List[RunInfo])
async def list_runs(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of runs to return"),
    use_case: Optional[str] = Query(None, description="Filter by use case"),
    variant: Optional[str] = Query(None, description="Filter by variant"),
) -> List[RunInfo]:
    """List all research runs from checkpointer database.

    Returns run information sorted by most recent first.
    Can be filtered by use_case and variant.
    """
    checkpointer = await _get_checkpointer()

    # Build filter for query
    config_filter: Dict[str, Any] = {}
    if use_case:
        config_filter["use_case"] = use_case
    if variant:
        config_filter["variant"] = variant

    try:
        # Query checkpoints from Postgres
        # We use an empty config to get all checkpoints
        runs_by_thread: Dict[str, RunInfo] = {}

        async for checkpoint_tuple in checkpointer.alist(
            config={"configurable": {}},
            filter=config_filter if config_filter else None,
            limit=limit * 2,  # Get more since we'll deduplicate by thread_id
        ):
            config, checkpoint, metadata = checkpoint_tuple
            thread_id = config.get("configurable", {}).get("thread_id", "")

            if not thread_id:
                continue

            # Extract run info from thread_id
            info = _extract_run_info_from_thread_id(thread_id)

            # Use thread_id as run_id (contains session_id)
            run_id = config.get("configurable", {}).get("session_id", thread_id)

            # Update checkpoint count
            if thread_id not in runs_by_thread:
                runs_by_thread[thread_id] = RunInfo(
                    run_id=run_id,
                    thread_id=thread_id,
                    timestamp=info.get("timestamp", datetime.now().isoformat()),
                    use_case=info.get("use_case"),
                    variant=info.get("variant"),
                    status="completed",
                    checkpoint_count=0,
                )

            runs_by_thread[thread_id].checkpoint_count += 1

        # Convert to list and sort by timestamp (most recent first)
        runs = list(runs_by_thread.values())
        runs.sort(key=lambda r: r.timestamp, reverse=True)

        return runs[:limit]

    except Exception as e:
        logger.error(f"[CHECKPOINTER API] Error listing runs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing runs: {str(e)}"
        ) from e


@router.get("/runs/{run_id}", response_model=RunDetails)
async def get_run(run_id: str) -> RunDetails:
    """Get detailed information about a specific run.

    Args:
        run_id: The run/session ID to retrieve

    Returns:
        Detailed run information including all checkpoints
    """
    checkpointer = await _get_checkpointer()

    try:
        # Query checkpoints for this run_id
        checkpoints_list: List[CheckpointInfo] = []

        # Query with session_id in metadata
        async for checkpoint_tuple in checkpointer.alist(
            config={"configurable": {"session_id": run_id}},
            limit=1000,  # Get all checkpoints for this run
        ):
            config, checkpoint, metadata = checkpoint_tuple

            # Extract checkpoint info
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id", "")
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            step = config.get("configurable", {}).get("step", 0)

            # Get timestamp from checkpoint or metadata
            checkpoint_data = checkpoint.get("channel_values", {})
            timestamp = checkpoint_data.get("timestamp", datetime.now().isoformat())

            checkpoints_list.append(CheckpointInfo(
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                step=step,
                timestamp=timestamp,
                is_configured=len(checkpoint_data) > 0,
            ))

        # Get run info from first checkpoint
        run_info: Optional[RunInfo] = None
        if checkpoints_list:
            # Extract from first checkpoint
            thread_id = f"deep_research"  # Default

            # Try to get more info from metadata
            # ...

            run_info = RunInfo(
                run_id=run_id,
                thread_id=thread_id,
                timestamp=datetime.now().isoformat(),
                status="completed",
                checkpoint_count=len(checkpoints_list),
            )

        return RunDetails(
            run_info=run_info or RunInfo(
                run_id=run_id,
                thread_id="",
                timestamp=datetime.now().isoformat(),
                status="unknown",
                checkpoint_count=len(checkpoints_list),
            ),
            checkpoints=checkpoints_list,
            total_checkpoints=len(checkpoints_list),
        )

    except Exception as e:
        logger.error(f"[CHECKPOINTER API] Error getting run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting run: {str(e)}"
        ) from e


@router.get("/sessions/{session_id}/state")
async def get_session_state(session_id: str) -> Dict[str, Any]:
    """Get the current state for a session.

    Args:
        session_id: The session ID to retrieve

    Returns:
        Current state dictionary
    """
    checkpointer = await _get_checkpointer()

    try:
        # Get the latest checkpoint for this session
        config = {"configurable": {"thread_id": session_id}}

        async for checkpoint_tuple in checkpointer.alist(config, limit=1):
            _, checkpoint, _ = checkpoint_tuple

            # Return the checkpoint data as state
            channel_values = checkpoint.get("channel_values", {})

            return {
                "session_id": session_id,
                "state": channel_values,
                "timestamp": channel_values.get("timestamp", datetime.now().isoformat()),
            }

        # No checkpoint found
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHECKPOINTER API] Error getting session state {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting session state: {str(e)}"
        ) from e


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if checkpointer database is accessible.

    Returns:
        Health status dictionary
    """
    try:
        checkpointer = await _get_checkpointer()

        # Try to list a single checkpoint to verify connection
        async for _ in checkpointer.alist(config={"configurable": {}}, limit=1):
            break

        return {
            "status": "healthy",
            "database": "Postgres",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"[CHECKPOINTER API] Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "Postgres",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


__all__ = ["router"]
