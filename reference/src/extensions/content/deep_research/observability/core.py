"""RunLogger factory functions for backward compatibility.

This module provides factory functions that were previously in the old
core.py implementation. The actual classes are now in modular files:
- RunLogger -> run_logger.py
- FileHandler -> file_handler.py
- MetricsTracker -> metrics_tracker.py
- RunMetrics, WorkerMetrics -> models.py

IMPORTANT: Session ID Integration
----------------------------------
When using the Context Engine, always pass session_id to ensure all
artifacts are properly linked:

    RunLogger(..., session_id=session_id)

The session_id links:
- LangGraph checkpoints (Postgres checkpointer)
- Context Engine artifacts (Neo4j graph + MinIO storage)
- Research sessions (Postgres metadata)
- File-based logs (local filesystem)

Usage:
    from deep_research.observability import RunLogger, get_run_logger

    # Direct instantiation
    logger = RunLogger("my_run", config, base_dir, session_id)

    # Or via factory function
    logger = get_run_logger("my_run", config, base_dir)

    with logger:
        logger.log_query("What is X?")
        logger.log_phase_call("worker", tokens=100, time=2.5)
        logger.set_response(response_text)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Import from new modular structure
from .run_logger import RunLogger
from .file_handler import FileHandler, FileHandlerFactory
from .metrics_tracker import MetricsTracker
from .models import RunMetrics, WorkerMetrics

from ..logging_utils import safe_get_logger

logger = safe_get_logger(__name__, "core")


# =============================================================================
# FACTORY FUNCTIONS (for backward compatibility)
# =============================================================================

def get_run_logger(
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    base_dir: Optional[str] = None,
    use_case: str = "rag",
    variant: str = "default",
    session_id: Optional[str] = None,
) -> RunLogger:
    """Create a new RunLogger instance.

    This is a convenience factory function for creating RunLogger instances.

    Args:
        run_name: Name for this run
        config: Configuration dictionary
        base_dir: Base directory for logs (defaults to LOGS_DIR/deep_research)
        use_case: Use case name (for metrics tracking)
        variant: Configuration variant
        session_id: Session ID for Context Engine linking (generated if not provided)

    Returns:
        Configured RunLogger instance

    Example:
        logger = get_run_logger("my_research", config, session_id="abc-123")
        with logger:
            logger.log_query("What is X?")
    """
    # Default to output/logs/deep_research in project root
    if base_dir is None:
        from src.constants import LOGS_DIR
        base_dir = LOGS_DIR / "deep_research"
    base_dir = Path(base_dir)

    # Create the RunLogger
    return RunLogger(
        run_name=run_name,
        config=config or {},
        base_dir=base_dir,
        session_id=session_id,
    )


@asynccontextmanager
async def with_run_logger(
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    base_dir: Optional[str] = None,
    use_case: str = "rag",
    variant: str = "default",
    session_id: Optional[str] = None,
):
    """Async context manager for RunLogger.

    This provides an async context manager that automatically saves
    the RunLogger on exit.

    Args:
        run_name: Name for this run
        config: Configuration dictionary
        base_dir: Base directory for logs
        use_case: Use case name
        variant: Configuration variant
        session_id: Session ID for Context Engine linking

    Yields:
        RunLogger instance

    Example:
        async with with_run_logger("my_research", config) as logger:
            logger.log_query("What is X?")
            # ... do research ...
    """
    run_logger = get_run_logger(
        run_name=run_name,
        config=config,
        base_dir=base_dir,
        use_case=use_case,
        variant=variant,
        session_id=session_id,
    )

    run_logger.open()
    try:
        yield run_logger
    finally:
        run_logger.close()


# =============================================================================
# RE-EXPORTS (for backward compatibility)
# =============================================================================

__all__ = [
    # Factory functions
    "get_run_logger",
    "with_run_logger",
    # Re-exported classes (for any direct imports from core.py)
    "RunLogger",
    "FileHandler",
    "FileHandlerFactory",
    "MetricsTracker",
    "RunMetrics",
    "WorkerMetrics",
]
