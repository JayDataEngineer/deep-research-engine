"""Storage module for deep research - Unified Context Engine storage.

This module provides a SINGLE SOURCE OF TRUTH for all storage operations:

1. **operations.py** - Unified save/load operations
2. **namespace_factory.py** - Consistent namespace construction  
3. **deduplication.py** - Content hashing and duplicate detection

All storage operations MUST go through these modules to ensure:
- Consistent namespace construction
- Automatic deduplication
- Proper error handling
- Observability logging

Example:
    from deep_research.storage import (
        save_worker_finding,
        save_citation,
        findings_namespace,
        compute_content_hash,
    )
    
    # Save a finding
    await save_worker_finding(
        store=store,
        session_id="uuid-123",
        worker_id="philosopher-1",
        content="Research finding...",
    )
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.state import StateBackend
from deepagents.backends import StoreBackend

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from .state_backend import ResearchStateBackend, OutputBackend

# Import unified storage operations (SINGLE SOURCE OF TRUTH)
from .operations import (
    # Worker findings
    save_worker_finding,
    save_worker_findings_batch,
    load_worker_findings,
    # Citations
    save_citation,
    load_citations,
    # Artifacts
    save_artifact,
    save_research_brief,
    load_research_brief,
    save_synthesis,
    load_synthesis,
    save_critique,
    load_critique,
    save_boss_decision,
    load_boss_decision,
    save_architect_finding,
    load_architect_findings,
    # Session metadata
    save_session_metadata,
    load_session_metadata,
)

# Import namespace factory (SINGLE SOURCE OF TRUTH)
from .namespace_factory import (
    findings_namespace,
    citations_namespace,
    artifacts_namespace,
    research_brief_namespace,
    synthesis_namespace,
    critique_namespace,
    boss_decision_namespace,
    architect_findings_namespace,
    session_metadata_namespace,
    graph_namespace,
    logs_namespace,
    metrics_namespace,
    namespace_to_str,
    str_to_namespace,
    namespace_prefix,
)

# Import deduplication (SINGLE SOURCE OF TRUTH)
from .deduplication import (
    compute_content_hash,
    compute_dict_hash,
    is_duplicate,
    find_similar_content,
    check_duplicates,
    deduplicate_batch,
    normalize_content,
    normalize_for_comparison,
)


def create_research_filesystem(
    runtime: "ToolRuntime",
    config: dict[str, Any]
) -> CompositeBackend:
    """Create the composite filesystem for deep research.

    Args:
        runtime: LangGraph ToolRuntime (provides access to runtime.store)
        config: Configuration dict with keys:
            - session_id: Research session ID

    Returns:
        CompositeBackend with routed filesystem

    Example:
        >>> from langchain.tools import ToolRuntime
        >>> runtime = ToolRuntime(...)
        >>> config = {"session_id": "session_123"}
        >>> filesystem = create_research_filesystem(runtime, config)
        >>> # Now use filesystem.ls_info(), filesystem.read(), etc.
    """
    # Create backends
    # StoreBackend uses runtime.store to persist data with semantic search
    research_backend = StoreBackend(runtime)

    state_backend = ResearchStateBackend(runtime)
    output_backend = OutputBackend(runtime)

    # Create composite with path routing
    composite = CompositeBackend(
        default=state_backend,
        routes={
            "/research/": research_backend,
            "/output/": output_backend,
        }
    )

    return composite


def create_session_id() -> str:
    """Generate a new research session ID.

    Returns:
        Session ID string (timestamp-based)

    Example:
        >>> session_id = create_session_id()
        >>> session_id.startswith("session_")
        True
    """
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def get_default_config(session_id: Optional[str] = None) -> dict[str, Any]:
    """Get default configuration for research filesystem.

    Args:
        session_id: Optional session ID (will generate if None)

    Returns:
        Configuration dict
    """
    if session_id is None:
        session_id = create_session_id()

    return {
        "session_id": session_id,
    }


# =============================================================================
# CONTEXT MANAGEMENT FOR FILESYSTEM
# =============================================================================

import contextvars

_filesystem_context: contextvars.ContextVar = contextvars.ContextVar(
    'filesystem',
    default=None
)
"""Context variable for passing filesystem through the graph execution."""


def set_filesystem_context(filesystem: CompositeBackend):
    """Set filesystem in context variable.

    Args:
        filesystem: CompositeBackend instance

    Returns:
        Context token (for resetting later)

    Example:
        >>> token = set_filesystem_context(filesystem)
        >>> # ... use get_filesystem_from_context() ...
        >>> reset_filesystem_context(token)
    """
    return _filesystem_context.set(filesystem)


def get_filesystem_from_context() -> Optional[CompositeBackend]:
    """Get filesystem from context variable.

    This is the standard way for LangGraph V1 nodes to access the filesystem
    without receiving it as a function parameter.

    Returns:
        CompositeBackend instance or None if not set

    Example:
        >>> async def my_node(state: GlobalState) -> Dict[str, Any]:
        ...     filesystem = get_filesystem_from_context()
        ...     sources = filesystem.ls_info("/research/tier_a/")
        ...     return {"data": sources}
    """
    return _filesystem_context.get(None)


def reset_filesystem_context(token):
    """Reset filesystem context.

    Args:
        token: Context token from set_filesystem_context
    """
    _filesystem_context.reset(token)


# =============================================================================
# EXPORTS
# =============================================================================

# Import MinIO helpers for centralized access
from .minio_helpers import (
    get_minio_backend,
    get_web_session_backend,
    get_dumps_backend,
    get_artifacts_backend,
    get_outputs_backend,
    get_logs_backend,
    get_raw_minio_backend,
)

__all__ = [
    # Filesystem creation
    "create_research_filesystem",
    "create_session_id",
    "get_default_config",
    # Context management
    "set_filesystem_context",
    "get_filesystem_from_context",
    "reset_filesystem_context",
    # State backends
    "ResearchStateBackend",
    "OutputBackend",
    # Research state persistence
    "save_research_state",
    "delete_research_state",
    "cleanup_old_states",
    # MinIO helpers (NEW - use these instead of direct S3Backend)
    "get_minio_backend",
    "get_web_session_backend",
    "get_dumps_backend",
    "get_artifacts_backend",
    "get_outputs_backend",
    "get_logs_backend",
    "get_raw_minio_backend",
    # ========================================================================
    # UNIFIED STORAGE OPERATIONS (SINGLE SOURCE OF TRUTH)
    # ========================================================================
    # Worker findings
    "save_worker_finding",
    "save_worker_findings_batch",
    "load_worker_findings",
    # Citations
    "save_citation",
    "load_citations",
    # Artifacts
    "save_artifact",
    "save_research_brief",
    "load_research_brief",
    "save_synthesis",
    "load_synthesis",
    "save_critique",
    "load_critique",
    "save_boss_decision",
    "load_boss_decision",
    "save_architect_finding",
    "load_architect_findings",
    # Session metadata
    "save_session_metadata",
    "load_session_metadata",
    # Namespace factory
    "findings_namespace",
    "citations_namespace",
    "artifacts_namespace",
    "research_brief_namespace",
    "synthesis_namespace",
    "critique_namespace",
    "boss_decision_namespace",
    "architect_findings_namespace",
    "session_metadata_namespace",
    "graph_namespace",
    "logs_namespace",
    "metrics_namespace",
    "namespace_to_str",
    "str_to_namespace",
    "namespace_prefix",
    # Deduplication
    "compute_content_hash",
    "compute_dict_hash",
    "is_duplicate",
    "find_similar_content",
    "check_duplicates",
    "deduplicate_batch",
    "normalize_content",
    "normalize_for_comparison",
]


# =============================================================================
# RESEARCH STATE PERSISTENCE (Stubs for checkpointing/resume)
# =============================================================================

_research_states: dict[str, dict] = {}


def save_research_state(thread_id: str, state: dict) -> None:
    """Save research state for resume capability.

    Args:
        thread_id: Research thread ID
        state: Current research state dict
    """
    _research_states[thread_id] = state


def delete_research_state(thread_id: str) -> None:
    """Delete a saved research state.

    Args:
        thread_id: Research thread ID to delete
    """
    _research_states.pop(thread_id, None)


def cleanup_old_states(max_age_hours: int = 24) -> int:
    """Clean up old research states.

    Args:
        max_age_hours: Maximum age in hours to keep states

    Returns:
        Number of states cleaned up
    """
    # For in-memory storage, just clear all
    count = len(_research_states)
    _research_states.clear()
    return count
