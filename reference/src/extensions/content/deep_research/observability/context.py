"""Thread-safe context management for observability.

.. deprecated:: 2.0.0
    Use LangFuse SDK context instead. LangFuse has built-in session
    management via session_id. See:
    src/extensions/integrations/langfuse/

    Migration:
    - ObservabilityContext -> LangFuse trace with session_id
    - observability_context() -> LangFuse context manager
    - get_context() -> Not needed, LangFuse handles context internally

This module provides a unified context management system using contextvars
for thread-safe, async-safe state management during research runs.

The ObservabilityContext is the single source of truth for all observability
state within a research session, including:
- RunLogger for logging
- GraphExecutionTracer for execution tracing
- DiagnosticLogger for issue detection
- DiagnosticConfig for diagnostic settings
- Health counters for monitoring

Usage:
    from .context import observability_context, get_context

    # Using context manager
    with observability_context(session_id="abc-123") as ctx:
        ctx.run_logger = RunLogger(...)
        ctx.tracer = GraphExecutionTracer()
        # ... research execution

    # Getting context from within nodes
    ctx = get_context_or_none()
    if ctx and ctx.run_logger:
        ctx.run_logger.log_query(query)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  ObservabilityContext                        │
    │  - session_id: Primary identifier                            │
    │  - run_logger: Optional[RunLogger]                           │
    │  - tracer: Optional[GraphExecutionTracer]                    │
    │  - diagnostic: Optional[DiagnosticLogger]                    │
    │  - diagnostic_config: Optional[DiagnosticConfig]             │
    │  - stream_emit_failures: int (health counter)                │
    │  - storage_fallback_count: int (health counter)              │
    └─────────────────────────────────────────────────────────────┘
"""

import warnings

warnings.warn(
    "ObservabilityContext is deprecated. Use LangFuse SDK context instead. "
    "See: src/extensions/integrations/langfuse/",
    DeprecationWarning,
    stacklevel=2,
)

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..logging_utils import safe_get_logger

if TYPE_CHECKING:
    from .run_logger import RunLogger
    from .graph_tracer import GraphExecutionTracer
    from .diagnostic_logging import DiagnosticLogger, DiagnosticConfig

logger = safe_get_logger(__name__, "context")


# =============================================================================
# CONTEXT DATA CLASS
# =============================================================================

@dataclass
class ObservabilityContext:
    """Container for all observability state for a single research run.

    This class holds all the observability-related state for a research
    session, making it easy to pass around and manage lifecycle.

    Attributes:
        session_id: Primary identifier for the research session
        run_logger: RunLogger instance for logging
        tracer: GraphExecutionTracer for execution tracing
        diagnostic: DiagnosticLogger for issue detection
        diagnostic_config: DiagnosticConfig for diagnostic settings
        stream_emit_failures: Count of streaming emit failures
        storage_fallback_count: Count of storage fallback operations
        metadata: Additional metadata for the context
    """
    session_id: str
    run_logger: Optional["RunLogger"] = None
    tracer: Optional["GraphExecutionTracer"] = None
    diagnostic: Optional["DiagnosticLogger"] = None
    diagnostic_config: Optional["DiagnosticConfig"] = None

    # Health monitoring counters
    stream_emit_failures: int = 0
    storage_fallback_count: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def increment_stream_failure(self) -> int:
        """Increment stream emit failure counter.

        Returns:
            New failure count
        """
        self.stream_emit_failures += 1
        return self.stream_emit_failures

    def increment_storage_fallback(self) -> int:
        """Increment storage fallback counter.

        Returns:
            New fallback count
        """
        self.storage_fallback_count += 1
        return self.storage_fallback_count

    def health_status(self) -> Dict[str, Any]:
        """Get health status of the observability context.

        Returns:
            Dict with health information
        """
        status = "healthy"

        if self.stream_emit_failures > 10:
            status = "degraded"
        elif self.storage_fallback_count > 10:
            status = "degraded"

        if self.stream_emit_failures > 50 or self.storage_fallback_count > 50:
            status = "unhealthy"

        return {
            "session_id": self.session_id,
            "status": status,
            "stream_emit_failures": self.stream_emit_failures,
            "storage_fallback_count": self.storage_fallback_count,
            "has_run_logger": self.run_logger is not None,
            "has_tracer": self.tracer is not None,
            "has_diagnostic": self.diagnostic is not None,
        }


# =============================================================================
# CONTEXT VARIABLE
# =============================================================================

# Context variable for thread-safe, async-safe access
_current_context: ContextVar[Optional[ObservabilityContext]] = ContextVar(
    'observability_context',
    default=None
)


# =============================================================================
# CONTEXT ACCESS FUNCTIONS
# =============================================================================

def get_context() -> ObservabilityContext:
    """Get current observability context.

    This should be used when you expect a context to be set (inside a
    research run). If no context is set, raises an error.

    Returns:
        The current ObservabilityContext

    Raises:
        RuntimeError: If no context is set (not inside a research run)

    Example:
        ctx = get_context()
        if ctx.run_logger:
            ctx.run_logger.log_query(query)
    """
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError(
            "No observability context set. "
            "Use observability_context() context manager before accessing context."
        )
    return ctx


def get_context_or_none() -> Optional[ObservabilityContext]:
    """Get current observability context, returning None if not set.

    This is safer than get_context() when you're not sure if a context
    is available (e.g., in utility functions that might be called outside
    a research run).

    Returns:
        The current ObservabilityContext or None if not set

    Example:
        ctx = get_context_or_none()
        if ctx and ctx.run_logger:
            ctx.run_logger.log_query(query)
    """
    return _current_context.get()


def set_context(ctx: ObservabilityContext) -> None:
    """Set the observability context directly.

    This is used internally by the context manager. External code should
    use observability_context() instead.

    Args:
        ctx: The context to set
    """
    _current_context.set(ctx)


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

@contextmanager
def observability_context(
    session_id: str,
    run_logger: Optional["RunLogger"] = None,
    tracer: Optional["GraphExecutionTracer"] = None,
    diagnostic: Optional["DiagnosticLogger"] = None,
    diagnostic_config: Optional["DiagnosticConfig"] = None,
    **metadata
):
    """Context manager for observability state.

    This sets up the observability context for a research run, making
    all observability components available via get_context().

    Args:
        session_id: Research session ID
        run_logger: Optional RunLogger instance
        tracer: Optional GraphExecutionTracer instance
        diagnostic: Optional DiagnosticLogger instance
        diagnostic_config: Optional DiagnosticConfig instance
        **metadata: Additional metadata to store in context

    Yields:
        ObservabilityContext instance

    Example:
        with observability_context(session_id) as ctx:
            ctx.run_logger = RunLogger(...)
            # ... research execution
    """
    ctx = ObservabilityContext(
        session_id=session_id,
        run_logger=run_logger,
        tracer=tracer,
        diagnostic=diagnostic,
        diagnostic_config=diagnostic_config,
        metadata=metadata,
    )

    token = _current_context.set(ctx)
    logger.debug(f"[CONTEXT] Set observability context for session: {session_id}")

    try:
        yield ctx
    finally:
        _current_context.reset(token)
        logger.debug(f"[CONTEXT] Reset observability context for session: {session_id}")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMPONENTS
# =============================================================================

def get_run_logger() -> Optional["RunLogger"]:
    """Get RunLogger from current context.

    Returns:
        RunLogger instance or None if not set
    """
    ctx = get_context_or_none()
    return ctx.run_logger if ctx else None


def get_tracer() -> Optional["GraphExecutionTracer"]:
    """Get or create GraphExecutionTracer from current context.

    If tracer doesn't exist in context, creates a new one.

    Returns:
        GraphExecutionTracer instance or None if no context
    """
    ctx = get_context_or_none()
    if ctx is None:
        return None

    if ctx.tracer is None:
        # Lazy import to avoid circular dependencies
        from .graph_tracer import GraphExecutionTracer
        ctx.tracer = GraphExecutionTracer()
        logger.debug(f"[CONTEXT] Created tracer for session: {ctx.session_id}")

    return ctx.tracer


def get_diagnostic_config() -> Optional["DiagnosticConfig"]:
    """Get DiagnosticConfig from current context.

    If config doesn't exist in context, returns a new default config.

    Returns:
        DiagnosticConfig instance or None if no context
    """
    ctx = get_context_or_none()
    if ctx is None:
        # Return default config if no context
        from .diagnostic_logging import DiagnosticConfig
        return DiagnosticConfig()

    if ctx.diagnostic_config is None:
        from .diagnostic_logging import DiagnosticConfig
        ctx.diagnostic_config = DiagnosticConfig()

    return ctx.diagnostic_config


def set_diagnostic_config(config: "DiagnosticConfig") -> None:
    """Set DiagnosticConfig in current context.

    Args:
        config: DiagnosticConfig instance to set
    """
    ctx = get_context_or_none()
    if ctx:
        ctx.diagnostic_config = config


def get_diagnostic_logger() -> Optional["DiagnosticLogger"]:
    """Get DiagnosticLogger from current context.

    Returns:
        DiagnosticLogger instance or None if not set
    """
    ctx = get_context_or_none()
    return ctx.diagnostic if ctx else None


def set_diagnostic_logger(diagnostic: "DiagnosticLogger") -> None:
    """Set DiagnosticLogger in current context.

    Args:
        diagnostic: DiagnosticLogger instance to set
    """
    ctx = get_context_or_none()
    if ctx:
        ctx.diagnostic = diagnostic


# =============================================================================
# HEALTH MONITORING
# =============================================================================

def increment_stream_failure() -> int:
    """Increment stream emit failure counter in current context.

    Returns:
        New failure count (0 if no context)
    """
    ctx = get_context_or_none()
    if ctx:
        return ctx.increment_stream_failure()
    return 0


def increment_storage_fallback() -> int:
    """Increment storage fallback counter in current context.

    Returns:
        New fallback count (0 if no context)
    """
    ctx = get_context_or_none()
    if ctx:
        return ctx.increment_storage_fallback()
    return 0


def get_health_status() -> Dict[str, Any]:
    """Get health status of current observability context.

    Returns:
        Health status dict, or empty dict if no context
    """
    ctx = get_context_or_none()
    if ctx:
        return ctx.health_status()
    return {"status": "no_context"}


# =============================================================================
# RE-EXPORTS
# =============================================================================

__all__ = [
    # Data class
    "ObservabilityContext",
    # Context access
    "get_context",
    "get_context_or_none",
    "set_context",
    # Context manager
    "observability_context",
    # Component convenience functions
    "get_run_logger",
    "get_tracer",
    "get_diagnostic_config",
    "set_diagnostic_config",
    "get_diagnostic_logger",
    "set_diagnostic_logger",
    # Health monitoring
    "increment_stream_failure",
    "increment_storage_fallback",
    "get_health_status",
]
