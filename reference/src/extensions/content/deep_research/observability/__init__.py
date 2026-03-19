"""Observability module for deep research framework.

This module provides comprehensive logging, metrics tracking, and API endpoints:
- RunLogger: Main logging class with session-based tracking
- FileHandler: File I/O with context manager support
- MetricsTracker: Metrics collection and calculation
- DiagnosticLogger: Issue detection and validation
- API routes: Checkpointer and output serving

All components use session_id for linking with Context Engine artifacts.

Usage:
    from deep_research.observability import RunLogger

    with RunLogger("my_run", config, base_dir, session_id) as logger:
        logger.log_query("What is X?")
        logger.log_phase_call("worker", tokens=100, time=2.5)
        logger.set_response(response_text)

Session ID Integration:
    The session_id is the "golden thread" that links:
    - LangGraph checkpoints (Postgres)
    - Context Engine artifacts (Neo4j + MinIO)
    - Research session metadata (Postgres)
    - File-based logs (local filesystem)
"""

# =============================================================================
# MAIN CLASSES (Modular Structure)
# =============================================================================

from .run_logger import RunLogger
from .file_handler import (
    FileHandler,
    FileHandlerFactory,
    create_file_handler_async,
    LogFormat,
    StructuredLogWriter,
)
from .metrics_tracker import MetricsTracker
from .diagnostic_logging import (
    DiagnosticLogger,
    DiagnosticConfig,
    validate_llm_response_relevance,
    get_diagnostic_config,
    set_diagnostic_config,
    get_off_topic_indicators,
    get_domain_keywords,
)

# Data models
from .models import (
    RunMetrics,
    WorkerMetrics,
    extract_provider_from_base_url,
    extract_llm_metadata,
    save_worker_outputs_json,
    save_raw_sources_json,
)

# Storage
from .storage import (
    StorageManager,
    StorageConfig,
    StorageBackend,
    ContextEngineBackend,
    LocalFileBackend,
    create_storage_manager,
)

# Context management
from .context import (
    ObservabilityContext,
    observability_context,
    get_context,
    get_context_or_none,
    set_context,
    get_run_logger,
    get_tracer,
    get_diagnostic_config,
    set_diagnostic_config,
    get_diagnostic_logger,
    set_diagnostic_logger,
    increment_stream_failure,
    increment_storage_fallback,
    get_health_status,
)

# Errors
from .errors import (
    ObservabilityError,
    ContextEngineError,
    ContextEngineConnectionError,
    ContextEngineSaveError,
    ContextEngineLoadError,
    StorageError,
    StorageWriteError,
    StorageReadError,
    raise_context_engine_error,
    raise_storage_error,
)

# Callbacks
from .callbacks import (
    ObservabilityCallback,
    create_observability_callback,
    SpawnerCallbackHandler,
    create_callback_handler,
)

# API routers
from .checkpointer import router as checkpointer_router
# Note: minio_routes and output_md removed - functionality moved to LangFuse
from .routes import router as observability_router

# Graph execution tracer
from .graph_tracer import (
    GraphExecutionTracer,
    get_tracer,
    reset_tracer,
    wrap_node_with_tracing,
)

# Factory functions (backward compatibility)
from .core import (
    get_run_logger,
    with_run_logger,
)

# =============================================================================
# VERSION
# =============================================================================

__version__ = "2.0.0"  # Major version bump for simplified architecture

__all__ = [
    # Main classes
    "RunLogger",
    "FileHandler",
    "FileHandlerFactory",
    "LogFormat",
    "StructuredLogWriter",
    "MetricsTracker",
    "DiagnosticLogger",

    # Configuration
    "DiagnosticConfig",
    "get_diagnostic_config",
    "set_diagnostic_config",
    "get_off_topic_indicators",
    "get_domain_keywords",

    # Validation functions
    "validate_llm_response_relevance",

    # Data models
    "RunMetrics",
    "WorkerMetrics",
    "save_worker_outputs_json",
    "save_raw_sources_json",

    # Metadata extraction
    "extract_provider_from_base_url",
    "extract_llm_metadata",

    # Storage
    "StorageManager",
    "StorageConfig",
    "StorageBackend",
    "ContextEngineBackend",
    "LocalFileBackend",
    "create_storage_manager",

    # Context management
    "ObservabilityContext",
    "observability_context",
    "get_context",
    "get_context_or_none",
    "set_context",
    "get_run_logger",
    "get_tracer",
    "get_diagnostic_config",
    "set_diagnostic_config",
    "get_diagnostic_logger",
    "set_diagnostic_logger",
    "increment_stream_failure",
    "increment_storage_fallback",
    "get_health_status",

    # Errors
    "ObservabilityError",
    "ContextEngineError",
    "ContextEngineConnectionError",
    "ContextEngineSaveError",
    "ContextEngineLoadError",
    "StorageError",
    "StorageWriteError",
    "StorageReadError",
    "raise_context_engine_error",
    "raise_storage_error",

    # Callbacks
    "ObservabilityCallback",
    "create_observability_callback",
    "SpawnerCallbackHandler",
    "create_callback_handler",

    # Graph execution tracer
    "GraphExecutionTracer",
    "get_tracer",
    "reset_tracer",
    "wrap_node_with_tracing",

    # API routers
    "checkpointer_router",
    "observability_router",

    # Factory functions
    "get_run_logger",
    "with_run_logger",

    # Async helpers
    "create_file_handler_async",
]
