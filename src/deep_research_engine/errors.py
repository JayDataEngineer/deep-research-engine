"""Centralized error types for Deep Research Engine.

This module defines all custom exceptions used across the package.
Errors are organized by subsystem for easy identification and handling.
"""

from __future__ import annotations

# =============================================================================
# BASE ERRORS
# =============================================================================


class DeepResearchError(Exception):
    """Base exception for all Deep Research Engine errors.

    All custom exceptions in this package inherit from this class.
    Use this for catching any deep-research-engine-specific error.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigError(DeepResearchError):
    """Raised when configuration loading or validation fails."""

    pass


class ConfigNotFoundError(ConfigError):
    """Raised when a requested configuration file does not exist."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    pass


# =============================================================================
# WORKER ERRORS
# =============================================================================


class WorkerError(DeepResearchError):
    """Base exception for worker-related errors."""

    pass


class WorkerSetupError(WorkerError):
    """Raised when worker setup fails."""

    pass


class WorkerExecutionError(WorkerError):
    """Raised when worker execution fails."""

    pass


class WorkerNotFoundError(WorkerError):
    """Raised when a requested worker type is not registered."""

    pass


class WorkerTimeoutError(WorkerError):
    """Raised when worker execution times out."""

    pass


# =============================================================================
# STORAGE ERRORS
# =============================================================================


class StorageError(DeepResearchError):
    """Base exception for storage-related errors."""

    pass


class StorageInitializationError(StorageError):
    """Raised when storage backend initialization fails."""

    pass


class StorageNotFoundError(StorageError):
    """Raised when requested data is not found in storage."""

    pass


class StorageConnectionError(StorageError):
    """Raised when storage connection fails."""

    pass


class StorageQueryError(StorageError):
    """Raised when storage query fails."""

    pass


# =============================================================================
# WORKFLOW ERRORS
# =============================================================================


class WorkflowError(DeepResearchError):
    """Base exception for workflow-related errors."""

    pass


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""

    pass


class WorkflowNotFoundError(WorkflowError):
    """Raised when a requested workflow is not found."""

    pass


class WorkflowValidationError(WorkflowError):
    """Raised when workflow input validation fails."""

    pass


# =============================================================================
# GRAPH ERRORS
# =============================================================================


class GraphError(DeepResearchError):
    """Base exception for graph-related errors."""

    pass


class GraphCompilationError(GraphError):
    """Raised when graph compilation fails."""

    pass


class GraphExecutionError(GraphError):
    """Raised when graph execution fails."""

    pass


class GraphStateError(GraphError):
    """Raised when graph state manipulation fails."""

    pass


# =============================================================================
# SYNTHESIS ERRORS
# =============================================================================


class SynthesisError(DeepResearchError):
    """Base exception for synthesis-related errors."""

    pass


class SynthesisGenerationError(SynthesisError):
    """Raised when synthesis generation fails."""

    pass


class SynthesisValidationError(SynthesisError):
    """Raised when synthesis output validation fails."""

    pass


# =============================================================================
# NODE ERRORS
# =============================================================================


class NodeError(DeepResearchError):
    """Base exception for node-related errors."""

    pass


class NodeExecutionError(NodeError):
    """Raised when node execution fails."""

    pass


class NodeNotFoundError(NodeError):
    """Raised when a requested node is not found."""

    pass
