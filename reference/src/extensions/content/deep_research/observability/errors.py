"""Standardized error formatting for observability.

This module provides consistent error formatting with ASCII box styling
for all observability errors. Using these classes ensures:
- Consistent error message format
- Proper error chaining (preserving stack traces)
- Clear identification of error types
- Easy error handling by callers

Usage:
    from .errors import raise_context_engine_error, StorageError

    # Using helper function
    try:
        save_output.invoke(...)
    except Exception as e:
        raise_context_engine_error("save_worker_outputs", e, f"Session: {session_id}")

    # Using exception class directly
    raise StorageError("save", "Failed to write metrics.json", backend="local")
"""

from typing import Any, Dict, Optional


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class ObservabilityError(Exception):
    """Base exception for all observability errors.

    All observability errors inherit from this class, making it easy
    to catch all observability-related errors with a single except clause.

    Attributes:
        operation: The operation that failed (e.g., "save", "load", "connect")
        message: Human-readable error message
        cause: The underlying exception that caused this error
        details: Additional context about the error
    """

    def __init__(
        self,
        operation: str,
        message: str,
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.message = message
        self.cause = cause
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message.

        Subclasses override this to customize formatting.
        """
        cause_str = f"\nCaused by: {type(self.cause).__name__}: {self.cause}" if self.cause else ""
        details_str = f"\nDetails: {self.details}" if self.details else ""
        return f"[{self.operation}] {self.message}{cause_str}{details_str}"


# =============================================================================
# CONTEXT ENGINE ERRORS
# =============================================================================

class ContextEngineError(ObservabilityError):
    """Context Engine operation failed.

    This error is raised when an operation involving the Context Engine
    (MinIO storage, Neo4j graph, etc.) fails. Context Engine is a critical
    dependency and has no fallbacks.

    The error message is formatted with an ASCII box for high visibility.
    """

    def _format_message(self) -> str:
        cause_str = f"\nError: {type(self.cause).__name__}: {self.cause}" if self.cause else ""
        details_lines = "\n".join(f"  {k}: {v}" for k, v in self.details.items())
        details_str = f"\n{details_lines}" if self.details else ""

        return (
            f"╔═══════════════════════════════════════════════════════════════════╗\n"
            f"║        CONTEXT ENGINE {self.operation.upper():^20} FAILED            ║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
            f"\n"
            f"Operation: {self.operation}\n"
            f"Message: {self.message}\n"
            f"{cause_str}\n"
            f"{details_str}\n"
            f"\n"
            f"Context Engine is REQUIRED for this operation.\n"
            f"No fallbacks available.\n"
        )


class ContextEngineConnectionError(ContextEngineError):
    """Failed to connect to Context Engine."""

    def __init__(self, component: str, cause: Optional[Exception] = None):
        super().__init__(
            operation="connect",
            message=f"Failed to connect to {component}",
            cause=cause,
            details={"component": component}
        )


class ContextEngineSaveError(ContextEngineError):
    """Failed to save data to Context Engine."""

    def __init__(
        self,
        output_type: str,
        cause: Optional[Exception] = None,
        session_id: Optional[str] = None
    ):
        details = {"output_type": output_type}
        if session_id:
            details["session_id"] = session_id
        super().__init__(
            operation="save",
            message=f"Failed to save {output_type} to Context Engine",
            cause=cause,
            details=details
        )


class ContextEngineLoadError(ContextEngineError):
    """Failed to load data from Context Engine."""

    def __init__(
        self,
        path: str,
        cause: Optional[Exception] = None,
        session_id: Optional[str] = None
    ):
        details = {"path": path}
        if session_id:
            details["session_id"] = session_id
        super().__init__(
            operation="load",
            message=f"Failed to load from {path}",
            cause=cause,
            details=details
        )


# =============================================================================
# STORAGE ERRORS
# =============================================================================

class StorageError(ObservabilityError):
    """Storage operation failed.

    This error is raised when a storage operation (save, load, delete)
    fails. The backend attribute indicates which storage backend failed.

    The error message is formatted with an ASCII box for visibility.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        backend: str = "unknown",
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.backend = backend
        details = details or {}
        details["backend"] = backend
        super().__init__(operation, message, cause, details)

    def _format_message(self) -> str:
        cause_str = f"\nError: {type(self.cause).__name__}: {self.cause}" if self.cause else ""
        details_lines = "\n".join(f"  {k}: {v}" for k, v in self.details.items() if k != "backend")
        details_str = f"\n{details_lines}" if details_lines else ""

        return (
            f"╔═══════════════════════════════════════════════════════════════════╗\n"
            f"║              STORAGE {self.operation.upper():^20} FAILED               ║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
            f"\n"
            f"Backend: {self.backend}\n"
            f"Operation: {self.operation}\n"
            f"Message: {self.message}\n"
            f"{cause_str}\n"
            f"{details_str}\n"
        )


class StorageWriteError(StorageError):
    """Failed to write to storage."""

    def __init__(
        self,
        path: str,
        backend: str = "unknown",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            operation="write",
            message=f"Failed to write to {path}",
            backend=backend,
            cause=cause,
            details={"path": path}
        )


class StorageReadError(StorageError):
    """Failed to read from storage."""

    def __init__(
        self,
        path: str,
        backend: str = "unknown",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            operation="read",
            message=f"Failed to read from {path}",
            backend=backend,
            cause=cause,
            details={"path": path}
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def raise_context_engine_error(
    operation: str,
    error: Exception,
    message: str = "",
    **details
) -> None:
    """Raise a formatted Context Engine error.

    This is a convenience function that creates and raises a ContextEngineError
    with proper error chaining.

    Args:
        operation: The operation that failed (e.g., "save_worker_outputs")
        error: The underlying exception
        message: Optional additional message
        **details: Additional context as keyword arguments

    Raises:
        ContextEngineError: Always raises this exception

    Example:
        try:
            save_output.invoke(...)
        except Exception as e:
            raise_context_engine_error("save_worker_outputs", e, session_id="abc-123")
    """
    raise ContextEngineError(
        operation=operation,
        message=message or "Operation failed",
        cause=error,
        details=details
    ) from error


def raise_storage_error(
    operation: str,
    error: Exception,
    backend: str = "unknown",
    message: str = "",
    **details
) -> None:
    """Raise a formatted Storage error.

    This is a convenience function that creates and raises a StorageError
    with proper error chaining.

    Args:
        operation: The operation that failed (e.g., "write", "read")
        error: The underlying exception
        backend: The storage backend that failed
        message: Optional additional message
        **details: Additional context as keyword arguments

    Raises:
        StorageError: Always raises this exception

    Example:
        try:
            storage.save(path, content)
        except Exception as e:
            raise_storage_error("write", e, backend="context_engine", path=path)
    """
    raise StorageError(
        operation=operation,
        message=message or "Storage operation failed",
        backend=backend,
        cause=error,
        details=details
    ) from error


# =============================================================================
# RE-EXPORTS
# =============================================================================

__all__ = [
    # Base exception
    "ObservabilityError",
    # Context Engine errors
    "ContextEngineError",
    "ContextEngineConnectionError",
    "ContextEngineSaveError",
    "ContextEngineLoadError",
    # Storage errors
    "StorageError",
    "StorageWriteError",
    "StorageReadError",
    # Helper functions
    "raise_context_engine_error",
    "raise_storage_error",
]
