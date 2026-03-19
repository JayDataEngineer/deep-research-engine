"""
Standardized Error Handling for Deep Research System

Provides consistent exception classes and error handling patterns
throughout the deep research system.
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Type
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger

from .constants import format_error_log, Phase, ErrorMessages


logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


# ============================================================================
# Base Exception Classes
# ============================================================================

class DeepResearchError(Exception):
    """Base exception for all deep research errors"""

    def __init__(
        self,
        message: str,
        phase: Optional[Phase] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.phase = phase
        self.context = context or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "phase": self.phase.value if self.phase else None,
            "context": self.context,
            "original_error": type(self.original_error).__name__ if self.original_error else None,
            "original_message": str(self.original_error) if self.original_error else None,
        }

    def log(self, run_logger=None):
        """Log this error using the standard format"""
        error_log = format_error_log(
            self.phase or Phase.SYNTHESIS,
            self.original_error or self,
            ", ".join(f"{k}={v}" for k, v in self.context.items())
        )
        if run_logger:
            run_logger._write_to_file(error_log)
        logger.error(error_log)


class PhaseError(DeepResearchError):
    """Error during a specific research phase"""
    pass


class WorkerError(DeepResearchError):
    """Error during worker execution"""

    def __init__(
        self,
        message: str,
        worker_id: str,
        phase: Optional[Phase] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.worker_id = worker_id
        context = context or {}
        context["worker_id"] = worker_id
        super().__init__(message, phase, context, original_error)


class ToolCallError(DeepResearchError):
    """Error during tool execution"""

    def __init__(
        self,
        message: str,
        tool_name: str,
        phase: Optional[Phase] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.tool_name = tool_name
        context = context or {}
        context["tool_name"] = tool_name
        super().__init__(message, phase, context, original_error)


class ParseError(DeepResearchError):
    """Error during parsing of LLM responses"""
    pass


class QualityGateError(DeepResearchError):
    """Error when quality gate fails"""
    pass


class TimeoutError(DeepResearchError):
    """Error when operation times out"""
    pass


# ============================================================================
# Specific Error Classes
# ============================================================================

class BossError(PhaseError):
    """Error during boss phase"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, Phase.BOSS, context, original_error)


class LabError(PhaseError):
    """Error during lab phase"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, Phase.LAB, context, original_error)


class CriticError(PhaseError):
    """Error during critic phase"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, Phase.CRITIC, context, original_error)


class SynthesisError(PhaseError):
    """Error during synthesis phase"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, Phase.SYNTHESIS, context, original_error)


# ============================================================================
# Error Handler Class
# ============================================================================

class ErrorHandler:
    """Centralized error handling for deep research system"""

    def __init__(self, run_logger=None):
        self.run_logger = run_logger
        self.errors: list[Dict[str, Any]] = []

    def handle(
        self,
        error: Exception,
        phase: Optional[Phase] = None,
        context: Optional[Dict[str, Any]] = None,
        raise_error: bool = False
    ) -> Optional[DeepResearchError]:
        """
        Handle an error consistently.

        Args:
            error: The exception to handle
            phase: The phase where error occurred
            context: Additional context information
            raise_error: Whether to re-raise after handling

        Returns:
            DeepResearchError if not raising, None otherwise
        """
        # Convert to DeepResearchError if needed
        if not isinstance(error, DeepResearchError):
            error = DeepResearchError(
                message=str(error),
                phase=phase,
                context=context,
                original_error=error
            )

        # Log the error
        error.log(self.run_logger)

        # Track error
        self.errors.append(error.to_dict())

        # Raise if requested
        if raise_error:
            raise error

        return error

    def log_and_continue(
        self,
        error: Exception,
        phase: Optional[Phase] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error but don't raise - use for non-fatal errors"""
        self.handle(error, phase, context, raise_error=False)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        if not self.errors:
            return {"total_errors": 0}

        return {
            "total_errors": len(self.errors),
            "by_phase": self._count_errors_by_phase(),
            "by_type": self._count_errors_by_type(),
            "errors": self.errors,
        }

    def _count_errors_by_phase(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for error in self.errors:
            phase = error.get("phase", "unknown")
            counts[phase] = counts.get(phase, 0) + 1
        return counts

    def _count_errors_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for error in self.errors:
            error_type = error.get("error_type", "Unknown")
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts


# ============================================================================
# Decorators for Error Handling
# ============================================================================

def handle_errors(
    phase: Phase,
    context: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = True
):
    """
    Decorator for consistent error handling in functions.

    Usage:
        @handle_errors(Phase.WORKER, raise_on_error=False)
        def my_function(state, config):
            # Function code
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DeepResearchError:
                raise  # Already handled, re-raise
            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.handle(e, phase, context, raise_error=raise_on_error)
                if raise_on_error:
                    raise
                return None

        return wrapper
    return decorator


def safe_execute(
    default_return: Any = None,
    error_handler: Optional[callable] = None
):
    """
    Decorator that catches all exceptions and returns default.

    Usage:
        @safe_execute(default_return=[])
        def get_workers(config):
            # Function code that might fail
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in {func.__name__}: {e}")
                if error_handler:
                    error_handler(e, *args, **kwargs)
                return default_return

        return wrapper
    return decorator


# ============================================================================
# Context Managers
# ============================================================================

@contextmanager
def error_context(
    phase: Phase,
    error_handler: Optional[ErrorHandler] = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Context manager for error handling within a block.

    Usage:
        with error_context(Phase.WORKER, context={"attempt": 1}) as handler:
            # Code that might raise exceptions
    """
    handler = error_handler or ErrorHandler()
    try:
        yield handler
    except DeepResearchError:
        raise
    except Exception as e:
        handler.handle(e, phase, context, raise_error=False)


def suppress_errors(*error_types: Type[Exception]):
    """
    Context manager that suppresses specified error types.

    Usage:
        with suppress_errors(ValueError, KeyError):
            # Code that might raise ValueError or KeyError
    """

    @contextmanager
    def _suppress():
        try:
            yield
        except error_types as e:
            logger.debug(f"Suppressed error: {e}")

    return _suppress()


# ============================================================================
# Utility Functions
# ============================================================================

def create_error(
    error_class: Type[DeepResearchError],
    message: str,
    **kwargs
) -> DeepResearchError:
    """Factory function to create errors with consistent format"""
    return error_class(message, **kwargs)


def wrap_error(
    original_error: Exception,
    message: str,
    error_class: Type[DeepResearchError] = DeepResearchError,
    **kwargs
) -> DeepResearchError:
    """Wrap an original error in a DeepResearchError"""
    return error_class(
        message=message,
        original_error=original_error,
        **kwargs
    )
