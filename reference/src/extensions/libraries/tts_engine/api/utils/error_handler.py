"""
Centralized error handling utilities for API endpoints.

This module provides common error handling patterns used across TTS and chunked TTS endpoints.
"""

import logging
from typing import Any, Callable

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str | None = None
    ):
        """Initialize APIError.

        Args:
            message: User-facing error message
            status_code: HTTP status code to return
            detail: Additional detail for logging/debugging
        """
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class TTSGenerationError(APIError):
    """Exception for TTS generation errors."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class ValidationError(APIError):
    """Exception for validation errors."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )


class NotFoundError(APIError):
    """Exception for resource not found errors."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )


def handle_api_error(
    error: Exception,
    context: str = "API request",
    reraise_http: bool = True
) -> HTTPException:
    """Handle an exception and convert it to HTTPException if needed.

    This function provides centralized error handling for API endpoints:
    - HTTPException is re-raised as-is if reraise_http=True
    - APIError is converted to HTTPException with appropriate status code
    - Other exceptions are logged and converted to 500 error

    Args:
        error: The exception that occurred
        context: Description of what operation was happening
        reraise_http: If True, re-raise HTTPException as-is

    Returns:
        HTTPException to raise in the endpoint

    Raises:
        HTTPException: Always raises, either the original or a new one
    """
    # HTTPException should be re-raised as-is (FastAPI handles these)
    if reraise_http and isinstance(error, HTTPException):
        raise error

    # APIError subclasses have status codes built-in
    if isinstance(error, APIError):
        logger.error(
            f"API Error in {context}: {error.message}",
            exc_info=True,
            extra={"detail": error.detail, "status_code": error.status_code}
        )
        raise HTTPException(
            status_code=error.status_code,
            detail=error.message
        ) from error

    # Generic exceptions - log and return 500
    error_message = str(error)
    logger.error(
        f"Unexpected error in {context}: {error_message}",
        exc_info=True
    )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"An error occurred: {error_message}"
    ) from error


def async_error_handler(
    context: str = "API request",
    reraise_http: bool = True
) -> Callable:
    """Decorator for async endpoints that handles exceptions consistently.

    Usage:
        @async_error_handler(context="TTS generation")
        async def generate_speech(...):
            ...

    Args:
        context: Description of what operation is happening
        reraise_http: If True, re-raise HTTPException as-is

    Returns:
        Decorator function
    """
    import functools

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handle_api_error(e, context=context, reraise_http=reraise_http)

        return wrapper
    return decorator


def validate_text_not_empty(text: str, field_name: str = "text") -> None:
    """Validate that text field is not empty or whitespace only.

    Args:
        text: The text to validate
        field_name: Name of the field for error message

    Raises:
        ValidationError: If text is empty or whitespace only
    """
    if not text or not text.strip():
        raise ValidationError(
            message=f"{field_name} cannot be empty.",
            detail=f"Field '{field_name}' was empty or contained only whitespace."
        )


def validate_text_length(
    text: str,
    max_length: int,
    field_name: str = "text"
) -> None:
    """Validate that text does not exceed maximum length.

    Args:
        text: The text to validate
        max_length: Maximum allowed length
        field_name: Name of the field for error message

    Raises:
        ValidationError: If text exceeds max length
    """
    if len(text) > max_length:
        raise ValidationError(
            message=f"{field_name} too long. Maximum length is {max_length} characters.",
            detail=f"Field '{field_name}' had length {len(text)} but maximum is {max_length}."
        )


def log_generation_start(
    text: str,
    voice: str | None,
    extension: str,
    **params: Any
) -> None:
    """Log the start of a TTS generation request.

    Args:
        text: The text to synthesize
        voice: Voice ID
        extension: Extension name
        **params: Additional generation parameters
    """
    text_preview = text[:100] + "..." if len(text) > 100 else text
    logger.info(
        f"TTS generation request: extension={extension}, voice={voice}, "
        f"text_length={len(text)}, text_preview='{text_preview}', params={params}"
    )


def log_generation_success(
    text_length: int,
    duration: float,
    generation_time: float,
    chunks: int = 1
) -> None:
    """Log successful TTS generation.

    Args:
        text_length: Length of input text
        duration: Duration of generated audio
        generation_time: Time taken to generate
        chunks: Number of chunks (for chunked generation)
    """
    ratio = generation_time / duration if duration > 0 else 0
    logger.info(
        f"TTS generation complete: {text_length} chars, {duration:.2f}s audio, "
        f"{generation_time:.2f}s generation (ratio: {ratio:.2f}x), {chunks} chunk(s)"
    )
