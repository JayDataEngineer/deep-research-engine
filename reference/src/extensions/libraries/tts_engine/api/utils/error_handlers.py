"""
TTS Error Handling Utilities

Provides consistent error handling for TTS endpoints.
"""

import logging
from collections.abc import Callable
from functools import wraps

from src.extensions.libraries.tts_engine.api.schemas.tts import TTSResponse
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Base exception for TTS-related errors."""

    def __init__(self, message: str, is_model_error: bool = False):
        self.message = message
        self.is_model_error = is_model_error
        super().__init__(message)


class ModelCorruptedError(TTSError):
    """Exception raised when a model file is corrupted."""

    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        super().__init__(message, is_model_error=True)


class ModelMissingError(TTSError):
    """Exception raised when a model file is missing or needs download."""

    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        super().__init__(message, is_model_error=True)


def extract_model_name_from_error(error_message: str) -> str | None:
    """Extract model name from error message."""
    error_str = error_message.lower()

    model_patterns = {
        "kokoro": "kokoro",
        "vibevoice-7b": "vibevoice-7b",
        "vibevoice": "vibevoice",
        "indextts": "indextts",
        "gptsovits": "gptsovits",
    }

    for pattern, model in model_patterns.items():
        if pattern in error_str:
            return model

    return None


def is_model_corruption_error(error_str: str) -> bool:
    """Check if error indicates model corruption."""
    corruption_keywords = [
        "corrupted",
        "checksum",
        "integrity",
        "invalid",
        "broken",
    ]
    return any(keyword in error_str for keyword in corruption_keywords)


def is_model_missing_error(error_str: str) -> bool:
    """Check if error indicates model is missing or needs download."""
    missing_keywords = [
        "model",
        "download",
        "checkpoint",
        "not found",
        "does not exist",
    ]
    return any(keyword in error_str for keyword in missing_keywords)


def create_error_response(
    error: Exception, context: str = ""
) -> TTSResponse:
    """Create a standardized TTSResponse for errors.

    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred

    Returns:
        TTSResponse with error information
    """
    error_str = str(error).lower()
    model_name = extract_model_name_from_error(str(error))

    if isinstance(error, TTSError):
        if isinstance(error, ModelCorruptedError):
            return TTSResponse(
                success=False,
                message="Model corruption detected",
                audio_url=None,
                duration=None
            )
        elif isinstance(error, ModelMissingError):
            return TTSResponse(
                success=False,
                message="Model required but not available",
                audio_url=None,
                duration=None
            )
        return TTSResponse(
            success=False,
            message=error.message,
            audio_url=None,
            duration=None
        )

    # Check for model-related errors
    if model_name:
        if is_model_corruption_error(error_str):
            logger.error(f"Model corruption detected: {model_name} - {error}")
            return TTSResponse(
                success=False,
                message="Model corruption detected",
                audio_url=None,
                duration=None
            )
        elif is_model_missing_error(error_str):
            logger.error(f"Model missing or download required: {model_name} - {error}")
            return TTSResponse(
                success=False,
                message="Model required but not available",
                audio_url=None,
                duration=None
            )

    # Generic error
    logger.error(f"TTS error{' at ' + context if context else ''}: {error}")
    return TTSResponse(
        success=False,
        message=str(error),
        audio_url=None,
        duration=None
    )


def handle_tts_errors(context: str = "TTS generation"):
    """Decorator for consistent TTS error handling in endpoints.

    Converts exceptions to appropriate HTTPExceptions or TTSResponse errors.

    Args:
        context: Context string for logging

    Example:
        @handle_tts_errors("speech generation")
        async def generate_speech(request: TTSRequest):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except TTSError as e:
                # Handle our custom TTS errors
                logger.error(f"TTSError in {context}: {e.message}")
                return create_error_response(e, context)
            except ValueError as e:
                # Validation errors
                logger.warning(f"Validation error in {context}: {str(e)}")
                return create_error_response(e, context)
            except Exception as e:
                # Unexpected errors
                logger.exception(f"Unexpected error in {context}")
                return create_error_response(e, context)

        return wrapper
    return decorator
