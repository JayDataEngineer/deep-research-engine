"""
Validation Utilities

Common validation patterns for TTS endpoints.
"""

import logging
import re

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class TextValidationError(ValidationError):
    """Exception raised when text input is invalid."""

    pass


class ParameterValidationError(ValidationError):
    """Exception raised when a parameter is invalid."""

    pass


def validate_text_length(text: str, max_length: int = 5000, min_length: int = 1) -> None:
    """Validate text length requirements.

    Args:
        text: Text to validate
        max_length: Maximum allowed length
        min_length: Minimum allowed length

    Raises:
        TextValidationError: If text doesn't meet requirements
    """
    if not text or not text.strip():
        raise TextValidationError("Text is required and cannot be empty")

    if len(text) > max_length:
        raise TextValidationError(
            f"Text too long. Maximum length is {max_length} characters, "
            f"got {len(text)} characters."
        )

    if len(text) < min_length:
        raise TextValidationError(
            f"Text too short. Minimum length is {min_length} characters, "
            f"got {len(text)} characters."
        )


def validate_voice_id(
    voice_id: str | None,
    extension_name: str,
    allow_empty: bool = False
) -> None:
    """Validate voice ID based on extension requirements.

    Args:
        voice_id: Voice ID to validate
        extension_name: Name of the extension
        allow_empty: Whether empty voice_id is acceptable

    Raises:
        ParameterValidationError: If voice_id is invalid for the extension
    """
    # Extensions that require voice selection
    voice_requiring_extensions = {
        "kokoro",
        "vibevoice",
    }

    # Extensions that use internal voice selection (no voice_id needed)

    # Voice cloning extensions (use ref_audio_path instead)

    if not voice_id or (isinstance(voice_id, str) and voice_id.strip() in ("", "None")):
        if not allow_empty and extension_name in voice_requiring_extensions:
            raise ParameterValidationError(
                f"Voice ID is required for {extension_name}"
            )
        # Empty voice_id is OK for extensions that don't require it
        return

    # Validate voice_id format
    if not isinstance(voice_id, str):
        raise ParameterValidationError(
            f"Voice ID must be a string, got {type(voice_id).__name__}"
        )

    if not voice_id.strip():
        raise ParameterValidationError("Voice ID cannot be empty or whitespace")


def validate_audio_format(format: str, allowed_formats: list[str] = None) -> None:
    """Validate audio format.

    Args:
        format: Audio format to validate
        allowed_formats: List of allowed formats (default: wav, mp3, ogg)

    Raises:
        ParameterValidationError: If format is invalid
    """
    if allowed_formats is None:
        allowed_formats = ["wav", "mp3", "ogg"]

    if format not in allowed_formats:
        raise ParameterValidationError(
            f"Invalid audio format '{format}'. Allowed formats: {', '.join(allowed_formats)}"
        )


def validate_range(
    value: float,
    param_name: str,
    min_value: float,
    max_value: float,
    allow_none: bool = False
) -> None:
    """Validate a numeric parameter is within range.

    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is an acceptable value

    Raises:
        ParameterValidationError: If value is out of range
    """
    if value is None:
        if not allow_none:
            raise ParameterValidationError(f"{param_name} cannot be None")
        return

    if not isinstance(value, (int, float)):
        raise ParameterValidationError(
            f"{param_name} must be a number, got {type(value).__name__}"
        )

    if value < min_value or value > max_value:
        raise ParameterValidationError(
            f"{param_name} must be between {min_value} and {max_value}, got {value}"
        )


def validate_file_upload(
    filename: str,
    content: bytes,
    max_size_mb: int = 50,
    allowed_extensions: list[str] = None
) -> None:
    """Validate uploaded file.

    Args:
        filename: Name of the uploaded file
        content: File content as bytes
        max_size_mb: Maximum file size in megabytes
        allowed_extensions: List of allowed file extensions (with dot)

    Raises:
        ParameterValidationError: If file is invalid
    """
    if allowed_extensions is None:
        allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    # Check file extension
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if file_ext not in allowed_extensions:
        raise ParameterValidationError(
            f"File type not supported: {file_ext}. "
            f"Supported types: {', '.join(allowed_extensions)}"
        )

    # Check file size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ParameterValidationError(
            f"File size too large. Maximum allowed size is {max_size_mb}MB, "
            f"got {size_mb:.2f}MB"
        )


def validate_tts_request(
    text: str,
    speed: float = 1.0,
    pitch: float = 1.0,
    format: str = "wav",
    max_text_length: int = 5000
) -> None:
    """Validate a complete TTS request.

    Args:
        text: Text to convert to speech
        speed: Speech speed multiplier
        pitch: Pitch adjustment
        format: Audio output format
        max_text_length: Maximum allowed text length

    Raises:
        ValidationError: If any parameter is invalid
    """
    validate_text_length(text, max_length=max_text_length)
    validate_range(speed, "speed", 0.25, 4.0)
    validate_range(pitch, "pitch", 0.5, 2.0)
    validate_audio_format(format)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove path components
    filename = filename.replace("\\", "/").split("/")[-1]

    # Replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)

    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255 - len(ext) + 1] + ext if ext else name[:255]

    return filename or "file"
