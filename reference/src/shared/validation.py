"""Centralized validation utilities to eliminate DRY violations.

This module provides standardized validation functions for common patterns:
- Text/string validation
- Numeric range validation
- File upload validation
- ID validation
- List/collection validation
- Filename sanitization

All validation functions raise appropriate exceptions for API responses.
"""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional, Set, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class TextValidationError(ValidationError):
    """Exception raised when text input is invalid."""

    pass


class ParameterValidationError(ValidationError):
    """Exception raised when a parameter is invalid."""

    pass


class FileValidationError(ValidationError):
    """Exception raised when file upload validation fails."""

    pass


# =============================================================================
# Text/String Validation
# =============================================================================

def validate_not_empty(
    value: Any,
    field_name: str = "field",
    allow_whitespace: bool = False
) -> str:
    """Validate that a value is a non-empty string.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        allow_whitespace: Whether to allow whitespace-only strings

    Returns:
        The validated string

    Raises:
        HTTPException: If validation fails
    """
    if value is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot be None"
        )

    if not isinstance(value, str):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a string, got {type(value).__name__}"
        )

    if allow_whitespace:
        if value == "":
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} cannot be empty"
            )
    else:
        if not value.strip():
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} cannot be empty or whitespace-only"
            )

    return value


def validate_text_length(
    text: str,
    max_length: int = 5000,
    min_length: int = 1,
    field_name: str = "text"
) -> str:
    """Validate text length requirements.

    Args:
        text: Text to validate
        max_length: Maximum allowed length
        min_length: Minimum allowed length
        field_name: Field name for error messages

    Returns:
        The validated text

    Raises:
        HTTPException: If text doesn't meet requirements
    """
    validate_not_empty(text, field_name)

    text_len = len(text)
    if text_len > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} too long. Maximum length is {max_length} characters, got {text_len} characters."
        )

    if text_len < min_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} too short. Minimum length is {min_length} characters, got {text_len} characters."
        )

    return text


def validate_url(
    url: str,
    field_name: str = "url",
    require_http: bool = True
) -> str:
    """Validate that a string is a valid URL.

    Args:
        url: URL to validate
        field_name: Field name for error messages
        require_http: Whether to require http:// or https:// scheme

    Returns:
        The validated URL

    Raises:
        HTTPException: If URL is invalid
    """
    validate_not_empty(url, field_name, allow_whitespace=False)

    if require_http and not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must start with http:// or https://"
        )

    return url


# =============================================================================
# Numeric Validation
# =============================================================================

def validate_range(
    value: Optional[float],
    param_name: str,
    min_value: float,
    max_value: float,
    allow_none: bool = False
) -> float:
    """Validate a numeric parameter is within range.

    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is an acceptable value

    Returns:
        The validated value

    Raises:
        HTTPException: If value is out of range
    """
    if value is None:
        if not allow_none:
            raise HTTPException(
                status_code=400,
                detail=f"{param_name} cannot be None"
            )
        return value

    if not isinstance(value, (int, float)):
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be a number, got {type(value).__name__}"
        )

    if value < min_value or value > max_value:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be between {min_value} and {max_value}, got {value}"
        )

    return value


def validate_positive_number(
    value: Optional[float],
    param_name: str,
    allow_zero: bool = False,
    allow_none: bool = False
) -> float:
    """Validate that a number is positive.

    Args:
        value: Value to validate
        param_name: Parameter name for error messages
        allow_zero: Whether zero is allowed
        allow_none: Whether None is acceptable

    Returns:
        The validated value

    Raises:
        HTTPException: If value is not positive
    """
    if value is None:
        if not allow_none:
            raise HTTPException(
                status_code=400,
                detail=f"{param_name} cannot be None"
            )
        return value

    if not isinstance(value, (int, float)):
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be a number, got {type(value).__name__}"
        )

    min_value = 0 if allow_zero else 0.001
    if value < min_value:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be positive, got {value}"
        )

    return value


# =============================================================================
# List/Collection Validation
# =============================================================================

def validate_not_empty_list(
    items: list,
    field_name: str = "items",
    max_items: Optional[int] = None
) -> list:
    """Validate that a list is not empty.

    Args:
        items: List to validate
        field_name: Field name for error messages
        max_items: Optional maximum number of items

    Returns:
        The validated list

    Raises:
        HTTPException: If list is empty or too long
    """
    if not isinstance(items, list):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a list, got {type(items).__name__}"
        )

    if len(items) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot be empty"
        )

    if max_items is not None and len(items) > max_items:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot contain more than {max_items} items, got {len(items)}"
        )

    return items


def validate_list_items(
    items: list,
    item_type: type,
    field_name: str = "items"
) -> list:
    """Validate that all items in a list are of the expected type.

    Args:
        items: List to validate
        item_type: Expected type for all items
        field_name: Field name for error messages

    Returns:
        The validated list

    Raises:
        HTTPException: If any item is not the expected type
    """
    validate_not_empty_list(items, field_name)

    for i, item in enumerate(items):
        if not isinstance(item, item_type):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}[{i}] must be {item_type.__name__}, got {type(item).__name__}"
            )

    return items


# =============================================================================
# ID Validation
# =============================================================================

def validate_id_exists(
    id_value: str,
    check_func: Callable[[str], bool],
    item_type: str = "Item"
) -> str:
    """Validate that an ID exists, raise 404 if not.

    Args:
        id_value: ID to check
        check_func: Function that returns True if ID exists
        item_type: Type name for error message

    Returns:
        The ID if valid

    Raises:
        HTTPException: 404 if ID doesn't exist
    """
    validate_not_empty(id_value, "id")

    if not check_func(id_value):
        raise HTTPException(
            status_code=404,
            detail=f"{item_type} '{id_value}' not found"
        )

    return id_value


def validate_id_not_exists(
    id_value: str,
    check_func: Callable[[str], bool],
    item_type: str = "Item"
) -> str:
    """Validate that an ID doesn't exist, raise 409 if it does.

    Args:
        id_value: ID to check
        check_func: Function that returns True if ID exists
        item_type: Type name for error message

    Returns:
        The ID if valid

    Raises:
        HTTPException: 409 if ID already exists
    """
    validate_not_empty(id_value, "id")

    if check_func(id_value):
        raise HTTPException(
            status_code=409,
            detail=f"{item_type} '{id_value}' already exists"
        )

    return id_value


# =============================================================================
# File Validation
# =============================================================================

# Maximum file sizes (in bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB default

# Allowed MIME types by category
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
}

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
    "audio/m4a",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
}

ALLOWED_FILE_EXTENSIONS = {
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a",
    # Video
    ".mp4", ".webm", ".mov",
}


def validate_file_size(
    file_data: bytes,
    max_size: int = MAX_FILE_SIZE,
    field_name: str = "file"
) -> None:
    """Validate file size.

    Args:
        file_data: File content as bytes
        max_size: Maximum allowed size in bytes
        field_name: Field name for error messages

    Raises:
        HTTPException: If file size exceeds limit
    """
    file_size = len(file_data)
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_mb:.2f}MB)"
        )


def validate_file_extension(
    filename: str,
    allowed_extensions: Optional[Set[str]] = None,
    field_name: str = "file"
) -> str:
    """Validate file extension.

    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed extensions (with dot)
        field_name: Field name for error messages

    Returns:
        The file extension (lowercase, with dot)

    Raises:
        HTTPException: If extension is not allowed
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_FILE_EXTENSIONS

    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"{field_name} extension '{ext}' is not allowed. Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    return ext


def validate_file_upload(
    filename: str,
    content: bytes,
    max_size_mb: int = 50,
    allowed_extensions: Optional[Set[str]] = None,
    field_name: str = "file"
) -> Tuple[str, bytes]:
    """Validate uploaded file.

    Args:
        filename: Name of the uploaded file
        content: File content as bytes
        max_size_mb: Maximum file size in megabytes
        allowed_extensions: List of allowed file extensions (with dot)
        field_name: Field name for error messages

    Returns:
        Tuple of (sanitized_filename, content)

    Raises:
        HTTPException: If file is invalid
    """
    # Sanitize filename first
    safe_filename = sanitize_filename(filename)

    # Validate extension
    validate_file_extension(safe_filename, allowed_extensions, field_name)

    # Validate file size
    max_size_bytes = max_size_mb * 1024 * 1024
    validate_file_size(content, max_size_bytes, field_name)

    return safe_filename, content


# =============================================================================
# Filename Sanitization
# =============================================================================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    This function:
    - Removes path separators
    - Removes null bytes
    - Replaces dangerous characters
    - Limits length to 255 characters

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

    # Limit length (preserve extension)
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        if ext:
            max_name_len = 255 - len(ext) - 1
            filename = name[:max_name_len] + '.' + ext
        else:
            filename = name[:255]

    # Ensure filename isn't empty after sanitization
    if not filename:
        filename = "upload"

    return filename


# =============================================================================
# Audio Format Validation
# =============================================================================

ALLOWED_AUDIO_FORMATS = {"wav", "mp3", "ogg"}


def validate_audio_format(
    format: str,
    allowed_formats: Optional[Set[str]] = None
) -> str:
    """Validate audio format.

    Args:
        format: Audio format to validate
        allowed_formats: Set of allowed formats (default: wav, mp3, ogg)

    Returns:
        The validated format

    Raises:
        HTTPException: If format is invalid
    """
    if allowed_formats is None:
        allowed_formats = ALLOWED_AUDIO_FORMATS

    if format not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format '{format}'. Allowed formats: {', '.join(sorted(allowed_formats))}"
        )

    return format


# =============================================================================
# Voice ID Validation (TTS-specific)
# =============================================================================

VOICE_REQUIRING_EXTENSIONS = {
    "kokoro",
    "vibevoice",
}


def validate_voice_id(
    voice_id: Optional[str],
    extension_name: str,
    allow_empty: bool = False
) -> Optional[str]:
    """Validate voice ID based on extension requirements.

    Args:
        voice_id: Voice ID to validate
        extension_name: Name of the extension
        allow_empty: Whether empty voice_id is acceptable

    Returns:
        The validated voice_id

    Raises:
        HTTPException: If voice_id is invalid for the extension
    """
    # Check if voice_id is empty
    if not voice_id or (isinstance(voice_id, str) and voice_id.strip() in ("", "None")):
        if not allow_empty and extension_name in VOICE_REQUIRING_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Voice ID is required for {extension_name}"
            )
        return None

    # Validate voice_id format
    if not isinstance(voice_id, str):
        raise HTTPException(
            status_code=400,
            detail=f"Voice ID must be a string, got {type(voice_id).__name__}"
        )

    if not voice_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Voice ID cannot be empty or whitespace"
        )

    return voice_id


# =============================================================================
# Compound Validators (for complete request validation)
# =============================================================================

def validate_tts_request(
    text: str,
    speed: float = 1.0,
    pitch: float = 1.0,
    format: str = "wav",
    max_text_length: int = 5000
) -> Tuple[str, float, float, str]:
    """Validate a complete TTS request.

    Args:
        text: Text to convert to speech
        speed: Speech speed multiplier
        pitch: Pitch adjustment
        format: Audio output format
        max_text_length: Maximum allowed text length

    Returns:
        Tuple of (validated_text, validated_speed, validated_pitch, validated_format)

    Raises:
        HTTPException: If any parameter is invalid
    """
    validated_text = validate_text_length(text, max_length=max_text_length)
    validated_speed = validate_range(speed, "speed", 0.25, 4.0)
    validated_pitch = validate_range(pitch, "pitch", 0.5, 2.0)
    validated_format = validate_audio_format(format)

    return validated_text, validated_speed, validated_pitch, validated_format
