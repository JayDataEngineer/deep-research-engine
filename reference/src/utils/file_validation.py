"""File validation utilities for secure file uploads.

Provides validation functions for file size, type, and content scanning
to prevent malicious file uploads.
"""

import io
import magic
import logging
from typing import Optional, Tuple, List
from pathlib import Path

from src.api.core.errors import APIError, ErrorCode

logger = logging.getLogger(__name__)

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
) -> None:
    """Validate file size.

    Args:
        file_data: File content as bytes
        max_size: Maximum allowed size in bytes

    Raises:
        APIError: If file size exceeds limit
    """
    file_size = len(file_data)
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_mb:.2f}MB)",
            status_code=413,
        )


def validate_file_type(
    file_data: bytes,
    filename: Optional[str] = None,
    allowed_types: Optional[set[str]] = None,
) -> str:
    """Validate file type using magic bytes detection.

    Args:
        file_data: File content as bytes
        filename: Original filename (for extension check)
        allowed_types: Set of allowed MIME types

    Returns:
        Detected MIME type

    Raises:
        APIError: If file type is not allowed
    """
    # Detect MIME type from file content
    mime = magic.from_buffer(file_data, mime=True)

    # Validate extension matches content type
    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_FILE_EXTENSIONS:
            raise APIError(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"File extension '{ext}' is not allowed",
                status_code=415,
            )

    # Check against allowed types
    if allowed_types and mime not in allowed_types:
        raise APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"File type '{mime}' is not allowed",
            status_code=415,
        )

    return mime


def validate_image_upload(
    file_data: bytes,
    filename: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Validate an uploaded image file.

    Args:
        file_data: Image file content as bytes
        filename: Original filename

    Returns:
        Tuple of (file_data, mime_type)

    Raises:
        APIError: If validation fails
    """
    # Check file size
    validate_file_size(file_data, MAX_IMAGE_SIZE)

    # Validate file type
    mime_type = validate_file_type(file_data, filename, ALLOWED_IMAGE_TYPES)

    # Additional check: verify it's actually an image
    if not mime_type.startswith("image/"):
        raise APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Uploaded file is not an image (detected: {mime_type})",
            status_code=415,
        )

    logger.info(f"Image upload validated: {filename} ({len(file_data)} bytes, {mime_type})")
    return file_data, mime_type


def validate_audio_upload(
    file_data: bytes,
    filename: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Validate an uploaded audio file.

    Args:
        file_data: Audio file content as bytes
        filename: Original filename

    Returns:
        Tuple of (file_data, mime_type)

    Raises:
        APIError: If validation fails
    """
    # Check file size
    validate_file_size(file_data, MAX_AUDIO_SIZE)

    # Validate file type
    mime_type = validate_file_type(file_data, filename, ALLOWED_AUDIO_TYPES)

    # Additional check: verify it's actually audio
    if not mime_type.startswith("audio/"):
        raise APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Uploaded file is not audio (detected: {mime_type})",
            status_code=415,
        )

    logger.info(f"Audio upload validated: {filename} ({len(file_data)} bytes, {mime_type})")
    return file_data, mime_type


def validate_video_upload(
    file_data: bytes,
    filename: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Validate an uploaded video file.

    Args:
        file_data: Video file content as bytes
        filename: Original filename

    Returns:
        Tuple of (file_data, mime_type)

    Raises:
        APIError: If validation fails
    """
    # Check file size
    validate_file_size(file_data, MAX_VIDEO_SIZE)

    # Validate file type
    mime_type = validate_file_type(file_data, filename, ALLOWED_VIDEO_TYPES)

    # Additional check: verify it's actually video
    if not mime_type.startswith("video/"):
        raise APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Uploaded file is not video (detected: {mime_type})",
            status_code=415,
        )

    logger.info(f"Video upload validated: {filename} ({len(file_data)} bytes, {mime_type})")
    return file_data, mime_type


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace("/", "").replace("\\", "")

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = f"{name[:255-len(ext)]}{ext}"

    # Ensure filename isn't empty after sanitization
    if not filename:
        filename = "upload"

    return filename


async def validate_upload_file(
    file_data: bytes,
    filename: str,
    file_category: str = "auto",
) -> Tuple[bytes, str, str]:
    """Validate an uploaded file based on category.

    Args:
        file_data: File content as bytes
        filename: Original filename
        file_category: Category hint ('image', 'audio', 'video', or 'auto')

    Returns:
        Tuple of (file_data, sanitized_filename, mime_type)

    Raises:
        APIError: If validation fails
    """
    # Sanitize filename
    safe_filename = sanitize_filename(filename)

    # Auto-detect category if needed
    if file_category == "auto":
        mime = magic.from_buffer(file_data, mime=True)
        if mime.startswith("image/"):
            file_category = "image"
        elif mime.startswith("audio/"):
            file_category = "audio"
        elif mime.startswith("video/"):
            file_category = "video"
        else:
            raise APIError(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Unsupported file type: {mime}",
                status_code=415,
            )

    # Validate based on category
    if file_category == "image":
        validated_data, mime_type = validate_image_upload(file_data, safe_filename)
    elif file_category == "audio":
        validated_data, mime_type = validate_audio_upload(file_data, safe_filename)
    elif file_category == "video":
        validated_data, mime_type = validate_video_upload(file_data, safe_filename)
    else:
        raise APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Unknown file category: {file_category}",
            status_code=400,
        )

    return validated_data, safe_filename, mime_type
