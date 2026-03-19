"""Media processing utilities."""

import base64
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Literal

from src.config import settings


class MediaType:
    """Media type constants."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


def get_mime_type(file_path: str) -> str:
    """Get MIME type for a file.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def get_media_type(file_path: str) -> Literal["image", "audio", "video", "unknown"]:
    """Determine media type from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Media type category
    """
    ext = Path(file_path).suffix.lower()

    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    audio_exts = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
    video_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

    if ext in image_exts:
        return MediaType.IMAGE
    elif ext in audio_exts:
        return MediaType.AUDIO
    elif ext in video_exts:
        return MediaType.VIDEO
    else:
        return "unknown"


def encode_file_to_base64(file_path: str, with_prefix: bool = True) -> str:
    """Encode a file to base64.

    Args:
        file_path: Path to the file
        with_prefix: Whether to include data URL prefix

    Returns:
        Base64 encoded string
    """
    mime_type = get_mime_type(file_path)

    with open(file_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")

    if with_prefix:
        return f"data:{mime_type};base64,{base64_data}"
    return base64_data


def save_uploaded_file(
    file_data: bytes,
    filename: str,
    media_type: Literal["image", "audio", "video"],
) -> tuple[str, Path]:
    """Save an uploaded file to disk.

    Args:
        file_data: Raw file bytes
        filename: Original filename
        media_type: Type of media (image, audio, video)

    Returns:
        Tuple of (relative URL, absolute path)
    """
    # Create unique filename
    ext = Path(filename).suffix
    unique_filename = f"{uuid.uuid4()}{ext}"

    # Create subdirectory for media type
    type_dir = settings.upload_dir / media_type
    type_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = type_dir / unique_filename
    with open(file_path, "wb") as f:
        f.write(file_data)

    # Return URL and path
    relative_url = f"/uploads/{media_type}/{unique_filename}"
    return relative_url, file_path


def delete_file(file_path: Path | str) -> bool:
    """Delete a file from disk.

    Args:
        file_path: Path to the file

    Returns:
        True if file was deleted, False otherwise
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
    except Exception:
        pass
    return False


def validate_file_size(file_data: bytes, max_size: int | None = None) -> bool:
    """Validate file size against maximum allowed.

    Args:
        file_data: Raw file bytes
        max_size: Maximum size in bytes (defaults to settings)

    Returns:
        True if file size is valid
    """
    max_size = max_size or settings.max_upload_size
    return len(file_data) <= max_size


def get_file_size_mb(file_data: bytes) -> float:
    """Get file size in megabytes.

    Args:
        file_data: Raw file bytes

    Returns:
        File size in MB
    """
    return len(file_data) / (1024 * 1024)


# Supported media formats
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".ogg", ".flac", ".aac"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".webm"}


def is_supported_format(filename: str, media_type: str) -> bool:
    """Check if a file format is supported.

    Args:
        filename: Name of the file
        media_type: Type of media (image, audio, video)

    Returns:
        True if format is supported
    """
    ext = Path(filename).suffix.lower()

    formats_map = {
        MediaType.IMAGE: SUPPORTED_IMAGE_FORMATS,
        MediaType.AUDIO: SUPPORTED_AUDIO_FORMATS,
        MediaType.VIDEO: SUPPORTED_VIDEO_FORMATS,
    }

    return ext in formats_map.get(media_type, set())


def save_base64_image(data_url: str) -> str:
    """Save a base64-encoded image to disk and return the URL.

    Args:
        data_url: Data URL (e.g., "data:image/png;base64,...")

    Returns:
        URL path to access the image (e.g., "/uploads/image/uuid.png")
    """
    # Parse the data URL
    if not data_url.startswith("data:image"):
        raise ValueError(f"Invalid image data URL: {data_url[:50]}...")

    # Extract the base64 data
    # Format: data:image/png;base64,{base64_data}
    try:
        header, base64_data = data_url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        ext = mimetypes.guess_extension(mime_type) or ".png"
    except ValueError:
        raise ValueError(f"Malformed data URL: {data_url[:50]}...")

    # Decode base64
    image_data = base64.b64decode(base64_data)

    # Save using existing function
    relative_url, _ = save_uploaded_file(
        image_data,
        f"image{ext}",  # Generic filename with extension
        MediaType.IMAGE,
    )

    return relative_url
