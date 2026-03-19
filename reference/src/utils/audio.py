"""Audio processing utilities."""

import base64
import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


def decode_base64_audio(base64_data: str) -> bytes:
    """Decode base64 audio data to bytes.

    Args:
        base64_data: Base64 encoded audio data (may include data URL prefix)

    Returns:
        Raw audio bytes
    """
    # Remove data URL prefix if present
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    return base64.b64decode(base64_data)


def encode_bytes_to_base64(audio_bytes: bytes) -> str:
    """Encode audio bytes to base64.

    Args:
        audio_bytes: Raw audio bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(audio_bytes).decode("utf-8")


def save_audio_to_file(
    audio_bytes: bytes, output_path: str, format: str = "wav"
) -> None:
    """Save audio bytes to a file.

    Args:
        audio_bytes: Raw audio bytes
        output_path: Path to save the file
        format: Audio format (wav, mp3, etc.)
    """
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def save_base64_audio_to_file(base64_data: str, output_path: str) -> None:
    """Save base64 audio data to a file.

    Args:
        base64_data: Base64 encoded audio data
        output_path: Path to save the file
    """
    audio_bytes = decode_base64_audio(base64_data)
    save_audio_to_file(audio_bytes, output_path)


def create_data_url(audio_bytes: bytes, format: str = "wav") -> str:
    """Create a data URL from audio bytes.

    Args:
        audio_bytes: Raw audio bytes
        format: Audio format

    Returns:
        Data URL string
    """
    base64_data = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/{format};base64,{base64_data}"


def parse_audio_chunks(chunks: list[str]) -> bytes:
    """Parse and concatenate base64 audio chunks.

    Args:
        chunks: List of base64 audio chunks

    Returns:
        Concatenated audio bytes
    """
    combined = "".join(chunks)
    return decode_base64_audio(combined)


class AudioBuffer:
    """Buffer for accumulating audio chunks."""

    def __init__(self):
        """Initialize an empty audio buffer."""
        self.chunks: list[str] = []

    def add_chunk(self, chunk: str) -> None:
        """Add a base64 audio chunk to the buffer.

        Args:
            chunk: Base64 encoded audio chunk
        """
        self.chunks.append(chunk)

    def get_bytes(self) -> bytes:
        """Get all audio data as bytes.

        Returns:
            Concatenated audio bytes
        """
        return parse_audio_chunks(self.chunks)

    def get_base64(self) -> str:
        """Get all audio data as base64 string.

        Returns:
            Concatenated base64 string
        """
        return "".join(self.chunks)

    def get_data_url(self, format: str = "wav") -> str:
        """Get all audio data as a data URL.

        Args:
            format: Audio format

        Returns:
            Data URL string
        """
        return create_data_url(self.get_bytes(), format)

    def save(self, path: str) -> None:
        """Save buffered audio to a file.

        Args:
            path: Output file path
        """
        save_audio_to_file(self.get_bytes(), path)

    def clear(self) -> None:
        """Clear the buffer."""
        self.chunks.clear()

    def __len__(self) -> int:
        """Get the number of chunks in the buffer."""
        return len(self.chunks)


# Audio format constants
AUDIO_FORMATS = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
}


def get_audio_mime_type(format: str) -> str:
    """Get MIME type for an audio format.

    Args:
        format: Audio format (wav, mp3, etc.)

    Returns:
        MIME type string
    """
    return AUDIO_FORMATS.get(format.lower(), "audio/wav")


def is_audio_format_supported(format: str) -> bool:
    """Check if an audio format is supported.

    Args:
        format: Audio format

    Returns:
        True if format is supported
    """
    return format.lower() in AUDIO_FORMATS
