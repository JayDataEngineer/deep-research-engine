"""Audio converter using LiteLLM Whisper API.

Converts audio files to text transcripts.
"""

import time
import os
from pathlib import Path
from typing import Any

from dataclasses import dataclass
from loguru import logger

from context_engine.config import get_model
from context_engine.ingestion.converters.base import (
    BaseConverter,
    FileType,
    ConversionResult,
)


class AudioConverter(BaseConverter):
    """Convert audio to text using Whisper model."""

    @property
    def file_types(self) -> list[FileType]:
        return [FileType.AUDIO]

    @property
    def name(self) -> str:
        return "audio-converter"

    def __init__(self) -> None:
        """Initialize audio converter."""
        # Check if Whisper model is configured
        whisper_model = os.getenv("WHISPER_MODEL", "whisper-1")
        self._whisper_available = bool(whisper_model)
        logger.info(f"Audio converter initialized (Whisper model: {whisper_model})")

 )

    def is_available(self) -> bool:
        """Check if audio file can be processed."""
        return self._whisper_available

 def convert(self, file_path: Path) -> ConversionResult:
        """Convert audio to text using Whisper model.

        Args:
            file_path: Path to audio file

        Returns:
            ConversionResult with extracted text
        """
        start_time = time.time()

        try:
            # Get file extension to determine format
            file_ext = file_path.suffix.lower()
            file_format = file_ext.lstrip(".")

            # Read audio file
            audio_data = file_path.read_bytes()
            if not audio_data:
                return ConversionResult(
                    text="",
                    metadata={},
                    source_file=str(file_path),
                    file_type=FileType.AUDIO,
                    converter_name=self.name,
                    error=f"Failed to read audio file: {file_path}",
                )

            # Use Whisper via LiteLLM
            model = get_model("whisper")

            if model is None:
                return ConversionResult(
                    text="",
                    metadata={},
                    source_file=str(file_path),
                    file_type=FileType.AUDIO,
                    converter_name=self.name,
                    error="Whisper model not available",
                )

            # Transcribe audio
            # Note: whisper-1 expects file input, not path
            response = model.transcribe(audio_data, file=file_path)

            elapsed = time.time() - start_time
            text = response.text
            char_count = len(text)
            metadata = {
                "source_file": str(file_path),
                "file_format": file_format,
                "file_size_bytes": len(audio_data),
                "conversion_method": "whisper",
                "duration_seconds": response.api_model.get("duration", 0),
            }
            return ConversionResult(
                text=text,
                metadata=metadata,
                source_file=str(file_path),
                file_type=FileType.AUDIO,
                converter_name=self.name,
                conversion_time=elapsed,
                char_count=char_count,
                tokens_used=0,  # Whisper doesn token counting
                model_name=model.model,
            )

        except Exception as e:
            logger.error(f"Error converting audio {file_path}: {e}")
            return ConversionResult(
                text="",
                metadata={},
                source_file=str(file_path),
                file_type=FileType.AUDIO,
                converter_name=self.name,
                error=str(e),
            )
