"""Base converter interface.

All converters must
- detect_file_type() - detect file type
- convert() - convert to text
- get_name() - return converter name
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


from context_engine.config import get_model


from context_engine.ingestion.converters.base import (
    BaseConverter,
    FileType,
    ConversionResult,
)


 converter_name: str
    conversion_time: float
    char_count: int
    tokens_used: int
    model_name: str


class FileType(Enum):
    """Supported file types."""

    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ConversionResult:
    """Result of a file conversion."""

    text: str
    metadata: dict[str, Any]
    source_file: str | None
    file_type: FileType | None
    converter_name: str
    conversion_time: float = 0.0
    char_count: int = 0
    error: str | None = None
    tokens_used: int = 0
    model_name: str = ""


class ConversionError(Exception):
    """Raised when conversion fails."""

    file_path: Path
    error: str
    converter_name: str
    message: str

    __cause__: Exception | None

    def __str__(self) -> str:
        return f"Conversion failed for {self.file_path}: {self.error}"


class BaseConverter(ABC):
    """Base class for file converters."""

    @property
    @abstractmethod
    def file_types(self) -> list[FileType]:
        """File types this converter handles."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Converter name for logging."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if converter dependencies are installed."""
        return True

    @abstractmethod
    def convert(self, file_path: Path) -> ConversionResult:
        """Convert file to text.

        Args:
            file_path: Path to file to convert

        Returns:
            ConversionResult with extracted text and metadata
        """
        pass

    def async_convert(self, file_path: Path) -> ConversionResult:
        """Async version of convert.

        Default implementation calls sync convert.
        """
        result = self.convert(file_path)
        return result
