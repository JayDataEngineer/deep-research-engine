"""Text converter for plain text files.
Handles .txt, .md, .json, .csv, .yaml, .yml.
"""

import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from loguru import logger

from context_engine.ingestion.converters.base import (
    BaseConverter,
    FileType,
    ConversionResult,
)


class TextConverter(BaseConverter):
    """Convert text files to text."""

    @property
    def file_types(self) -> list[FileType]:
        return [FileType.TEXT]

    @property
    def name(self) -> str:
        return "text-converter"

    def is_available(self) -> bool:
        """Text files are always processable."""
        return True

    def convert(self, file_path: Path) -> ConversionResult:
        """Read text file and return content.

        Args:
            file_path: Path to text file

        Returns:
            ConversionResult with file content
        """
        start_time = time.time()

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                return ConversionResult(
                    text="",
                    metadata={},
                    source_file=str(file_path),
                    file_type=FileType.TEXT,
                    converter_name=self.name,
                    error=f"Empty file: {file_path}",
                )

            elapsed = time.time() - start_time
            char_count = len(content)

            metadata = {
                "source_file": str(file_path),
                "file_format": file_path.suffix,
                "file_size_bytes": file_path.stat().st_size,
                "encoding": "utf-8",
                "line_count": content.count("\n"),
            }

            return ConversionResult(
                text=content,
                metadata=metadata,
                source_file=str(file_path),
                file_type=FileType.TEXT,
                converter_name=self.name,
                conversion_time=elapsed,
                char_count=char_count,
            )

        except Exception as e:
            logger.error(f"Error converting text file {file_path}: {e}")
            return ConversionResult(
                text="",
                metadata={},
                source_file=str(file_path),
                file_type=FileType.TEXT,
                converter_name=self.name,
                error=str(e),
            )
