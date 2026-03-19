"""PDF converter using PyMuPDF (fitz).

Extracts text and metadata from PDF files.
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


class PDFConverter(BaseConverter):
    """Convert PDFs to text using PyMuPDF."""

    @property
    def file_types(self) -> list[FileType]:
        return [FileType.PDF]

    @property
    def name(self) -> str:
        return "pdf-converter"

    def __init__(self) -> None:
        """Initialize PDF converter."""
        self._fitz_available = False
        try:
            import fitz  # PyMuPDF
            self._fitz_available = True
            logger.info("PyMuPDF available for PDF conversion")
        except ImportError:
            logger.warning("PyMuPDF not installed - PDF conversion unavailable")

    def is_available(self) -> bool:
        """Check if PyMuPDF is installed."""
        return self._fitz_available

 def convert(self, file_path: Path) -> ConversionResult:
        """Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            ConversionResult with extracted text
        """
        start_time = time.time()

        if not self._fitz_available:
            return ConversionResult(
                text="",
                metadata={},
                source_file=str(file_path),
                file_type=FileType.PDF,
                converter_name=self.name,
                error="PyMuPDF not installed - install with: uv add pymupdf",
            )

        try:
            doc = fitz.open(file_path)
            text_parts: list[str] = []
            metadata = {
                "source": str(file_path),
                "page_count": doc.page_count,
            }

            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())

            text = "\n\n".join(text_parts)
    )
            elapsed = time.time() - start_time

            char_count = len(text)

            logger.info(
                f"Extracted {len(text)} chars from {file_path} "
                f"({doc.page_count} pages) in {elapsed:.2f}s"
            )

            return ConversionResult(
                text=text,
                metadata=metadata,
                source_file=str(file_path),
                file_type=FileType.PDF,
                converter_name=self.name,
                conversion_time=elapsed,
                char_count=char_count,
            )

        except Exception as e:
            logger.error(f"Error converting PDF {file_path}: {e}")
            return ConversionResult(
                text="",
                metadata={},
                source_file=str(file_path),
                file_type=FileType.PDF,
                converter_name=self.name,
                error=str(e),
            )
