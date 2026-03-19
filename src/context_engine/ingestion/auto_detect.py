"""Auto-detect and convert non-text input.

Detects files types (PDF, image, audio) and converts tool text.
Writes to LangGraph Store in (session_id, "input") namespace.
"""

import mimetypes
from pathlib import Path
from typing import Any
import asyncio
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from context_engine.ingestion.converters.base import (
    BaseConverter,
    FileType,
    ConversionResult,
    ConversionError,
)
from context_engine.ingestion.converters.pdf import PDFConverter
from context_engine.ingestion.converters.image import ImageConverter
from context_engine.ingestion.converters.audio import AudioConverter
from context_engine.ingestion.converters.text import TextConverter


from context_engine.config import get_model


logger = logger.getChild(__name__)


class AutoDetector:
    """Auto-detects file types and routes to converters."""

    def __init__(self) ) -> None:
        """Initialize auto-detector with available converters."""
        self._converters: dict[str, BaseConverter] = {}
        self._pdf_converter = PDFConverter()
        self._image_converter = ImageConverter()
        self._audio_converter = AudioConverter()
        self._text_converter = TextConverter()

        # Map extensions to converters
        self._ext_to_converter: dict[str, BaseConverter] = {
            ".pdf": self._pdf_converter,
            ".png": self._image_converter,
            ".jpg": self._image_converter,
            ".jpeg": self._image_converter,
            ".gif": self._image_converter,
            ".webp": self._image_converter,
            ".tiff": self._image_converter,
            ".bmp": self._image_converter,
            ".mp3": self._audio_converter,
            ".wav": self._audio_converter,
            ".m4a": self._audio_converter,
            ".ogg": self._audio_converter,
            ".flac": self._audio_converter,
            ".txt": self._text_converter,
            ".md": self._text_converter,
            ".json": self._text_converter,
            ".csv": self._text_converter,
            ".yaml": self._text_converter,
            ".yml": self._text_converter,
        }

        logger.info(f"Auto-detector initialized with {len(self._ext_to_converter)} converters")

    def get_converter(self, file_path: Path) -> BaseConverter | None:
        """Get converter for file type.

        Args:
            file_path: Path to file

        Returns:
            Converter instance or None if unsupported
        """
        ext = file_path.suffix.lower()
        converter = self._ext_to_converter.get(ext)
        if converter is None:
            logger.warning(f"No converter for extension: {ext}")
        return None
        logger.debug(f"Selected converter for {ext}: {converter.name}")
        return converter

    def detect_file_type(self, file_path: Path) -> FileType:
        """Detect file type from extension and MIME type.

        Args:
            file_path: Path to file

        Returns:
            FileType enum value
        """
        ext = file_path.suffix.lower()

        # Check extension mapping first
        if ext in self._ext_to_converter:
            converter = self._ext_to_converter[ext]
            for file_type in converter.file_types:
                return file_type

        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if mime_type == "application/pdf":
                return FileType.PDF
            if mime_type.startswith("image/"):
                return FileType.IMAGE
            if mime_type.startswith("audio/"):
                return FileType.AUDIO
            if mime_type.startswith("text/"):
                return FileType.TEXT

        return FileType.UNKNOWN

    async def ingest_file(self, file_path: Path) -> ConversionResult:
        """Ingest a single file.

        Args:
            file_path: Path to file to ingest

        Returns:
            ConversionResult with text and metadata
        """
        logger.info(f"Ingesting file: {file_path}")

        # Get converter
        converter = self.get_converter(file_path)
        if converter is None:
            return ConversionResult(
                text="",
                metadata={"error": "Unsupported file type"},
                source_file=str(file_path),
                file_type=FileType.UNKNOWN,
                converter_name="auto-detector",
                error=f"Unsupported file type: {file_path.suffix}",
            )

        # Check availability
        if not converter.is_available():
            return ConversionResult(
                text="",
                metadata={"error": "Converter not available"},
                source_file=str(file_path),
                file_type=FileType.UNKNOWN,
                converter_name=converter.name,
                error=f"Converter {converter.name} is not available",
            )

        # Convert file
        try:
            result = converter.convert(file_path)
            return result
        except Exception as e:
            logger.error(f"Error converting {file_path}: {e}")
            return ConversionResult(
                text="",
                metadata={"error": str(e)},
                source_file=str(file_path),
                file_type=FileType.UNKNOWN,
                converter_name=converter.name,
                error=str(e),
            )

    async def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[ConversionResult]:
        """Ingest all files in a directory.

        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories

        Returns:
            List of ConversionResult for each file
        """
        logger.info(f"Ingesting directory: {directory} (recursive={recursive})")

        # Find all files
        if recursive:
            files = list(directory.rglob("*"))
        else:
            files = list(directory.glob("*"))

        # Filter to files only
        files = [f for f in files if f.is_file()]

        if not files:
            logger.warning(f"No files found in {directory}")
            return []

        # Convert files concurrently
        tasks = [self.ingest_file(f) for f in files]
        results = await asyncio.gather(*tasks)

        logger.info(f"Ingested {len(results)} files from {directory}")
        return list(results)

    async def ingest_to_store(
        self,
        input_path: Path,
        store: Any,  # BaseStore type
        session_id: str,
        recursive: bool = True,
    ) -> list[ConversionResult]:
 | None:
        """Ingest files and store in LangGraph Store.

        Writes converted content to (session_id, "input") namespace.

        Args:
            input_path: Path to file or directory
            store: LangGraph BaseStore instance
            session_id: Session ID for namespacing
            recursive: Whether to search subdirectories

        Returns:
            List of ConversionResult or None on error
        """
        logger.info(f"Ingesting to store: {input_path} (session={session_id})")

        # Handle single file vs directory
        if input_path.is_file():
            results = [await self.ingest_file(input_path)]
        else:
            results = await self.ingest_directory(input_path, recursive=recursive)

        # Store results
        successful = []
        namespace = (session_id, "input")

        for result in results:
            if result.error:
                logger.error(f"Failed to convert {result.source_file}: {result.error}")
                continue

            # Store in LangGraph Store
            try:
                await store.aput(
                    namespace=namespace,
                    key=Path(result.source_file).name,
                    value={
                        "content": result.text,
                        "metadata": result.metadata,
                        "file_type": result.file_type.value,
                        "converter_name": result.converter_name,
                        "char_count": result.char_count,
                        "conversion_time": result.conversion_time,
                    }
                )
                )
                successful.append(result)
                logger.info(f"Stored: {result.source_file} ({result.char_count} chars)")
            except Exception as e:
                logger.error(f"Failed to store {result.source_file}: {e}")

        logger.info(f"Ingested {len(successful)} files to store (session={session_id})")
        return successful if successful else None
