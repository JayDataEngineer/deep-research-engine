"""Kreuzberg extractor wrapper - document extraction via Kreuzberg.

Wraps the Kreuzberg library to provide a unified interface for extracting
text, tables, metadata, and chunks from 91+ document formats.

Output format is designed to integrate with our content_classifier pipeline:
- Structured chunks with metadata
- Tables extracted separately
- File metadata preserved for RAG indexing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from kreuzberg import (
        ChunkingConfig,
        ExtractionConfig,
        ExtractionResult,
        extract_file,
    )
    from kreuzberg import batch_extract_files_sync as kreuzberg_batch_sync
    HAS_KREUZBERG = True
except ImportError:
    HAS_KREUZBERG = False

from ..utils.logging import get_loguru_logger

_internal_logger = get_loguru_logger("ingestion.kreuzberg")


@dataclass
class KreuzbergExtractionResult:
    """Structured result from Kreuzberg extraction.

    Designed to feed directly into content_classifier.
    """

    success: bool
    file_path: str
    """Original file path."""

    content: str = ""
    """Full extracted text content."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    """Text chunks (if chunking was enabled). Each chunk has:
    - text: chunk text
    - index: chunk number
    - heading_context: heading breadcrumb (if markdown chunker)
    - metadata: source file, page info, etc.
    """

    tables: list[dict[str, Any]] = field(default_factory=list)
    """Extracted tables. Each table has:
    - markdown: table as markdown string
    - cells: 2D cell array
    - has_header: bool
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Document metadata (title, author, dates, page count, etc.)."""

    pages: list[dict[str, Any]] = field(default_factory=list)
    """Per-page content (if page extraction enabled)."""

    images: list[dict[str, Any]] = field(default_factory=list)
    """Extracted images from the document (if enabled)."""

    detected_languages: list[str] = field(default_factory=list)
    """Detected document languages."""

    keywords: list[str] = field(default_factory=list)
    """Extracted keywords (if enabled)."""

    error: str | None = None
    """Error message if extraction failed."""

    char_count: int = 0
    """Total character count of extracted content."""

    table_count: int = 0
    """Number of tables extracted."""

    chunk_count: int = 0
    """Number of chunks generated."""


@dataclass
class KreuzbergConfig:
    """Configuration for Kreuzberg extraction.

    Wraps Kreuzberg's ExtractionConfig with sensible defaults for our pipeline.
    """

    chunking_enabled: bool = True
    chunk_max_chars: int = 4000
    chunk_max_overlap: int = 200
    """Chunking parameters for RAG-friendly text splitting."""

    output_format: str = "markdown"
    """Output format: "plain", "markdown", "djot", "html"."""

    ocr_enabled: bool = True
    ocr_backend: str = "tesseract"
    ocr_language: str = "eng"
    """OCR configuration for scanned documents."""

    force_ocr: bool = False
    """Force OCR even on searchable PDFs."""

    table_extraction: bool = True
    """Enable table extraction."""

    image_extraction: bool = False
    """Extract embedded images from documents."""

    language_detection: bool = True
    """Detect document language(s)."""

    keyword_extraction: bool = False
    """Extract keywords from content."""

    def to_kreuzberg_config(self) -> ExtractionConfig:
        """Convert to Kreuzberg's ExtractionConfig."""
        if not HAS_KREUZBERG:
            raise ImportError(
                "Kreuzberg is not installed. Install with: pip install kreuzberg"
            )

        kwargs: dict[str, Any] = {
            "output_format": self.output_format,
            "use_cache": True,
            "enable_quality_processing": True,
        }

        if self.chunking_enabled:
            kwargs["chunking"] = ChunkingConfig(
                max_chars=self.chunk_max_chars,
                max_overlap=self.chunk_max_overlap,
            )

        if self.ocr_enabled:
            from kreuzberg import OcrConfig

            ocr_kwargs: dict[str, Any] = {
                "backend": self.ocr_backend,
                "language": self.ocr_language,
            }
            if self.table_extraction:
                from kreuzberg import TesseractConfig

                ocr_kwargs["tesseract_config"] = TesseractConfig(
                    enable_table_detection=True,
                    table_min_confidence=0.5,
                )

            kwargs["ocr"] = OcrConfig(**ocr_kwargs)

        if self.force_ocr:
            kwargs["force_ocr"] = True

        if self.language_detection:
            from kreuzberg import LanguageDetectionConfig

            kwargs["language_detection"] = LanguageDetectionConfig()

        if self.keyword_extraction:
            from kreuzberg import KeywordConfig

            kwargs["keywords"] = KeywordConfig()

        return ExtractionConfig(**kwargs)


class KreuzbergExtractor:
    """High-level extractor wrapping Kreuzberg's API.

    Provides async and sync extraction with pipeline-friendly output.
    """

    def __init__(self, config: KreuzbergConfig | None = None):
        self.config = config or KreuzbergConfig()
        self._kreuzberg_config = self.config.to_kreuzberg_config()

    async def extract_file(self, file_path: str | Path) -> KreuzbergExtractionResult:
        """Extract content from a single file asynchronously.

        Args:
            file_path: Path to the document file.

        Returns:
            KreuzbergExtractionResult with structured output.
        """
        if not HAS_KREUZBERG:
            return KreuzbergExtractionResult(
                success=False,
                file_path=str(file_path),
                error="Kreuzberg is not installed. Install with: pip install kreuzberg",
            )

        path = Path(file_path)
        if not path.exists():
            return KreuzbergExtractionResult(
                success=False,
                file_path=str(path),
                error=f"File not found: {path}",
            )

        _internal_logger.info(f"Extracting: {path.name}")

        try:
            result: ExtractionResult = await extract_file(
                path, config=self._kreuzberg_config
            )
            return self._build_result(result, str(path))

        except Exception as e:
            _internal_logger.error(f"Extraction failed for {path.name}: {e}")
            return KreuzbergExtractionResult(
                success=False,
                file_path=str(path),
                error=str(e),
            )

    def extract_file_sync(self, file_path: str | Path) -> KreuzbergExtractionResult:
        """Extract content from a single file synchronously.

        Args:
            file_path: Path to the document file.

        Returns:
            KreuzbergExtractionResult with structured output.
        """
        if not HAS_KREUZBERG:
            return KreuzbergExtractionResult(
                success=False,
                file_path=str(file_path),
                error="Kreuzberg is not installed. Install with: pip install kreuzberg",
            )

        path = Path(file_path)
        if not path.exists():
            return KreuzbergExtractionResult(
                success=False,
                file_path=str(path),
                error=f"File not found: {path}",
            )

        _internal_logger.info(f"Extracting (sync): {path.name}")

        try:
            from kreuzberg import extract_file_sync as extract_sync

            result: ExtractionResult = extract_sync(path, config=self._kreuzberg_config)
            return self._build_result(result, str(path))

        except Exception as e:
            _internal_logger.error(f"Sync extraction failed for {path.name}: {e}")
            return KreuzbergExtractionResult(
                success=False,
                file_path=str(path),
                error=str(e),
            )

    def extract_files_batch(
        self, file_paths: list[str | Path]
    ) -> list[KreuzbergExtractionResult]:
        """Extract content from multiple files synchronously (parallel).

        Args:
            file_paths: List of document file paths.

        Returns:
            List of KreuzbergExtractionResult, one per file.
        """
        if not HAS_KREUZBERG:
            return [
                KreuzbergExtractionResult(
                    success=False,
                    file_path=str(p),
                    error="Kreuzberg is not installed. Install with: pip install kreuzberg",
                )
                for p in file_paths
            ]

        valid_paths = [Path(p) for p in file_paths if Path(p).exists()]
        missing = [
            KreuzbergExtractionResult(
                success=False,
                file_path=str(p),
                error=f"File not found: {p}",
            )
            for p in file_paths
            if not Path(p).exists()
        ]

        if not valid_paths:
            _internal_logger.warning("No valid files for batch extraction")
            return missing

        _internal_logger.info(f"Batch extracting {len(valid_paths)} files")

        try:
            results = kreuzberg_batch_sync(valid_paths, config=self._kreuzberg_config)
            batch_results = [
                self._build_result(r, str(p)) for r, p in zip(results, valid_paths, strict=True)
            ]
            return batch_results + missing

        except Exception as e:
            _internal_logger.error(f"Batch extraction failed: {e}")
            return missing + [
                KreuzbergExtractionResult(
                    success=False,
                    file_path=str(p),
                    error=str(e),
                )
                for p in valid_paths
            ]

    def _build_result(
        self, kreuzberg_result: ExtractionResult, file_path: str
    ) -> KreuzbergExtractionResult:
        """Convert Kreuzberg's ExtractionResult to our structured format.

        Args:
            kreuzberg_result: Raw result from Kreuzberg.
            file_path: Original file path for reference.

        Returns:
            KreuzbergExtractionResult with standardized fields.
        """
        content = getattr(kreuzberg_result, "content", "")
        tables = getattr(kreuzberg_result, "tables", [])
        metadata = getattr(kreuzberg_result, "metadata", {}) or {}
        chunks = getattr(kreuzberg_result, "chunks", []) or []
        pages = getattr(kreuzberg_result, "pages", []) or []
        images = getattr(kreuzberg_result, "images", []) or []
        detected_languages = getattr(kreuzberg_result, "detected_languages", []) or []
        keywords = getattr(kreuzberg_result, "keywords", []) or []

        # Normalize chunks to our schema
        normalized_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data: dict[str, Any] = {
                "text": chunk.get("text", chunk.get("content", "")) if isinstance(chunk, dict) else str(chunk),
                "index": i,
                "source_file": file_path,
            }
            if isinstance(chunk, dict):
                chunk_data.update({
                    k: v for k, v in chunk.items()
                    if k not in ("text", "content") and v is not None
                })
            normalized_chunks.append(chunk_data)

        # Normalize tables to our schema
        normalized_tables = []
        for table in tables:
            table_data: dict[str, Any] = {
                "markdown": table.get("markdown", ""),
            }
            if "cells" in table:
                table_data["cells"] = table["cells"]
            if "has_header" in table:
                table_data["has_header"] = table["has_header"]
            normalized_tables.append(table_data)

        # Normalize pages
        normalized_pages = []
        for page in pages:
            page_data: dict[str, Any] = {}
            if isinstance(page, dict):
                page_data = dict(page)
            else:
                page_data = {"content": str(page)}
            page_data["source_file"] = file_path
            normalized_pages.append(page_data)

        # Normalize images
        normalized_images = []
        for img in images:
            img_data: dict[str, Any] = {}
            if isinstance(img, dict):
                img_data = dict(img)
            else:
                img_data = {"path": str(img)}
            img_data["source_file"] = file_path
            normalized_images.append(img_data)

        result = KreuzbergExtractionResult(
            success=True,
            file_path=file_path,
            content=content,
            chunks=normalized_chunks,
            tables=normalized_tables,
            metadata=metadata,
            pages=normalized_pages,
            images=normalized_images,
            detected_languages=detected_languages,
            keywords=keywords,
            char_count=len(content),
            table_count=len(normalized_tables),
            chunk_count=len(normalized_chunks),
        )

        _internal_logger.info(
            f"Extracted {file_path}: {result.char_count} chars, "
            f"{result.chunk_count} chunks, {result.table_count} tables"
        )

        return result
