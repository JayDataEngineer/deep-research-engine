"""Document router - detect input types and route to the right extractor.

Two extraction paths:
1. **Document path** → Kreuzberg (PDF, DOCX, XLSX, PPTX, HTML, email, archives, etc.)
2. **Telegram path** → Custom JSON parser + HTML extraction

This module replaces the hand-rolled extraction code in nodes/ingestion.py.
It provides a single entry point: ``route_documents()`` which takes a list
of file paths and returns structured extraction results.

Design:
- Documents flow through Kreuzberg (91+ formats, OCR, chunking, tables)
- Telegram exports use our custom parser (Telegram JSON schema is app-specific)
- Media files (audio, video, photos) are NOT routed here - they go through
  the multimodal pipeline in the ingestion node's Phase 2-6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging import get_loguru_logger
from .kreuzberg_extractor import (
    KreuzbergConfig,
    KreuzbergExtractionResult,
    KreuzbergExtractor,
)
from .telegram_parser import TelegramParser, has_telegram_export

_internal_logger = get_loguru_logger("ingestion.router")

# File extension sets for routing decisions
DOCUMENT_EXTENSIONS = {
    # PDF
    ".pdf", ".PDF",
    # Word
    ".docx", ".odt", ".doc",
    # Spreadsheets
    ".xlsx", ".xls", ".xlsb", ".xla", ".xlam", ".xltm", ".ods",
    # Presentations
    ".pptx", ".ppt", ".ppsx",
    # eBooks
    ".epub", ".fb2",
    # Markup
    ".html", ".htm", ".xhtml", ".xml",
    # Data
    ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv",
    # Text
    ".txt", ".md", ".markdown", ".rst", ".org", ".rtf",
    # Email
    ".eml", ".msg",
    # Archives
    ".zip", ".tar", ".tgz", ".gz", ".7z",
    # Academic
    ".bib", ".ris", ".nbib", ".enw", ".csl", ".tex", ".latex",
    ".typ", ".jats", ".ipynb", ".docbook", ".opml",
}


@dataclass
class DocumentRoutingResult:
    """Result from routing documents through extractors.

    All outputs are in a unified schema suitable for the content_classifier.
    """

    success: bool

    # Kreuzberg document extractions
    documents: list[KreuzbergExtractionResult] = field(default_factory=list)
    """Documents extracted via Kreuzberg (PDF, DOCX, XLSX, etc.)."""

    # Telegram extraction
    telegram: TelegramParser | None = None
    telegram_result: TelegramParser | None = None
    """Telegram extraction (if export detected)."""

    telegram_items: list[dict[str, Any]] = field(default_factory=list)
    """Parsed Telegram messages in unified item schema."""

    # Routing stats
    total_files_scanned: int = 0
    documents_routed: int = 0
    telegram_detected: bool = False

    # Errors
    errors: list[dict[str, str]] = field(default_factory=list)
    """Per-file errors for failed extractions."""

    # Summary
    total_text_chars: int = 0
    total_chunks: int = 0
    total_tables: int = 0
    total_telegram_messages: int = 0

    # Document chunks for downstream pipeline
    document_chunks: list[dict[str, Any]] = field(default_factory=list)
    """All chunks from all documents, ready for content_classifier."""

    # Items for multimodal pipeline (extracted from Telegram + doc metadata)
    pipeline_items: list[dict[str, Any]] = field(default_factory=list)
    """Unified items for the multimodal pipeline (audio/video/photos from Telegram)."""


class DocumentRouter:
    """Routes document files to the appropriate extractor.

    Usage:
        router = DocumentRouter(input_dir)
        result = await router.route()
        # result.documents → Kreuzberg extractions
        # result.telegram_items → Telegram messages
        # result.document_chunks → All chunks for content_classifier
    """

    def __init__(
        self,
        input_dir: Path,
        kreuzberg_config: KreuzbergConfig | None = None,
    ):
        self.input_dir = input_dir
        self.extractor = KreuzbergExtractor(kreuzberg_config)
        self.telegram_parser = TelegramParser(input_dir)

    async def route(self) -> DocumentRoutingResult:
        """Scan input and route all documents through extractors.

        This is the main entry point. It:
        1. Scans for Telegram exports
        2. Finds document files (PDF, DOCX, XLSX, etc.)
        3. Extracts via Kreuzberg (batch)
        4. Parses Telegram (if present)
        5. Returns unified results

        Returns:
            DocumentRoutingResult with all extraction outputs.
        """
        _internal_logger.info(f"Routing documents from: {self.input_dir}")

        # 1. Check for Telegram export
        telegram_detected = has_telegram_export(self.input_dir)
        telegram_items: list[dict[str, Any]] = []
        telegram_result = None

        if telegram_detected:
            _internal_logger.info("Telegram export detected, parsing...")
            telegram_result = await self.telegram_parser.parse()
            if telegram_result.success:
                telegram_items = telegram_result.items
                _internal_logger.info(
                    f"Parsed {len(telegram_items)} Telegram messages"
                )
            else:
                _internal_logger.warning(
                    f"Telegram parsing failed: {telegram_result.error}"
                )

        # 2. Find document files (excluding Telegram subdirectories)
        doc_files = self._find_document_files()
        _internal_logger.info(f"Found {len(doc_files)} document files")

        # 3. Extract documents via Kreuzberg (batch)
        document_results: list[KreuzbergExtractionResult] = []
        if doc_files:
            document_results = self.extractor.extract_files_batch(doc_files)
            _internal_logger.info(
                f"Extracted {len(document_results)} documents via Kreuzberg"
            )

        # 4. Collect document chunks for downstream pipeline
        all_chunks = []
        for doc_result in document_results:
            all_chunks.extend(doc_result.chunks)

        # 5. Build pipeline items from Telegram + document metadata
        pipeline_items = self._build_pipeline_items(
            telegram_items, document_results
        )

        # 6. Calculate summary stats
        total_text_chars = sum(d.char_count for d in document_results if d.success)
        total_chunks = sum(d.chunk_count for d in document_results if d.success)
        total_tables = sum(d.table_count for d in document_results if d.success)

        # 7. Collect errors
        errors = [
            {"file": doc_result.file_path, "error": doc_result.error}
            for doc_result in document_results
            if not doc_result.success and doc_result.error
        ]

        result = DocumentRoutingResult(
            success=True,
            documents=document_results,
            telegram_result=telegram_result,
            telegram_items=telegram_items,
            total_files_scanned=len(doc_files),
            documents_routed=len(document_results),
            telegram_detected=telegram_detected,
            errors=errors,
            total_text_chars=total_text_chars,
            total_chunks=total_chunks,
            total_tables=total_tables,
            total_telegram_messages=len(telegram_items),
            document_chunks=all_chunks,
            pipeline_items=pipeline_items,
        )

        _internal_logger.info(
            f"Document routing complete: "
            f"{result.documents_routed} documents, "
            f"{result.total_chunks} chunks, "
            f"{result.total_tables} tables, "
            f"{result.total_telegram_messages} Telegram messages"
        )

        return result

    def route_sync(self) -> DocumentRoutingResult:
        """Synchronous version of route()."""
        _internal_logger.info(f"Routing documents (sync) from: {self.input_dir}")

        telegram_detected = has_telegram_export(self.input_dir)
        telegram_items: list[dict[str, Any]] = []
        telegram_result = None

        if telegram_detected:
            _internal_logger.info("Telegram export detected, parsing...")
            telegram_result = self.telegram_parser.parse_json()
            if telegram_result.success:
                telegram_items = telegram_result.items

        doc_files = self._find_document_files()
        document_results: list[KreuzbergExtractionResult] = []
        if doc_files:
            document_results = self.extractor.extract_files_batch(doc_files)

        all_chunks = []
        for doc_result in document_results:
            all_chunks.extend(doc_result.chunks)

        pipeline_items = self._build_pipeline_items(
            telegram_items, document_results
        )

        total_text_chars = sum(d.char_count for d in document_results if d.success)
        total_chunks = sum(d.chunk_count for d in document_results if d.success)
        total_tables = sum(d.table_count for d in document_results if d.success)

        errors = [
            {"file": doc_result.file_path, "error": doc_result.error}
            for doc_result in document_results
            if not doc_result.success and doc_result.error
        ]

        return DocumentRoutingResult(
            success=True,
            documents=document_results,
            telegram_result=telegram_result,
            telegram_items=telegram_items,
            total_files_scanned=len(doc_files),
            documents_routed=len(document_results),
            telegram_detected=telegram_detected,
            errors=errors,
            total_text_chars=total_text_chars,
            total_chunks=total_chunks,
            total_tables=total_tables,
            total_telegram_messages=len(telegram_items),
            document_chunks=all_chunks,
            pipeline_items=pipeline_items,
        )

    def _find_document_files(self) -> list[Path]:
        """Find document files in input directory.

        Excludes Telegram export subdirectories (photos/, voice_messages/,
        video_files/, stickers/, files/) since those are media files
        handled by the multimodal pipeline.

        Returns:
            List of document file paths.
        """
        tg_subdirs = {"photos", "voice_messages", "video_files", "stickers", "files"}
        files: list[Path] = []

        for ext in DOCUMENT_EXTENSIONS:
            for path in self.input_dir.rglob(f"*{ext}"):
                # Exclude files inside Telegram media subdirectories
                if path.parent.name in tg_subdirs:
                    continue
                # Exclude Telegram export itself (handled separately)
                if path.name == "messages.html" and has_telegram_export(self.input_dir):
                    continue
                files.append(path)

        return sorted(set(files))

    def _build_pipeline_items(
        self,
        telegram_items: list[dict[str, Any]],
        document_results: list[KreuzbergExtractionResult],
    ) -> list[dict[str, Any]]:
        """Build unified pipeline items from all extraction sources.

        Pipeline items are consumed by the multimodal pipeline phases
        (visual processing, acoustic processing, identity resolution).

        Args:
            telegram_items: Parsed Telegram messages.
            document_results: Kreuzberg document extractions.

        Returns:
            List of unified pipeline items.
        """
        items: list[dict[str, Any]] = []

        # Add Telegram items (already in unified schema)
        items.extend(telegram_items)

        # Add document text items
        for doc_result in document_results:
            if not doc_result.success:
                continue

            file_path = Path(doc_result.file_path)
            items.append({
                "type": "document",
                "path": doc_result.file_path,
                "text": doc_result.content,
                "timestamp": doc_result.metadata.get(
                    "creation_date",
                    doc_result.metadata.get(
                        "modified_date",
                        datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat() if file_path.exists() else "Unknown",
                    ),
                ),
                "tz_offset": "",
                "sender": doc_result.metadata.get("author", "Unknown"),
                "forwarded": False,
            })

        return items


# Import datetime locally to avoid top-level import issue
from datetime import datetime  # noqa: E402
