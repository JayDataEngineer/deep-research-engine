"""Ingestion module - document extraction and routing.

Two extraction paths:
1. **Document path** (PDF, DOCX, XLSX, PPTX, HTML, email, archives, etc.)
   → Kreuzberg extraction → structured chunks → content classifier
2. **Multimodal path** (audio, video, photos)
   → 6-phase pipeline (temporal → visual → acoustic → GPU lifecycle → acoustic clustering → multimodal synthesis)

Cross-cutting:
- Telegram exports (JSON + HTML) → Kreuzberg for HTML, custom parser for JSON
- Content classification → LLM-based filler detection, entity/topic extraction
- Content clustering → Semantic grouping for downstream RAG indexing
"""

from .document_router import DocumentRouter, DocumentRoutingResult
from .kreuzberg_extractor import KreuzbergConfig, KreuzbergExtractionResult, KreuzbergExtractor
from .telegram_parser import TelegramExtractionResult, TelegramParser

__all__ = [
    "DocumentRouter",
    "DocumentRoutingResult",
    "KreuzbergConfig",
    "KreuzbergExtractionResult",
    "KreuzbergExtractor",
    "TelegramExtractionResult",
    "TelegramParser",
]
