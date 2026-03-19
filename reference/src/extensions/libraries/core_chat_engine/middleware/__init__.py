"""Middleware for Core Chat Engine.

This module provides middleware components that are shared across all chat modes
(companion, roleplay, standard, deep_research).

The middleware layer handles:
- LangChain callback handlers
- Custom event emission (progress, images, tool results)
- Citation validation and management
- Context length management with summarization/clipping

Usage:
    from src.extensions.libraries.core_chat_engine.middleware import (
        CitationMiddleware,
        ContextMiddleware,
        CommunityScrapeMiddleware,
    )

    middleware = [
        CitationMiddleware(require_citations=True, min_citations=1),
        ContextMiddleware(max_input_tokens=100000, strategy="clip"),
    ]

    agent = create_deep_agent(middleware=middleware)
"""

from .callbacks import (
    TokenUsageHandler,
    WriterCallbackHandler,
)
from .community_scrape import CommunityScrapeMiddleware
from .citation import (
    CitationMiddleware,
    CitationModel,
    CitationReport,
    get_citation_report,
    validate_citations_response,
)
from .context import (
    ContextMiddleware,
    ContextStats,
    get_context_stats,
    truncate_text_for_context,
)

__all__ = [
    # Callbacks
    "TokenUsageHandler",
    "WriterCallbackHandler",
    # Community scraping
    "CommunityScrapeMiddleware",
    # Citation validation
    "CitationMiddleware",
    "CitationModel",
    "CitationReport",
    "get_citation_report",
    "validate_citations_response",
    # Context management
    "ContextMiddleware",
    "ContextStats",
    "get_context_stats",
    "truncate_text_for_context",
]
