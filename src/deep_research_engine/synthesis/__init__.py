"""Synthesis module for formatting research findings.

This module provides format-specific synthesis workers that transform
research findings into structured outputs using the content_formatter pattern.

Available formats:
- research_report: Comprehensive report with sections and citations
- markdown: Simple markdown-formatted output
- structured_json: JSON output for programmatic consumption

Usage:
    from deep_research_engine.synthesis import SynthesisWorker

    worker = SynthesisWorker(config)
    result = await worker.execute(state, config)
"""

from .formats.markdown.schema import MarkdownOutput, MarkdownSource
from .formats.research_report.schema import Citation, KeyFinding, ResearchReport, Section
from .formats.structured_json.schema import JSONFinding, JSONSource, StructuredJSON

__all__ = [
    "Citation",
    "JSONFinding",
    "JSONSource",
    "KeyFinding",
    # Markdown
    "MarkdownOutput",
    "MarkdownSource",
    # Research Report
    "ResearchReport",
    "Section",
    # Structured JSON
    "StructuredJSON",
]
