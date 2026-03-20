"""Artifact Maker module for formatting research findings into artifacts.

This module provides format-specific artifact maker workers that transform
research findings into structured artifacts using the content_formatter pattern.

Available formats:
- research_report: Comprehensive report with sections and citations
- markdown: Simple markdown-formatted output
- structured_json: JSON output for programmatic consumption
- podcast_script: Creative transformation into a podcast script
- substack_article: Long-form writing for a Substack article
- twitter_thread: Social compression into a Twitter thread
- skill: Structured knowledge representation
- llms_txt: Documentation format
- code: Generated code

Usage:
    from deep_research_engine.workers.implementations.artifact_maker import ArtifactMakerWorker

    worker = ArtifactMakerWorker(config)
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
