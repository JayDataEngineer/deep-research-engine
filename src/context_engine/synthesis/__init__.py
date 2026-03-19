"""Synthesis module - Optional output generation (separate CLI).

Generates formatted outputs from context:
- Twitter threads
- Podcast scripts
- Research reports

These are SEPARATE CLI commands, NOT part of the main pipeline.

Usage:
    uv run context-engine synthesize --session-id abc123 --format twitter
    uv run context-engine synthesize --session-id abc123 --format podcast
    uv run context-engine synthesize --session-id abc123 --format report
"""

from .base import (
    BaseSynthesizer,
    get_context_for_synthesis,
    parse_json_response,
)
from .twitter import (
    Tweet,
    TweetThread,
    synthesize_twitter_thread,
    format_thread_output,
)
from .podcast import (
    DialogueLine,
    PodcastSegment,
    PodcastScript,
    synthesize_podcast,
    format_podcast_output,
)
from .report import (
    ReportSection,
    Report,
    synthesize_report,
    format_report_output,
)

__all__ = [
    # Base
    "BaseSynthesizer",
    "get_context_for_synthesis",
    "parse_json_response",
    # Twitter
    "Tweet",
    "TweetThread",
    "synthesize_twitter_thread",
    "format_thread_output",
    # Podcast
    "DialogueLine",
    "PodcastSegment",
    "PodcastScript",
    "synthesize_podcast",
    "format_podcast_output",
    # Report
    "ReportSection",
    "Report",
    "synthesize_report",
    "format_report_output",
]
