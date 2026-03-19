"""Long-Form Content Generator Extension."""

from src.extensions.core.base_research import ResearchExtension


class LongformGenExtension(ResearchExtension):
    """Long-form content generator for articles, essays, and newsletters."""

    # No custom profile needed - uses BaseResearchProfile
    # No custom output strategy needed - uses MarkdownFormatter
