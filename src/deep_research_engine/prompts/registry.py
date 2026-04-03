"""Prompt registry - Centralized prompt name definitions.

All prompts are stored in LangFuse. This module defines the names.

Usage:
    from deep_research_engine.prompts import PromptRegistry

    prompt = await get_prompt(PromptRegistry.RESEARCH_PLANNER)
"""


class PromptRegistry:
    """Centralized prompt name registry.

    Prompt names match those in LangFuse and the upload script.
    """

    # Boss Node
    RESEARCH_PLANNER = "research-planner"
    """Creates research plans by analyzing queries."""

    # Critic Node
    RESEARCH_EVALUATOR = "research-evaluator"
    """Evaluates research output quality."""

    # Synthesis Node
    RESEARCH_REPORT = "research-report"
    """Formats findings into research reports."""

    MARKDOWN_FORMAT = "markdown-format"
    """Formats findings into markdown."""

    STRUCTURED_JSON = "structured-json"
    """Formats findings into structured JSON."""

    # Identity Resolution
    IDENTITY_RESEARCH = "identity-research"
    """Deduces identities across multimodal data."""

    IDENTITY_FORMAT = "identity-format"
    """Formats identity research into structured mapping."""

    # Content Clustering
    CONTENT_CLUSTER = "content-cluster"
    """Groups content items into thematic clusters."""

    # All prompts
    ALL = [
        RESEARCH_PLANNER,
        RESEARCH_EVALUATOR,
        RESEARCH_REPORT,
        MARKDOWN_FORMAT,
        STRUCTURED_JSON,
        IDENTITY_RESEARCH,
        IDENTITY_FORMAT,
        CONTENT_CLUSTER,
    ]

    @classmethod
    def is_valid(cls, name: str) -> bool:
        """Check if prompt name is valid."""
        return name in cls.ALL
