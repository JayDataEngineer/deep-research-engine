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

    # All prompts
    ALL = [
        RESEARCH_PLANNER,
        RESEARCH_EVALUATOR,
        RESEARCH_REPORT,
        MARKDOWN_FORMAT,
        STRUCTURED_JSON,
    ]

    @classmethod
    def is_valid(cls, name: str) -> bool:
        """Check if prompt name is valid."""
        return name in cls.ALL
