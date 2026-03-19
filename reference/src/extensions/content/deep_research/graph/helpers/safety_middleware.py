"""Safety Summarization Middleware for Deep Research.

This module provides automatic content summarization when approaching
context limits, following LangGraph best practices.

Key Features:
- Uses LangChain's count_tokens_approximately for accurate token counting
- Supports configurable triggers (fraction, tokens) like SummarizationMiddleware
- LLM-based summarization with focus areas
- Graceful fallback to truncation if summarization fails
- Works with any LLM provider
- Configuration loaded from YAML config

Based on LangGraph patterns:
- https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/#summarize-messages

Configuration (in config.yaml):
    middleware:
      summarization:
        enabled: true
        local_model_context: 8192
        cloud_model_context: 32768
        trigger_type: "fraction"
        trigger_threshold: 0.8
        safety_margin: 0.2
        target_output_fraction: 0.7
        prefer_truncation: false
        preserve_start_chars: 500
        preserve_end_chars: 500
"""

import re
from typing import Optional, Any, List, Union, Tuple, Callable, Dict
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages.utils import count_tokens_approximately


class TriggerType(Enum):
    """Trigger types for summarization, matching LangGraph pattern."""
    FRACTION = "fraction"
    TOKENS = "tokens"


@dataclass
class SummarizationConfig:
    """Configuration for summarization middleware loaded from YAML.

    Maps directly to middleware.summarization in config.yaml.
    """
    # Context window sizes (tokens)
    local_model_context: int = 8192
    cloud_model_context: int = 32768

    # Trigger configuration
    trigger_type: str = "fraction"  # "fraction" or "tokens"
    trigger_threshold: float = 0.8  # 80% for fraction, or absolute tokens

    # Safety settings
    safety_margin: float = 0.2
    target_output_fraction: float = 0.7

    # Fallback behavior
    prefer_truncation: bool = False
    preserve_start_chars: int = 500
    preserve_end_chars: int = 500

    @classmethod
    def from_yaml(cls, yaml_config: Dict[str, Any]) -> "SummarizationConfig":
        """Create SummarizationConfig from YAML config dict.

        Args:
            yaml_config: The full YAML config dict (looks for middleware.summarization)

        Returns:
            SummarizationConfig with values from YAML or defaults
        """
        # Navigate to middleware.summarization
        middleware = yaml_config.get("middleware", {})
        if not isinstance(middleware, dict):
            middleware = {}

        summ_config = middleware.get("summarization", {})
        if not isinstance(summ_config, dict):
            summ_config = {}

        return cls(
            local_model_context=summ_config.get("local_model_context", 8192),
            cloud_model_context=summ_config.get("cloud_model_context", 32768),
            trigger_type=summ_config.get("trigger_type", "fraction"),
            trigger_threshold=summ_config.get("trigger_threshold", 0.8),
            safety_margin=summ_config.get("safety_margin", 0.2),
            target_output_fraction=summ_config.get("target_output_fraction", 0.7),
            prefer_truncation=summ_config.get("prefer_truncation", False),
            preserve_start_chars=summ_config.get("preserve_start_chars", 500),
            preserve_end_chars=summ_config.get("preserve_end_chars", 500),
        )


# Type alias for trigger configuration (matches LangGraph pattern)
TriggerConfig = Union[
    Tuple[TriggerType, Union[float, int]],
    List[Tuple[TriggerType, Union[float, int]]]
]


def count_tokens(content: str) -> int:
    """Count tokens in content using LangChain's tokenizer.

    Uses count_tokens_approximately which provides a consistent
    approximation across different models.

    Args:
        content: Text content to count tokens for

    Returns:
        Approximate token count
    """
    if not content:
        return 0
    return count_tokens_approximately([content])


def needs_summarization(
    content: str,
    config: Optional[SummarizationConfig] = None,
    max_context: Optional[int] = None,
    trigger: Optional[TriggerConfig] = None
) -> bool:
    """Check if content needs summarization.

    Args:
        content: The content to check
        config: SummarizationConfig from YAML (preferred)
        max_context: Maximum context window in tokens (fallback if no config)
        trigger: Trigger configuration (fraction or tokens threshold)
            - ("fraction", 0.8): Trigger at 80% of safe limit
            - ("tokens", 5000): Trigger at 5000 tokens
            If None, uses config or default

    Returns:
        True if content should be summarized
    """
    if not content:
        return False

    # Use config if provided, otherwise use defaults
    if config is None:
        config = SummarizationConfig()

    context_limit = max_context if max_context else config.local_model_context
    tokens = count_tokens(content)
    safe_limit = int(context_limit * (1 - config.safety_margin))

    # Parse trigger configuration (LangGraph pattern)
    if trigger is None:
        # Use config trigger
        if config.trigger_type == "fraction":
            threshold = int(safe_limit * config.trigger_threshold)
        else:  # tokens
            threshold = int(config.trigger_threshold)
    elif isinstance(trigger, tuple):
        trigger_type, value = trigger
        if trigger_type == TriggerType.FRACTION or trigger_type == "fraction":
            threshold = int(safe_limit * float(value))
        else:  # TOKENS
            threshold = int(value)
    else:
        # List of triggers - summarize if ANY threshold is met
        for t in trigger:
            if needs_summarization(content, config, max_context, t):
                return True
        return False

    return tokens > threshold


def truncate_content(
    content: str,
    max_tokens: int,
    config: Optional[SummarizationConfig] = None,
    preserve_start: Optional[int] = None,
    preserve_end: Optional[int] = None
) -> str:
    """Truncate content while preserving start and end.

    Args:
        content: Content to truncate
        max_tokens: Maximum tokens to keep
        config: SummarizationConfig from YAML (for preserve chars defaults)
        preserve_start: Characters to preserve at start (overrides config)
        preserve_end: Characters to preserve at end (overrides config)

    Returns:
        Truncated content with ellipsis in middle
    """
    if not content:
        return content

    # Use config for preserve chars if not specified
    if config is None:
        config = SummarizationConfig()

    start_chars = preserve_start if preserve_start is not None else config.preserve_start_chars
    end_chars = preserve_end if preserve_end is not None else config.preserve_end_chars

    current_tokens = count_tokens(content)
    if current_tokens <= max_tokens:
        return content

    # Approximate chars from tokens (roughly 4 chars per token)
    max_chars = max_tokens * 4

    if len(content) <= max_chars:
        return content

    # Preserve start and end
    start_part = content[:start_chars]
    end_part = content[-end_chars:]

    # Calculate how much middle we can keep
    remaining_chars = max_chars - start_chars - end_chars - 50  # 50 for ellipsis

    if remaining_chars > 0:
        middle_start = start_chars
        middle_end = start_chars + remaining_chars
        middle_part = content[middle_start:middle_end]
        return f"{start_part}\n\n... [content summarized] ...\n\n{middle_part}\n\n... [content summarized] ...\n\n{end_part}"
    else:
        return f"{start_part}\n\n... [content truncated - {current_tokens - max_tokens} tokens removed] ...\n\n{end_part}"


async def summarize_content(
    content: str,
    llm: Any,
    max_output_tokens: int = 2000,
    run_logger: Optional[Any] = None,
    focus_areas: Optional[List[str]] = None
) -> str:
    """Summarize content using LLM.

    Follows the LangGraph pattern of using LLM-based summarization
    to preserve semantic meaning while reducing token count.

    Args:
        content: Content to summarize
        llm: LLM instance to use for summarization
        max_output_tokens: Maximum tokens in summary
        run_logger: Optional logger for debugging
        focus_areas: Optional list of topics to focus on

    Returns:
        Summarized content
    """
    if not content:
        return content

    focus_instruction = ""
    if focus_areas:
        focus_instruction = f"\n\nPay special attention to these topics:\n" + "\n".join(f"- {area}" for area in focus_areas)

    # LangGraph-style summary prompt
    prompt = f"""<role>
Content Summarization Assistant
</role>

<primary_objective>
Your objective is to extract and summarize the most important information from the content below, reducing token count while preserving all key facts, insights, and citations.
</primary_objective>

<constraints>
Target length: ~{max_output_tokens} tokens (concise but comprehensive)
{focus_instruction}
</constraints>

<instructions>
1. Identify the main themes and topics in the content
2. Extract key findings, facts, and insights
3. Preserve important quotes, citations, or source references
4. Note any actionable conclusions or recommendations
5. Structure the summary clearly with headers if appropriate
</instructions>

<content_to_summarize>
{"-" * 60}
{content}
{"-" * 60}
</content_to_summarize>

Provide ONLY the summarized content. Do not include any meta-commentary."""

    try:
        from langchain_core.messages import SystemMessage

        if run_logger:
            current_tokens = count_tokens(content)
            run_logger._write_to_file(
                f"[SUMMARIZATION] Summarizing {current_tokens} tokens -> target {max_output_tokens}"
            )

        # CRITICAL: Use SystemMessage because the content to summarize is SYSTEM-GENERATED
        # (research findings, context, etc.) - NOT user input
        response = await llm.ainvoke([SystemMessage(content=prompt)])

        if hasattr(response, 'content'):
            summary = response.content
        else:
            summary = str(response)

        if run_logger:
            summary_tokens = count_tokens(summary)
            run_logger._write_to_file(
                f"[SUMMARIZATION] Summary produced: {summary_tokens} tokens ({len(summary)} chars)"
            )

        return summary

    except Exception as e:
        if run_logger:
            run_logger._write_to_file(f"[SUMMARIZATION] Failed: {e}, falling back to truncation")
        # Fallback to truncation
        return truncate_content(content, max_output_tokens)


async def ensure_content_fits(
    content: str,
    llm: Any,
    yaml_config: Optional[Dict[str, Any]] = None,
    run_logger: Optional[Any] = None,
    focus_areas: Optional[List[str]] = None,
) -> str:
    """Ensure content fits within context limits.

    Main entry point following LangGraph pattern:
    1. Check if summarization is needed using trigger
    2. If yes, use LLM to summarize content
    3. If summarization fails, fall back to truncation

    Args:
        content: Content to check/process
        llm: LLM instance for summarization
        yaml_config: Full YAML config dict (reads middleware.summarization)
        run_logger: Optional logger
        focus_areas: Topics to preserve during summarization

    Returns:
        Content guaranteed to fit within limits
    """
    if not content:
        return content

    # Load config from YAML
    config = SummarizationConfig.from_yaml(yaml_config) if yaml_config else SummarizationConfig()

    # Calculate safe limit (account for response)
    safe_limit = int(config.local_model_context * (1 - config.safety_margin))

    if not needs_summarization(content, config):
        return content

    current_tokens = count_tokens(content)

    if run_logger:
        run_logger._write_to_file(
            f"[SUMMARIZATION] Content exceeds limit: {current_tokens} tokens > {safe_limit} safe limit"
        )

    if config.prefer_truncation:
        return truncate_content(content, safe_limit, config)

    # Calculate target size for summary
    target_tokens = int(safe_limit * config.target_output_fraction)

    return await summarize_content(
        content=content,
        llm=llm,
        max_output_tokens=target_tokens,
        run_logger=run_logger,
        focus_areas=focus_areas
    )


async def ensure_findings_fit(
    findings: str,
    query: str,
    llm: Any,
    yaml_config: Optional[Dict[str, Any]] = None,
    run_logger: Optional[Any] = None,
) -> str:
    """Ensure worker findings fit within context limits.

    Specialized version for synthesis that extracts query-relevant focus areas.

    Args:
        findings: Worker findings from Context Engine
        query: Original research query (used to extract focus areas)
        llm: LLM instance for summarization
        yaml_config: Full YAML config dict (reads middleware.summarization)
        run_logger: Optional logger

    Returns:
        Findings guaranteed to fit within limits
    """
    if not findings:
        return findings

    # Extract key terms from query for focus (semantic relevance)
    query_terms = re.findall(r'\b\w{4,}\b', query.lower())
    focus_areas = list(set(query_terms))[:5]  # Top 5 unique terms

    return await ensure_content_fits(
        content=findings,
        llm=llm,
        yaml_config=yaml_config,
        run_logger=run_logger,
        focus_areas=focus_areas,
    )


class ContentSummarizer:
    """Content summarization class following LangGraph middleware patterns.

    Can be used as a reusable component with configurable settings.
    Similar to SummarizationMiddleware but for arbitrary content (not messages).

    Example:
        summarizer = ContentSummarizer.from_yaml(yaml_config)
        summarized = await summarizer.process(content, llm)
    """

    def __init__(
        self,
        config: Optional[SummarizationConfig] = None,
    ):
        """Initialize content summarizer.

        Args:
            config: SummarizationConfig from YAML (or defaults)
        """
        self.config = config or SummarizationConfig()

    @classmethod
    def from_yaml(cls, yaml_config: Dict[str, Any]) -> "ContentSummarizer":
        """Create ContentSummarizer from YAML config.

        Args:
            yaml_config: Full YAML config dict

        Returns:
            ContentSummarizer with config from YAML
        """
        config = SummarizationConfig.from_yaml(yaml_config)
        return cls(config=config)

    async def process(
        self,
        content: str,
        llm: Any,
        run_logger: Optional[Any] = None,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """Process content to ensure it fits limits.

        Args:
            content: Content to process
            llm: LLM instance for summarization
            run_logger: Optional logger
            focus_areas: Topics to preserve during summarization

        Returns:
            Content guaranteed to fit within limits
        """
        return await ensure_content_fits(
            content=content,
            llm=llm,
            yaml_config=None,  # Use self.config directly
            run_logger=run_logger,
            focus_areas=focus_areas,
        )

    async def process_with_config(
        self,
        content: str,
        llm: Any,
        run_logger: Optional[Any] = None,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """Process content using the stored config.

        This is the main method that uses self.config for all settings.
        """
        if not content:
            return content

        safe_limit = int(self.config.local_model_context * (1 - self.config.safety_margin))

        if not needs_summarization(content, self.config):
            return content

        current_tokens = count_tokens(content)

        if run_logger:
            run_logger._write_to_file(
                f"[SUMMARIZATION] Content exceeds limit: {current_tokens} tokens > {safe_limit} safe limit"
            )

        if self.config.prefer_truncation:
            return truncate_content(content, safe_limit, self.config)

        target_tokens = int(safe_limit * self.config.target_output_fraction)

        return await summarize_content(
            content=content,
            llm=llm,
            max_output_tokens=target_tokens,
            run_logger=run_logger,
            focus_areas=focus_areas
        )

    def __call__(
        self,
        content: str,
        llm: Any,
        run_logger: Optional[Any] = None,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """Synchronous wrapper (raises error - must use async)."""
        raise RuntimeError(
            "ContentSummarizer is async-only. Use await summarizer.process(...) instead."
        )
