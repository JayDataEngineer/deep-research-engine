"""Context Middleware - Manages content length with summarization/clipping.

This middleware provides:
1. Automatic content length management for long texts
2. Intelligent summarization using LLM
3. Smart clipping with context preservation
4. Token counting and management

Usage:
    from src.extensions.libraries.core_chat_engine.middleware import ContextMiddleware

    middleware = ContextMiddleware(
        max_input_tokens=100000,
        max_output_tokens=50000,
        strategy="summarize",  # or "clip"
        preserve_structure=True
    )

    agent = create_deep_agent(middleware=[middleware])
"""

import logging
import re
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict, Optional, List, Literal, Dict, Any
from collections.abc import Callable, Awaitable

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR CONTEXT MANAGEMENT
# =============================================================================

class ContextStats(BaseModel):
    """Statistics about context processing."""

    original_length: int = Field(description="Original content length in characters")
    original_tokens: int = Field(description="Estimated original token count")
    processed_length: int = Field(description="Processed content length in characters")
    processed_tokens: int = Field(description="Estimated processed token count")
    compression_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of processed to original (1.0 = no compression)"
    )
    strategy_used: str = Field(description="Strategy used (none, clip, summarize)")
    sections_preserved: int = Field(default=0, description="Number of structural sections preserved")
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about content processing"
    )


# =============================================================================
# MIDDLEWARE STATE
# =============================================================================

class ContextMiddlewareState(AgentState):
    """State schema for ContextMiddleware.

    Attributes:
        context_stats: Statistics about context processing.
            Marked as private so it's not included in the final agent state.
        original_content: Original unprocessed content.
            Marked as private.
    """

    context_stats: NotRequired[Annotated[ContextStats, PrivateStateAttr]]
    original_content: NotRequired[Annotated[str, PrivateStateAttr]]


class ContextMiddlewareStateUpdate(TypedDict):
    """State update for ContextMiddleware."""

    context_stats: ContextStats
    original_content: str


# =============================================================================
# CONTEXT MIDDLEWARE
# =============================================================================

class ContextMiddleware(AgentMiddleware):
    """Middleware for managing content length with summarization/clipping.

    This middleware:
    1. Monitors input and output content length
    2. Applies strategies when content exceeds limits
    3. Preserves structure when clipping (headers, lists, etc.)
    4. Can summarize content using LLM when needed

    Args:
        max_input_tokens: Maximum input tokens (default: 100000)
        max_output_tokens: Maximum output tokens (default: 50000)
        strategy: Strategy when exceeding limits ("clip", "summarize", or "hybrid")
        preserve_structure: If True, try to preserve structural elements
        summary_ratio: When summarizing, target this ratio of original size
        clip_from: When clipping, which part to keep ("start", "end", "middle")
        chars_per_token: Approximate characters per token (default: 4)
    """

    state_schema = ContextMiddlewareState

    def __init__(
        self,
        *,
        max_input_tokens: int = 100000,
        max_output_tokens: int = 50000,
        strategy: Literal["none", "clip", "summarize", "hybrid"] = "clip",
        preserve_structure: bool = True,
        summary_ratio: float = 0.3,
        clip_from: Literal["start", "end", "middle"] = "start",
        chars_per_token: float = 4.0,
    ):
        """Initialize the context middleware.

        Args:
            max_input_tokens: Maximum input tokens allowed.
            max_output_tokens: Maximum output tokens allowed.
            strategy: Strategy when exceeding limits.
            preserve_structure: If True, preserve structural elements.
            summary_ratio: Target ratio when summarizing.
            clip_from: Which part to keep when clipping.
            chars_per_token: Approximate characters per token.
        """
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.strategy = strategy
        self.preserve_structure = preserve_structure
        self.summary_ratio = max(0.1, min(1.0, summary_ratio))
        self.clip_from = clip_from
        self.chars_per_token = chars_per_token

        logger.info(
            f"[ContextMiddleware] Initialized: max_input={max_input_tokens}, "
            f"max_output={max_output_tokens}, strategy={strategy}, "
            f"preserve_structure={preserve_structure}"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # Rough estimate: chars / chars_per_token
        # For English text, 1 token ≈ 4 characters
        return int(len(text) / self.chars_per_token)

    def _preserve_structure_clip(
        self,
        text: str,
        max_chars: int
    ) -> tuple[str, int]:
        """Clip text while preserving structural elements.

        Preserves:
        - Headers (markdown #, ##, etc.)
        - List items
        - Code blocks
        - Section breaks

        Args:
            text: Text to clip
            max_chars: Maximum characters to keep

        Returns:
            Tuple of (clipped_text, sections_preserved)
        """
        if len(text) <= max_chars:
            return text, 0

        # Split into sections
        sections = []
        current_section = {"type": "text", "content": "", "lines": []}

        lines = text.split("\n")
        for line in lines:
            # Detect section type
            if line.strip().startswith("#"):
                # Header
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"type": "header", "content": line, "lines": [line]}
            elif line.strip().startswith("```"):
                # Code block
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"type": "code", "content": line, "lines": [line]}
            elif re.match(r'^\s*[-*+]\s', line.strip()):
                # List item
                current_section["type"] = "list"
                current_section["lines"].append(line)
                current_section["content"] = "\n".join(current_section["lines"])
            else:
                # Regular text
                current_section["type"] = "text"
                current_section["lines"].append(line)
                current_section["content"] = "\n".join(current_section["lines"])

        if current_section["content"]:
            sections.append(current_section)

        # Build result keeping sections until we hit the limit
        result = []
        current_length = 0

        for section in sections:
            section_length = len(section["content"]) + 1  # +1 for newline

            if current_length + section_length > max_chars:
                # This section would exceed limit, truncate it
                remaining = max_chars - current_length
                if remaining > 50:  # Only add if we have meaningful space
                    truncated = section["content"][:remaining]
                    result.append(truncated + "...")
                break

            result.append(section["content"])
            current_length += section_length

        clipped_text = "\n".join(result)
        sections_preserved = len([s for s in sections if s["content"] in clipped_text])

        return clipped_text, sections_preserved

    def _simple_clip(
        self,
        text: str,
        max_chars: int
    ) -> tuple[str, int]:
        """Simple clip without structure preservation.

        Args:
            text: Text to clip
            max_chars: Maximum characters to keep

        Returns:
            Tuple of (clipped_text, sections_preserved)
        """
        if len(text) <= max_chars:
            return text, 0

        if self.clip_from == "start":
            clipped = text[:max_chars-3] + "..."
        elif self.clip_from == "end":
            clipped = "..." + text[-(max_chars-3):]
        else:  # middle
            half = (max_chars - 6) // 2
            clipped = text[:half] + "..." + text[-half:]

        return clipped, 0

    def _summarize_text(self, text: str, target_ratio: float = 0.3) -> str:
        """Summarize text using LLM.

        Note: This is a placeholder. In production, you'd want to:
        1. Use the actual LLM from the context
        2. Cache summaries
        3. Handle async properly
        4. Add error handling

        Args:
            text: Text to summarize
            target_ratio: Target size as ratio of original

        Returns:
            Summarized text
        """
        target_length = max(200, int(len(text) * target_ratio))

        # For now, return a clipped version with summary indicator
        # In production, this would call an LLM
        logger.warning(
            "[ContextMiddleware] LLM summarization not yet implemented, "
            "falling back to smart clipping"
        )

        clipped, _ = self._preserve_structure_clip(text, target_length)
        return f"[SUMMARY] {clipped} [END SUMMARY]"

    def _process_content(
        self,
        content: str,
        max_tokens: int,
        is_input: bool = True
    ) -> tuple[str, ContextStats]:
        """Process content according to strategy.

        Args:
            content: Content to process
            max_tokens: Maximum tokens allowed
            is_input: True if input, False if output

        Returns:
            Tuple of (processed_content, stats)
        """
        original_length = len(content)
        original_tokens = self._estimate_tokens(content)

        if original_tokens <= max_tokens:
            # No processing needed
            stats = ContextStats(
                original_length=original_length,
                original_tokens=original_tokens,
                processed_length=original_length,
                processed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used="none",
                sections_preserved=0,
                warnings=[]
            )
            return content, stats

        # Need to process
        target_chars = int(max_tokens * self.chars_per_token)
        warnings = []

        if self.strategy == "clip":
            if self.preserve_structure:
                processed, sections = self._preserve_structure_clip(content, target_chars)
            else:
                processed, sections = self._simple_clip(content, target_chars)
            strategy_used = "clip"

        elif self.strategy == "summarize":
            processed = self._summarize_text(content, self.summary_ratio)
            sections = 0
            strategy_used = "summarize"
            warnings.append("LLM summarization not yet fully implemented")

        elif self.strategy == "hybrid":
            # Try summarization first, fall back to clipping
            if original_tokens > max_tokens * 2:
                processed = self._summarize_text(content, self.summary_ratio)
                sections = 0
                strategy_used = "hybrid_summarize"
            else:
                if self.preserve_structure:
                    processed, sections = self._preserve_structure_clip(content, target_chars)
                else:
                    processed, sections = self._simple_clip(content, target_chars)
                strategy_used = "hybrid_clip"
            warnings.append("LLM summarization not yet fully implemented")

        else:  # none
            processed = content
            sections = 0
            strategy_used = "none"
            warnings.append(f"Content exceeds {max_tokens} tokens but strategy is 'none'")

        processed_length = len(processed)
        processed_tokens = self._estimate_tokens(processed)
        compression_ratio = processed_tokens / original_tokens if original_tokens > 0 else 1.0

        stats = ContextStats(
            original_length=original_length,
            original_tokens=original_tokens,
            processed_length=processed_length,
            processed_tokens=processed_tokens,
            compression_ratio=compression_ratio,
            strategy_used=strategy_used,
            sections_preserved=sections,
            warnings=warnings
        )

        return processed, stats

    def before_agent(
        self,
        state: ContextMiddlewareState,
        runtime: Runtime,
        config: RunnableConfig
    ) -> ContextMiddlewareStateUpdate | None:
        """Process input content before agent execution.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with context stats.
        """
        # Skip if already processed
        if "context_stats" in state:
            return None

        # Initialize with empty stats
        return ContextMiddlewareStateUpdate(
            context_stats=ContextStats(
                original_length=0,
                original_tokens=0,
                processed_length=0,
                processed_tokens=0,
                compression_ratio=1.0,
                strategy_used="none",
                sections_preserved=0,
                warnings=[]
            ),
            original_content=""
        )

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Process input content in the request.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with processed content if needed.
        """
        # Process system message
        if request.system_message:
            processed, stats = self._process_content(
                request.system_message,
                self.max_input_tokens
            )
            if stats.processed_length != stats.original_length:
                logger.info(
                    f"[ContextMiddleware] Processed system message: "
                    f"{stats.original_tokens} -> {stats.processed_tokens} tokens "
                    f"({stats.compression_ratio:.2f} ratio)"
                )
                request.state["context_stats"] = stats
                request = request.override(system_message=processed)

        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Wrap model call to process output content.

        Args:
            request: Model request being processed.
            handler: Handler function to call.

        Returns:
            Model response with processed content if needed.
        """
        response = handler(request)

        # Process response content
        response_text = response.content if hasattr(response, 'content') else str(response)

        processed, stats = self._process_content(
            response_text,
            self.max_output_tokens,
            is_input=False
        )

        if stats.processed_length != stats.original_length:
            logger.info(
                f"[ContextMiddleware] Processed response: "
                f"{stats.original_tokens} -> {stats.processed_tokens} tokens "
                f"({stats.compression_ratio:.2f} ratio)"
            )

            # Update response with processed content
            if hasattr(response, 'content'):
                response.content = processed
            else:
                # If response is a string, we can't modify it directly
                # Log a warning
                logger.warning(
                    "[ContextMiddleware] Could not modify response content "
                    "(response is not a ModelResponse object)"
                )

        # Store stats in state
        request.state["context_stats"] = stats

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Async wrap model call to process output content.

        Args:
            request: Model request being processed.
            handler: Async handler function to call.

        Returns:
            Model response with processed content if needed.
        """
        response = await handler(request)

        # Process response content
        response_text = response.content if hasattr(response, 'content') else str(response)

        processed, stats = self._process_content(
            response_text,
            self.max_output_tokens,
            is_input=False
        )

        if stats.processed_length != stats.original_length:
            logger.info(
                f"[ContextMiddleware] Processed response: "
                f"{stats.original_tokens} -> {stats.processed_tokens} tokens "
                f"({stats.compression_ratio:.2f} ratio)"
            )

            # Update response with processed content
            if hasattr(response, 'content'):
                response.content = processed
            else:
                logger.warning(
                    "[ContextMiddleware] Could not modify response content "
                    "(response is not a ModelResponse object)"
                )

        # Store stats in state
        request.state["context_stats"] = stats

        return response


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_context_stats(state: Dict[str, Any]) -> Optional[ContextStats]:
    """Extract context stats from agent state.

    Args:
        state: Agent state dictionary

    Returns:
        ContextStats if available, None otherwise
    """
    stats_data = state.get("context_stats")
    if stats_data:
        if isinstance(stats_data, ContextStats):
            return stats_data
        # If it's a dict, deserialize it
        return ContextStats(**stats_data)
    return None


def truncate_text_for_context(
    text: str,
    max_tokens: int,
    strategy: str = "clip",
    preserve_structure: bool = True
) -> str:
    """Truncate text to fit within token limit.

    Utility function to truncate text outside of middleware context.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        strategy: Truncation strategy ("clip", "summarize", "hybrid")
        preserve_structure: If True, preserve structural elements

    Returns:
        Truncated text
    """
    middleware = ContextMiddleware(
        max_input_tokens=max_tokens,
        strategy=strategy,
        preserve_structure=preserve_structure
    )
    processed, _ = middleware._process_content(text, max_tokens)
    return processed
