"""LangChain callback handlers for Core Chat Engine.

Provides custom callback handlers for:
- Token usage tracking
- LangSmith logging
- Custom event emission via writer callbacks
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class TokenUsageHandler(BaseCallbackHandler):
    """Callback handler for tracking token usage.

    Collects token counts from LLM calls for cost tracking and limits.
    """

    def __init__(self):
        """Initialize the token usage handler."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Handle LLM completion.

        Args:
            response: LLM result with token usage
        """
        # Extract token usage from response
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                self.prompt_tokens += token_usage.get('prompt_tokens', 0)
                self.completion_tokens += token_usage.get('completion_tokens', 0)
                self.total_tokens += token_usage.get('total_tokens', 0)
                self.call_count += 1

                logger.debug(
                    f"[TOKENS] Prompt: {token_usage.get('prompt_tokens')}, "
                    f"Completion: {token_usage.get('completion_tokens')}, "
                    f"Total: {token_usage.get('total_tokens')}"
                )

    def get_usage(self) -> Dict[str, int]:
        """Get current token usage stats.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, call_count
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def reset(self) -> None:
        """Reset token usage counters."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0


class WriterCallbackHandler(BaseCallbackHandler):
    """Callback handler that emits events via a writer callback.

    Used for streaming events (CoT steps, progress, etc.) to the frontend.
    """

    def __init__(self, writer: Optional[Callable] = None):
        """Initialize the writer callback handler.

        Args:
            writer: Optional callback function for emitting events
        """
        self.writer = writer
        self.events_emitted: List[dict] = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Handle LLM start.

        Args:
            serialized: Serialized LLM config
            prompts: List of prompts
        """
        if self.writer:
            try:
                self.writer({
                    "type": "status",
                    "message": "Thinking...",
                    "icon": "brain",
                })
            except Exception as e:
                logger.warning(f"[WRITER] Failed to emit start event: {e}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Handle LLM completion.

        Args:
            response: LLM result
        """
        if self.writer:
            try:
                self.writer({
                    "type": "status",
                    "message": "Complete",
                    "icon": "check",
                })
            except Exception as e:
                logger.warning(f"[WRITER] Failed to emit end event: {e}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Handle tool start.

        Args:
            serialized: Serialized tool config
            input_str: Tool input
        """
        tool_name = serialized.get("name", "unknown")
        logger.info(f"[TOOL] Starting: {tool_name}")

        if self.writer:
            try:
                self.writer({
                    "type": "tool_start",
                    "tool_name": tool_name,
                    "input": input_str[:200],  # Truncate for logging
                })
            except Exception as e:
                logger.warning(f"[WRITER] Failed to emit tool start: {e}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Handle tool completion.

        Args:
            output: Tool output
        """
        logger.info(f"[TOOL] Completed: {len(output)} chars output")

        if self.writer:
            try:
                self.writer({
                    "type": "tool_end",
                    "output_length": len(output),
                })
            except Exception as e:
                logger.warning(f"[WRITER] Failed to emit tool end: {e}")

    def emit_custom(self, event: dict) -> None:
        """Emit a custom event.

        Args:
            event: Event data to emit
        """
        self.events_emitted.append(event)

        if self.writer:
            try:
                self.writer(event)
            except Exception as e:
                logger.warning(f"[WRITER] Failed to emit custom event: {e}")

    def get_emitted_events(self) -> List[dict]:
        """Get all emitted events.

        Returns:
            List of emitted events
        """
        return self.events_emitted.copy()


__all__ = [
    "TokenUsageHandler",
    "WriterCallbackHandler",
]

