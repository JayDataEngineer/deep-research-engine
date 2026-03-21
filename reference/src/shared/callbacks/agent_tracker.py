"""Agent Tracker Callback - Combined LLM and tool call tracking.

This module provides a unified callback handler for tracking LangChain/LangGraph
agent execution metrics including:
- LLM invocations
- Tool calls with timing
- Token usage (when available)

Usage:
    from .callbacks import AgentTracker

    tracker = AgentTracker()
    agent_config = {
        "callbacks": [tracker],
        "debug": True,
    }
    result = await agent.ainvoke(input_data, config=agent_config)

    # Access metrics
    print(f"LLM calls: {tracker.llm_calls}")
    print(f"Tool calls: {len(tracker.tool_calls)}")
    print(f"Total tokens: {tracker.total_tokens}")
"""

import logging
import time
from typing import Any, Dict, List
from collections import Counter

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class AgentTracker(BaseCallbackHandler):
    """Combined tracker for LLM calls, tool calls, and token usage.

    Tracks:
    - Number of LLM invocations
    - Tool calls with timing information
    - Token usage (prompt + completion when available)

    Attributes:
        llm_calls: Number of LLM invocations
        tool_calls: List of tool call records with timing
        total_tokens: Total token count (estimated when not provided)
        prompt_tokens: Total prompt tokens
        completion_tokens: Total completion tokens
    """

    def __init__(self, track_tokens: bool = True):
        """Initialize the agent tracker.

        Args:
            track_tokens: If True, track token usage (may not always be available)
        """
        super().__init__()
        self.llm_calls: int = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self._tool_start_times: Dict[str, float] = {}

        # Token tracking
        self.track_tokens = track_tokens
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts a new invocation.

        Args:
            serialized: Serialized LLM information
            prompts: List of prompt strings
            **kwargs: Additional keyword arguments
        """
        self.llm_calls += 1

        if logger.isEnabledFor(logging.DEBUG):
            model_name = serialized.get("id", serialized.get("name", "unknown"))
            total_prompt_chars = sum(len(p) for p in prompts)
            logger.debug(f"[AgentTracker] LLM start: model={model_name}, prompt_chars={total_prompt_chars}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes an invocation.

        Args:
            response: LLM result with token usage information
            **kwargs: Additional keyword arguments
        """
        # Extract token usage if available
        if self.track_tokens and hasattr(response, "llm_output"):
            llm_output = response.llm_output
            if hasattr(llm_output, "token_usage"):
                token_usage = llm_output.token_usage
                if token_usage:
                    self.prompt_tokens += token_usage.get("prompt_tokens", 0)
                    self.completion_tokens += token_usage.get("completion_tokens", 0)
                    self.total_tokens += token_usage.get("total_tokens", 0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[AgentTracker] LLM end: total_tokens={self.total_tokens}")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error.

        Args:
            error: The exception that occurred
            **kwargs: Additional keyword arguments
        """
        logger.warning(f"[AgentTracker] LLM error: {type(error).__name__}: {error}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool is called.

        Args:
            serialized: Serialized tool information (contains tool name)
            input_str: Tool input string
            **kwargs: Additional keyword arguments (including run_id)
        """
        tool_name = serialized.get("name", "unknown")
        self._tool_start_times[tool_name] = time.time()

        self.tool_calls.append({
            "tool": tool_name,
            "input": input_str[:500] if input_str else "",  # Truncate long inputs
            "timestamp": kwargs.get("run_id", ""),
        })

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[AgentTracker] Tool start: {tool_name}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes execution.

        Args:
            output: Tool output string
            **kwargs: Additional keyword arguments (including name)
        """
        # Find the most recent tool call and add duration/output info
        tool_name = kwargs.get("name", "unknown")

        if self.tool_calls and self._tool_start_times:
            # Get the most recent tool call for this tool
            for tc in reversed(self.tool_calls):
                if tc["tool"] == tool_name and "duration" not in tc:
                    duration = time.time() - self._tool_start_times.get(tool_name, time.time())
                    tc["duration"] = duration

                    # Store output info
                    if hasattr(output, "content"):
                        tc["output_size"] = len(str(output.content))
                        tc["output"] = str(output.content)[:2000]  # Truncate
                    else:
                        tc["output_size"] = len(str(output))
                        tc["output"] = str(output)[:2000]
                    break

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[AgentTracker] Tool end: {tool_name}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error.

        Args:
            error: The exception that occurred
            **kwargs: Additional keyword arguments
        """
        tool_name = kwargs.get("name", "unknown")
        logger.warning(f"[AgentTracker] Tool error: {tool_name} - {type(error).__name__}: {error}")

    def get_tool_summary(self) -> Dict[str, int]:
        """Get summary of tool calls by tool name.

        Returns:
            Dictionary mapping tool names to call counts
        """
        return dict(Counter(tc["tool"] for tc in self.tool_calls))

    def get_tool_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary for tool calls.

        Returns:
            Dictionary mapping tool names to timing stats
            {
                "tool_name": {"count": N, "total_time": X, "avg_time": Y}
            }
        """
        timing = {}
        for tool_name, count in self.get_tool_summary().items():
            calls_with_time = [tc for tc in self.tool_calls if tc["tool"] == tool_name and "duration" in tc]
            if calls_with_time:
                total_time = sum(tc["duration"] for tc in calls_with_time)
                timing[tool_name] = {
                    "count": count,
                    "total_time": total_time,
                    "avg_time": total_time / count,
                    "max_time": max(tc["duration"] for tc in calls_with_time),
                }
        return timing

    def get_summary(self) -> Dict[str, Any]:
        """Get complete summary of all metrics.

        Returns:
            Dictionary with all tracking metrics
        """
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": len(self.tool_calls),
            "tool_summary": self.get_tool_summary(),
            "tool_timing": self.get_tool_timing_summary(),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def reset(self) -> None:
        """Reset all tracking metrics to zero."""
        self.llm_calls = 0
        self.tool_calls = []
        self._tool_start_times = {}
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


class ChainTracker(BaseCallbackHandler):
    """Tracker for chain/graph execution metrics.

    Tracks chain execution timing and metadata for LangGraph chains.

    Attributes:
        chain_starts: Number of times chain started
        chain_ends: Number of times chain ended
        chain_errors: Number of chain errors
    """

    def __init__(self):
        """Initialize the chain tracker."""
        super().__init__()
        self.chain_starts: int = 0
        self.chain_ends: int = 0
        self.chain_errors: int = 0
        self._start_time: float | None = None

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts execution.

        Args:
            serialized: Serialized chain information
            inputs: Chain inputs
            **kwargs: Additional keyword arguments
        """
        self.chain_starts += 1
        self._start_time = time.time()

        if logger.isEnabledFor(logging.DEBUG):
            chain_name = serialized.get("id", serialized.get("name", "unknown"))
            logger.debug(f"[ChainTracker] Chain start: {chain_name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes execution.

        Args:
            outputs: Chain outputs
            **kwargs: Additional keyword arguments
        """
        self.chain_ends += 1
        duration = time.time() - self._start_time if self._start_time else 0

        if logger.isEnabledFor(logging.DEBUG):
            chain_name = kwargs.get("name", "unknown")
            logger.debug(f"[ChainTracker] Chain end: {chain_name}, duration={duration:.2f}s")

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain encounters an error.

        Args:
            error: The exception that occurred
            **kwargs: Additional keyword arguments
        """
        self.chain_errors += 1
        logger.warning(f"[ChainTracker] Chain error: {type(error).__name__}: {error}")

    def get_summary(self) -> Dict[str, int]:
        """Get summary of chain execution metrics.

        Returns:
            Dictionary with chain metrics
        """
        return {
            "chain_starts": self.chain_starts,
            "chain_ends": self.chain_ends,
            "chain_errors": self.chain_errors,
        }


# Convenience function for quick setup
def create_agent_tracker(track_tokens: bool = True) -> AgentTracker:
    """Create an agent tracker callback.

    Convenience function for quickly creating a tracker.

    Args:
        track_tokens: Whether to track token usage

    Returns:
        Configured AgentTracker instance
    """
    return AgentTracker(track_tokens=track_tokens)
