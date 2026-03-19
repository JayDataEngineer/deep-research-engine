"""Tool tracking and management utilities for the deep research graph."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage


@dataclass
class ToolCallRecord:
    """Record of a tool call with timing and result information."""
    tool_name: str
    input_params: Dict[str, Any]
    output_result: Any
    duration_ms: float
    timestamp: datetime
    success: bool = True
    error_message: str = None


class AgentTracker(BaseCallbackHandler):
    """Tracks agent activity including LLM calls and tool usage.

    Inherits from BaseCallbackHandler to work with LangChain's callback system.
    """

    def __init__(self):
        super().__init__()
        self.llm_calls = 0
        self.tool_calls = []
        self.start_time = datetime.now()
        self._current_llm_start: float = 0
        self._current_tool_start: float = 0
        self._current_tool_name: str = ""

    def record_llm_call(self):
        """Increment the LLM call counter."""
        self.llm_calls += 1

    def record_tool_call(self, tool_name: str, input_params: Dict[str, Any],
                         output_result: Any, duration_ms: float,
                         success: bool = True, error_message: str = None):
        """Record a tool call with its details."""
        record = ToolCallRecord(
            tool_name=tool_name,
            input_params=input_params,
            output_result=output_result,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        self.tool_calls.append(record)

    # ========== LangChain Callback Methods ==========

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing."""
        self._current_llm_start = datetime.now().timestamp()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes processing."""
        self.record_llm_call()

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing."""
        self._current_tool_start = datetime.now().timestamp()
        # Get tool name from serialized dict or kwargs
        if serialized and isinstance(serialized, dict):
            self._current_tool_name = serialized.get("name", kwargs.get("name", "unknown"))
        else:
            self._current_tool_name = kwargs.get("name", "unknown")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing."""
        duration_ms = (datetime.now().timestamp() - self._current_tool_start) * 1000
        self.record_tool_call(
            tool_name=self._current_tool_name,
            input_params={},
            output_result=output,
            duration_ms=duration_ms
        )

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error."""
        duration_ms = (datetime.now().timestamp() - self._current_tool_start) * 1000
        self.record_tool_call(
            tool_name=self._current_tool_name,
            input_params={},
            output_result=None,
            duration_ms=duration_ms,
            success=False,
            error_message=str(error)
        )

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain (phase) starts."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain (phase) ends."""
        pass

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain encounters an error."""
        pass

    def on_agent_action(self, action: Any, **kwargs) -> Any:
        """Called when agent takes an action."""
        pass

    def on_agent_finish(self, finish: Any, **kwargs) -> Any:
        """Called when agent finishes."""
        pass

    def on_text(self, text: str, **kwargs) -> None:
        """Called when text is emitted."""
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called on each new token (streaming)."""
        pass

    @property
    def run_inline(self) -> bool:
        """Whether to run inline (as part of the execution) or asynchronously.

        LangChain callbacks may check this property to determine callback behavior.
        """
        return True


def track_tool_calls(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract and track tool calls from agent messages.
    
    Args:
        messages: List of messages from the agent run
        
    Returns:
        List of dictionaries with tool call information
    """
    tool_calls = []
    
    for msg in messages:
        # Check for tool messages (results)
        if hasattr(msg, "name") and msg.name:  # ToolMessage
            tool_name = msg.name
            tool_output = str(msg.content) if hasattr(msg, "content") else ""

            # Look for the corresponding tool call in previous messages
            # For now, just record what we have
            tool_calls.append({
                "tool": tool_name,
                "input": "",  # Would need to parse from previous messages
                "output": tool_output[:2000],  # Truncate for storage
                "output_size": len(tool_output),
                "duration": 0,  # Not available from message history
            })

        # Also check for AIMessage with tool_calls
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tool_calls.append({
                    "tool": tool_name,
                    "input": str(tool_args)[:500],
                    "output": "",  # Output will be in a separate ToolMessage
                    "output_size": 0,
                    "duration": 0,
                })
    
    return tool_calls


def aggregate_tool_usage(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics about tool usage.
    
    Args:
        tool_calls: List of tool call records
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not tool_calls:
        return {}
    
    stats = {
        "total_calls": len(tool_calls),
        "successful_calls": len([call for call in tool_calls if "error" not in call]),
        "failed_calls": len([call for call in tool_calls if "error" in call]),
        "unique_tools": list(set(call["tool"] for call in tool_calls)),
        "tool_frequency": {},
        "total_duration": sum(call.get("duration", 0) for call in tool_calls),
    }
    
    # Count frequency of each tool
    for call in tool_calls:
        tool_name = call["tool"]
        if tool_name in stats["tool_frequency"]:
            stats["tool_frequency"][tool_name] += 1
        else:
            stats["tool_frequency"][tool_name] = 1
    
    return stats