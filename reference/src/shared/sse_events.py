"""Simple SSE Events for Streaming.

This module provides clean, simple SSE event formatting for streaming LangGraph events.
Unlike the complex Vercel AI SDK protocol, these events are straightforward JSON objects
that are easy to debug, test, and extend.

Usage:
    from src.shared.sse_events import SSEvent
    
    # In your streaming generator
    yield SSEvent.thinking("Searching database...")
    yield SSEvent.text("Hello! ")
    yield SSEvent.tool_call("search", {"query": "..."})
    yield SSEvent.tool_result("search", {"results": [...]})
"""

import json
from typing import Any, Dict, Optional, Literal


class SSEvent:
    """Simple SSE event formatter.
    
    All events follow the same format:
        data: {"type": "...", ...payload...}\n\n
    """
    
    @staticmethod
    def _format(data: Dict[str, Any]) -> str:
        """Format data as SSE event."""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    # ========================================================================
    # Core Events
    # ========================================================================
    
    @staticmethod
    def text(content: str) -> str:
        """Stream text content.
        
        Args:
            content: Text to stream
            
        Returns:
            SSE formatted event
        """
        return SSEvent._format({
            "type": "text",
            "content": content
        })
    
    @staticmethod
    def thinking(content: str, label: Optional[str] = None) -> str:
        """Stream thinking/reasoning content.
        
        Args:
            content: Thinking content
            label: Optional short label (e.g., "Searching", "Analyzing")
            
        Returns:
            SSE formatted event
        """
        data: Dict[str, Any] = {
            "type": "thinking",
            "content": content
        }
        if label:
            data["label"] = label
        return SSEvent._format(data)
    
    @staticmethod
    def start(message_id: Optional[str] = None) -> str:
        """Start of message stream.
        
        Args:
            message_id: Optional message ID
            
        Returns:
            SSE formatted event
        """
        data: Dict[str, Any] = {"type": "start"}
        if message_id:
            data["message_id"] = message_id
        return SSEvent._format(data)
    
    @staticmethod
    def end() -> str:
        """End of message stream."""
        return SSEvent._format({"type": "end"})
    
    # ========================================================================
    # Tool Events
    # ========================================================================
    
    @staticmethod
    def tool_call(name: str, args: Dict[str, Any], tool_call_id: Optional[str] = None) -> str:
        """Tool call started.
        
        Args:
            name: Tool name
            args: Tool arguments
            tool_call_id: Optional tool call ID
            
        Returns:
            SSE formatted event
        """
        data: Dict[str, Any] = {
            "type": "tool_call",
            "name": name,
            "args": args
        }
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        return SSEvent._format(data)
    
    @staticmethod
    def tool_result(name: str, result: Any, tool_call_id: Optional[str] = None) -> str:
        """Tool call completed.
        
        Args:
            name: Tool name
            result: Tool result
            tool_call_id: Optional tool call ID
            
        Returns:
            SSE formatted event
        """
        data: Dict[str, Any] = {
            "type": "tool_result",
            "name": name,
            "result": result
        }
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        return SSEvent._format(data)
    
    # ========================================================================
    # Progress Events (for deep research)
    # ========================================================================
    
    @staticmethod
    def progress(
        node: str,
        phase: str,
        thread_id: str,
        **kwargs: Any
    ) -> str:
        """Progress update for multi-step operations.
        
        Args:
            node: Current node name
            phase: Human-readable phase name
            thread_id: Thread/session ID
            **kwargs: Additional progress data
            
        Returns:
            SSE formatted event
        """
        data: Dict[str, Any] = {
            "type": "progress",
            "node": node,
            "phase": phase,
            "thread_id": thread_id,
            **kwargs
        }
        return SSEvent._format(data)
    
    # ========================================================================
    # Error Events
    # ========================================================================
    
    @staticmethod
    def error(message: str, recoverable: bool = True) -> str:
        """Error event.
        
        Args:
            message: Error message
            recoverable: Whether the error is recoverable
            
        Returns:
            SSE formatted event
        """
        return SSEvent._format({
            "type": "error",
            "message": message,
            "recoverable": recoverable
        })
    
    # ========================================================================
    # Custom Events
    # ========================================================================
    
    @staticmethod
    def custom(event_type: str, **data: Any) -> str:
        """Custom event type.
        
        Args:
            event_type: Custom event type name
            **data: Event payload
            
        Returns:
            SSE formatted event
        """
        return SSEvent._format({
            "type": event_type,
            **data
        })


# =============================================================================
# Convenience Functions
# =============================================================================

def format_sse(event_type: str, data: Dict[str, Any]) -> str:
    """Format custom SSE event.
    
    Args:
        event_type: Event type name
        data: Event data
        
    Returns:
        SSE formatted string
    """
    return SSEvent._format({"type": event_type, **data})
