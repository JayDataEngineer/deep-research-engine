"""Shared JSON and SSE serialization utilities.

Provides standardized functions for:
- SSE (Server-Sent Events) formatting
- JSON serialization with common patterns
- datetime/object serialization helpers
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# =============================================================================
# SSE (Server-Sent Events) Formatting
# =============================================================================

def format_sse(event_type: str, data: Dict[str, Any]) -> str:
    """Format data as an SSE (Server-Sent Events) message.

    Args:
        event_type: The type of event (e.g., "start", "text-delta", "finish")
        data: The data payload to send

    Returns:
        Formatted SSE string: "data: {json}\\n\\n"
    """
    return f"data: {json.dumps(data)}\n\n"


def format_sse_start(message_id: str) -> str:
    """Format an SSE start event.

    Args:
        message_id: Unique identifier for the message

    Returns:
        Formatted SSE start event
    """
    return format_sse("start", {"type": "start", "messageId": message_id})


def format_sse_text_delta(text: str, message_id: str) -> str:
    """Format an SSE text-delta event for streaming text.

    Args:
        text: The text delta to send
        message_id: Unique identifier for the message

    Returns:
        Formatted SSE text-delta event
    """
    return format_sse("text-delta", {
        "type": "text-delta",
        "id": message_id,
        "delta": text
    })


def format_sse_finish(finish_reason: str = "stop") -> str:
    """Format an SSE finish event.

    Args:
        finish_reason: Reason for finishing (e.g., "stop", "length", "error")

    Returns:
        Formatted SSE finish event
    """
    return format_sse("finish", {"type": "finish", "finishReason": finish_reason})


def format_sse_error(error_message: str) -> str:
    """Format an SSE error event.

    Args:
        error_message: The error message

    Returns:
        Formatted SSE error event
    """
    return format_sse("error", {"type": "error", "message": error_message})


# =============================================================================
# JSON Serialization Helpers
# =============================================================================

def json_compact(data: Any) -> str:
    """Serialize data to compact JSON (no spaces).

    Args:
        data: The data to serialize

    Returns:
        Compact JSON string
    """
    return json.dumps(data, separators=(',', ':'))


def json_pretty(data: Any, indent: int = 2) -> str:
    """Serialize data to pretty-printed JSON.

    Args:
        data: The data to serialize
        indent: Number of spaces for indentation (default: 2)

    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def json_for_log(data: Any, max_length: int = 500) -> str:
    """Serialize data to JSON for logging, with truncation.

    Args:
        data: The data to serialize
        max_length: Maximum length before truncation (default: 500)

    Returns:
        JSON string, truncated if necessary
    """
    json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if len(json_str) > max_length:
        json_str = json_str[:max_length] + "..."
    return json_str


# =============================================================================
# Custom JSON Encoders
# =============================================================================

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for common types.

    Handles:
    - datetime objects (ISO format)
    - UUID objects (string representation)
    - set objects (converted to list)
    - bytes objects (base64 encoded)
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return super().default(obj)


def json_serialize(data: Any, pretty: bool = False) -> str:
    """Serialize data to JSON using custom encoder.

    Args:
        data: The data to serialize
        pretty: Whether to pretty-print (default: False)

    Returns:
        JSON string
    """
    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False, cls=JSONEncoder)
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False, cls=JSONEncoder)


# =============================================================================
# Response Formatting Helpers
# =============================================================================

def create_success_response(
    message: str,
    data: Any = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a standardized success response.

    Args:
        message: Success message
        data: Optional data payload
        meta: Optional metadata

    Returns:
        Dictionary with success, message, data, and meta fields
    """
    response = {
        "success": True,
        "message": message
    }
    if data is not None:
        response["data"] = data
    if meta:
        response["meta"] = meta
    return response


def create_list_response(
    items: List[Any],
    total_count: int,
    page: int = 1,
    page_size: int = 50
) -> Dict[str, Any]:
    """Create a standardized list response with pagination.

    Args:
        items: List of items
        total_count: Total number of items
        page: Current page number
        page_size: Number of items per page

    Returns:
        Dictionary with items and pagination info
    """
    total_pages = (total_count + page_size - 1) // page_size
    return {
        "items": items,
        "pagination": {
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }
