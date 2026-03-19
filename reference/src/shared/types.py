"""Shared type definitions for Pydantic models.

This module contains common type aliases and literals used across the codebase
to avoid duplication and ensure consistency.
"""

from typing import Literal, Union, Optional, Dict, List, Any

# =============================================================================
# MESSAGE ROLE TYPES
# =============================================================================

# Standard chat roles (most common)
ChatRole = Literal["system", "user", "assistant"]

# Extended roles including tool messages
ExtendedRole = Literal["system", "user", "assistant", "tool"]

# API-only roles (for API layer messages)
ApiRole = Literal["system", "user", "assistant"]

# =============================================================================
# STREAMING TYPES
# =============================================================================

StreamProtocol = Literal["data", "text", "raw"]

# =============================================================================
# CHAT MODE TYPES
# =============================================================================

# Chat modes for the API - matches actual extension names
ChatMode = Literal[
    "direct",  # Pure LLM, no tools
    "tool_call",  # Smart router with tools
    "deep_agent",  # Multi-agent deep reasoning
    "deep_research", "research"  # Multi-agent research with citations
]
# Legacy mode type (deprecated, kept for data migration)
Mode = Literal["standard", "companion", "rp", "deep_research"]

# =============================================================================
# PERSONALITY TYPES
# =============================================================================

Personality = Literal["friendly", "professional", "witty", "supportive"]

# =============================================================================
# CONTENT TYPES
# =============================================================================

ContentType = Literal["text", "image_url", "input_audio", "audio", "video_url", "video"]
TextContent = Union[str, List[Dict[str, Any]]]

# =============================================================================
# RESEARCH TYPES
# =============================================================================

UseCase = Literal["rag", "podcast", "trading", "osint", "news", "content", "web_research"]

# =============================================================================
# AUDIO TYPES
# =============================================================================

AudioFormat = Literal["wav", "mp3", "ogg"]
VoiceType = Literal["Cherry", "Serena", "Ethan", "Chelsie"]

# =============================================================================
# RESPONSE TYPES
# =============================================================================

# Generic metadata dict type
MetadataDict = Dict[str, Any]

# Extensions dict type
ExtensionsDict = Dict[str, Any]

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def is_empty_string(value: Any) -> bool:
    """Check if a value is None or an empty/whitespace-only string.

    Args:
        value: Any value to check

    Returns:
        True if value is None, empty string, or whitespace-only string
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def is_valid_url_string(url: str) -> bool:
    """Check if a string is a valid URL format.

    Args:
        url: URL string to validate

    Returns:
        True if url starts with http:// or https://
    """
    return isinstance(url, str) and url.startswith(("http://", "https://"))
