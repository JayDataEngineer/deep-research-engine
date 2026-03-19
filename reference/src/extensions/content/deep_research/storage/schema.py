"""Utility functions for deep research.

This module provides helper functions for generating stable source IDs.
The SQLite-based ResearchDatabase class has been removed in favor of
LangGraph Store.
"""

from datetime import datetime


def get_session_id() -> str:
    """Generate a unique research session ID.

    Returns:
        Session ID string (timestamp-based)
    """
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def generate_source_id(url: str, title: str = "") -> str:
    """Generate a stable source ID from URL and title.

    Args:
        url: The source URL
        title: Optional title for additional uniqueness

    Returns:
        Full hexadecimal source ID (64 chars SHA256) - first 12 chars used as short ID
    """
    import hashlib

    payload = f"{url}{title}".encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def generate_source_id_short(url: str, title: str = "") -> str:
    """Generate a stable short source ID from URL and title.

    Args:
        url: The source URL
        title: Optional title for additional uniqueness

    Returns:
        12-character hexadecimal source ID (truncated SHA256)
    """
    return generate_source_id(url, title)[:12]


__all__ = [
    "get_session_id",
    "generate_source_id",
    "generate_source_id_short",
]
