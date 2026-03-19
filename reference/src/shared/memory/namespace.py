"""Namespace utilities for LangGraph Store.

DRY utility for constructing consistent namespaces across the system.
All memory storage should use these utilities to ensure consistent
session isolation and subcategory routing.

Namespace format: (session_id, "memories", subcategory)
- session_id: Research session identifier
- "memories": Fixed prefix for all auto-vectorized content
- subcategory: Type of content (dumps, web, insights)
"""

from typing import Tuple, Optional

# Subcategory constants
SUBCAT_DUMPS = "dumps"       # Entry Point A: Context overflow
SUBCAT_WEB = "web"           # Entry Point B: Webpage content
SUBCAT_INSIGHTS = "insights"  # Entry Point C: Worker findings


def make_namespace(
    session_id: str,
    subcategory: Optional[str] = None,
) -> Tuple[str, ...]:
    """Create a namespace tuple for LangGraph Store.

    Args:
        session_id: Research session identifier
        subcategory: Optional subcategory (dumps, web, insights)

    Returns:
        Namespace tuple for store operations

    Examples:
        >>> make_namespace("research_123")
        ("research_123", "memories")

        >>> make_namespace("research_123", SUBCAT_WEB)
        ("research_123", "memories", "web")
    """
    if subcategory:
        return (session_id, "memories", subcategory)
    return (session_id, "memories")


def make_namespace_prefix(session_id: str) -> Tuple[str, str]:
    """Create a namespace prefix for searching across all subcategories.

    Use this with store.asearch() to search across all memories
    regardless of subcategory.

    Args:
        session_id: Research session identifier

    Returns:
        Namespace prefix tuple

    Examples:
        >>> store.asearch(make_namespace_prefix("research_123"), query="...")
    """
    return (session_id, "memories")


def parse_namespace(namespace: Tuple[str, ...]) -> dict:
    """Parse a namespace tuple into components.

    Args:
        namespace: Namespace tuple from store

    Returns:
        Dict with session_id, prefix, and subcategory

    Examples:
        >>> parse_namespace(("research_123", "memories", "web"))
        {"session_id": "research_123", "prefix": "memories", "subcategory": "web"}
    """
    result = {
        "session_id": namespace[0] if len(namespace) > 0 else None,
        "prefix": namespace[1] if len(namespace) > 1 else None,
        "subcategory": namespace[2] if len(namespace) > 2 else None,
    }
    return result


__all__ = [
    # Constants
    "SUBCAT_DUMPS",
    "SUBCAT_WEB",
    "SUBCAT_INSIGHTS",
    # Functions
    "make_namespace",
    "make_namespace_prefix",
    "parse_namespace",
]
