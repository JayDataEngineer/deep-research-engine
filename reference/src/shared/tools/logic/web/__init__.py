"""
web_logic - MCP-only web search & scraping library.

All web operations go through the MCP server at 100.102.244.97:8000.
No local fallbacks - MCP or bust.

This module re-exports LangChain @tool decorated functions from
ulli_langchain.web to avoid code duplication.

Usage as library:
    from logic.web import web_search, web_read, community_scrape

    results = await web_search(query="AI news", num_results=5)
    content = await web_read(url="https://example.com")
    posts = await community_scrape(platform="reddit", community="r/politics", limit=50)

Direct MCP access:
    from logic.web import mcp_web_search, mcp_web_scrape

    # Direct access to MCP server
    results = await mcp_web_search(query="AI news")
    content = await mcp_web_scrape(url="https://example.com")

Worker context tracking (for multi-agent systems):
    from logic.web import set_worker_context, get_worker_context

    # Set worker ID before tool calls
    set_worker_context("researcher_0")
"""

import contextvars

# Re-export LangChain tools (single source of truth)
from .search import web_search
from .reader import web_read
from .community import community_scrape

# MCP client functions (low-level access)
from .custom_mcp import (
    mcp_web_search,
    mcp_web_fetch,
    mcp_web_scrape,
    get_mcp_client,
    close_mcp_client,
    CUSTOM_MCP_BASE_URL,
)

__version__ = "0.2.0"

# =============================================================================
# WORKER CONTEXT TRACKING
# =============================================================================

# Context variable for tracking the calling worker (for multi-agent systems)
_worker_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "worker_id",
    default=""
)


def set_worker_context(worker_id: str) -> None:
    """Set the current worker ID for context-aware logging.

    Use this in multi-agent systems to track which worker is calling tools.
    The worker ID will be included in log messages.

    Args:
        worker_id: Worker identifier (e.g., "researcher_0", "left_research")

    Example:
        set_worker_context("researcher_0")
        result = await web_search(query="AI news")  # Logs will include [researcher_0]
    """
    _worker_id_context.set(worker_id)


def get_worker_context() -> str:
    """Get the current worker ID from context.

    Returns:
        Current worker ID, or empty string if not set
    """
    return _worker_id_context.get("")


def clear_worker_context() -> None:
    """Clear the current worker ID from context."""
    _worker_id_context.set("")


__all__ = [
    # Web tools (LangChain @tool decorated)
    "web_search",
    "web_read",
    "community_scrape",
    # MCP tools (low-level client)
    "mcp_web_search",
    "mcp_web_fetch",
    "mcp_web_scrape",
    "get_mcp_client",
    "close_mcp_client",
    "CUSTOM_MCP_BASE_URL",
    # Worker context
    "set_worker_context",
    "get_worker_context",
    "clear_worker_context",
]
