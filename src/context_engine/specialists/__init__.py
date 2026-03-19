"""Specialists module - Web Search, Document processing.

This module provides specialist tools and subagents that can be used
by the Main Agent or Flash Explorers for specific tasks.

All tools use LiteLLM's MCP integration to access webtools-mcp server.
"""

__all__ = [
    # MCP Tools
    "create_mcp_web_search_tool",
    "create_mcp_scrape_tool",
    "create_mcp_crawl_tool",
    "create_all_mcp_tools",
    # Specialist Subagents
    "create_web_search_specialist",
    "create_document_specialist",
]


def __getattr__(name: str):
    """Lazy import for specialist modules."""
    if name in ("create_mcp_web_search_tool", "create_mcp_scrape_tool",
                "create_mcp_crawl_tool", "create_all_mcp_tools"):
        from context_engine.specialists.web_search import (
            create_mcp_crawl_tool,
            create_mcp_scrape_tool,
            create_mcp_web_search_tool,
            create_all_mcp_tools,
        )
        return locals()[name]

    if name == "create_web_search_specialist":
        from context_engine.specialists.web_search import create_web_search_specialist
        return create_web_search_specialist

    if name == "create_document_specialist":
        from context_engine.specialists.document import create_document_specialist
        return create_document_specialist

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
