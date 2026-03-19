"""MCP Web Client - Re-export from logic.web.custom_mcp.

This module is kept for backwards compatibility.
All functionality is now in src.shared.tools.logic.web.custom_mcp.

Usage (unchanged):
    from src.shared.tools.ulli_langchain.web.mcp_web_client import (
        MCPWebClient,
        mcp_web_search,
        mcp_web_scrape,
        mcp_web_fetch,
    )
"""

# Re-export everything from the canonical implementation
from src.shared.tools.logic.web.custom_mcp import (
    MCPWebClient,
    get_mcp_client,
    close_mcp_client,
    mcp_web_search,
    mcp_web_scrape,
    mcp_web_fetch,
    CUSTOM_MCP_BASE_URL,
)

# Aliases for backwards compatibility
MCP_WEB_SERVER_URL = CUSTOM_MCP_BASE_URL
MCP_WEB_SERVER_NAME = "web_tools"


__all__ = [
    "MCPWebClient",
    "get_mcp_client",
    "close_mcp_client",
    "mcp_web_search",
    "mcp_web_scrape",
    "mcp_web_fetch",
    "CUSTOM_MCP_BASE_URL",
    "MCP_WEB_SERVER_URL",
    "MCP_WEB_SERVER_NAME",
]
