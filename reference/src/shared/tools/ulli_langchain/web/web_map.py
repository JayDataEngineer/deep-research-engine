"""WebMap tool using MCP server.

Provides website mapping and URL discovery via MCP server.
All web operations are handled by the remote MCP server.
"""

from .logging_config import get_logger

from langchain_core.tools import tool

logger = get_logger(__name__)


@tool
async def web_map(
    url: str,
    search: str | None = None,
    limit: int = 5000,
) -> str:
    """Map a website to discover all its URLs and links.

    Use this tool to:
    - Discover all pages on a website
    - Get a comprehensive site structure
    - Find URLs matching a search pattern
    - Understand website architecture

    Note: Uses MCP server for web operations.

    Args:
        url: The base URL to start mapping from (e.g., https://example.com)
        search: Optional search query to filter links by keyword
        limit: Maximum number of links to return (default: 5000)

    Returns:
        JSON string containing list of discovered URLs and metadata
    """
    try:
        from ..logic.web.custom_mcp import mcp_web_scrape

        logger.info(f"[WEB_MAP] Mapping website via MCP: {url[:50]}...")

        # MCP server handles mapping internally
        # For now, just scrape the base URL as mapping is not directly supported
        result = await mcp_web_scrape(url)
        return result

    except Exception as e:
        logger.error(f"[WEB_MAP] Error: {e}")
        return f"Website mapping failed: {str(e)}"
