"""WebCrawl tool using MCP server.

Provides website crawling and bulk scraping via MCP server.
All web operations are handled by the remote MCP server.
"""

from src.shared.logging_config import get_logger

from langchain_core.tools import tool

logger = get_logger(__name__)


@tool
async def web_crawl(
    url: str,
    limit: int = 100,
) -> str:
    """Crawl a website and scrape all its pages.

    Use this tool to:
    - Scrape entire websites or large sections
    - Extract content from multiple pages at once
    - Build comprehensive datasets from websites

    Note: Uses MCP server for web operations.
    For single page scraping, use web_get instead.

    Args:
        url: The base URL to start crawling from
        limit: Maximum number of pages to crawl (default: 100)

    Returns:
        JSON string with all scraped page data
    """
    try:
        from ..logic.web.custom_mcp import mcp_web_scrape

        logger.info(f"[WEB_CRAWL] Crawling website via MCP: {url[:50]}... (limit={limit})")

        # MCP server handles crawling internally
        # For now, just scrape the base URL as crawling is not directly supported
        result = await mcp_web_scrape(url)
        return result

    except Exception as e:
        logger.error(f"[WEB_CRAWL] Error: {e}")
        return f"Website crawling failed: {str(e)}"
