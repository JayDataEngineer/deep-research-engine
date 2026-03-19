"""Web Search Specialist - MCP Tools for web research.

Provides tools and subagent for web search and scraping through LiteLLM's
MCP integration with webtools-mcp server.

MCP Tools available:
- search_web: Web search with reranking
- scrape_url: Scrape single pages
- map_domain: Discover URLs from sitemaps
- crawl_site: Deep site crawling
- scrape_structured: Schema-based JSON extraction

Usage:
    from context_engine.specialists import create_mcp_web_search_tool

    tool = create_mcp_web_search_tool()
"""

from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger


# =============================================================================
# MCP TOOL DEFINITIONS
# =============================================================================

def create_mcp_web_search_tool() -> Any:
    """Create web search tool using LiteLLM's MCP integration.

    The search_web tool is provided by webtools-mcp server and accessed
    through LiteLLM's MCP tool routing.

    Returns:
        LangChain tool for web search
    """
    @tool
    async def search_web(query: str, num_results: int = 5) -> str:
        """Search the web for current information.

        Uses webtools-mcp server through LiteLLM for web search with reranking.

        Args:
            query: Search query string
            num_results: Number of results to return (default: 5)

        Returns:
            Search results with titles, snippets, and URLs

        Example:
            >>> results = await search_web("latest AI developments")
            >>> print(results)
            # Returns top 5 results with titles and URLs
        """
        # Tool definition for LLM - actual execution via LiteLLM MCP routing
        return f"[MCP Tool: search_web] Query: {query}, Results: {num_results}"

    return search_web


def create_mcp_scrape_tool() -> Any:
    """Create URL scraping tool using LiteLLM's MCP integration.

    Returns:
        LangChain tool for URL scraping
    """
    @tool
    async def scrape_url(url: str, extract_links: bool = False) -> str:
        """Scrape and extract content from a URL.

        Uses webtools-mcp server through LiteLLM.

        Args:
            url: URL to scrape
            extract_links: Whether to extract links from the page (default: False)

        Returns:
            Extracted content from the page, optionally with links
        """
        return f"[MCP Tool: scrape_url] URL: {url}, Extract links: {extract_links}"

    return scrape_url


def create_mcp_crawl_tool() -> Any:
    """Create site crawling tool using LiteLLM's MCP integration.

    Returns:
        LangChain tool for site crawling
    """
    @tool
    async def crawl_site(
        base_url: str,
        max_pages: int = 10,
        max_depth: int = 2,
    ) -> str:
        """Crawl a website and extract content from multiple pages.

        Uses webtools-mcp server through LiteLLM.

        Args:
            base_url: Starting URL for the crawl
            max_pages: Maximum number of pages to crawl (default: 10)
            max_depth: Maximum crawl depth from base URL (default: 2)

        Returns:
            Content from all crawled pages
        """
        return f"[MCP Tool: crawl_site] URL: {base_url}, Max pages: {max_pages}, Depth: {max_depth}"

    return crawl_site


def create_mcp_map_domain_tool() -> Any:
    """Create domain mapping tool using LiteLLM's MCP integration.

    Returns:
        LangChain tool for discovering URLs from sitemaps
    """
    @tool
    async def map_domain(domain: str) -> str:
        """Discover URLs from a domain's sitemap.

        Uses webtools-mcp server through LiteLLM.

        Args:
            domain: Domain to map (e.g., "example.com")

        Returns:
            List of discovered URLs from sitemap
        """
        return f"[MCP Tool: map_domain] Domain: {domain}"

    return map_domain


def create_mcp_scrape_structured_tool() -> Any:
    """Create structured data extraction tool using LiteLLM's MCP integration.

    Returns:
        LangChain tool for schema-based JSON extraction
    """
    @tool
    async def scrape_structured(url: str) -> str:
        """Extract structured data from a URL based on a schema.

        Uses webtools-mcp server through LiteLLM.

        Args:
            url: URL to extract data from

        Returns:
            Extracted data matching the schema
        """
        return f"[MCP Tool: scrape_structured] URL: {url}"

    return scrape_structured


def create_all_mcp_tools() -> list[Any]:
    """Create all MCP web tools.

    Returns:
        List of all MCP tool wrappers
    """
    return [
        create_mcp_web_search_tool(),
        create_mcp_scrape_tool(),
        create_mcp_crawl_tool(),
        create_mcp_map_domain_tool(),
        create_mcp_scrape_structured_tool(),
    ]


# =============================================================================
# WEB SEARCH SPECIALIST SUBAGENT
# =============================================================================

def create_web_search_specialist(
    store: BaseStore,
    session_id: str,
) -> dict[str, Any]:
    """Create web search specialist subagent definition.

    This specialist can be delegated to for comprehensive web research.
    Uses FAST model (Haiku) with all MCP web tools.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        Subagent definition dict for web search specialization
    """
    from context_engine.config import get_model
    from context_engine.store.tools import create_search_vectors_tool

    # Get FAST model (Haiku)
    model = get_model("fast")

    # Tools: context search + all MCP web tools
    tools = [
        create_search_vectors_tool(store, session_id),
        *create_all_mcp_tools(),
    ]

    return {
        "name": "web-search-specialist",
        "description": (
            "Comprehensive web research specialist. "
            "Can search the web (search_web), scrape URLs (scrape_url), "
            "crawl sites (crawl_site), and discover pages (map_domain). "
            "Give a research topic or question. Returns detailed findings with sources."
        ),
        "system_prompt": """You are a Web Search Specialist.

Your job is to conduct thorough web research on assigned topics.

## Available Tools

- **search_web(query)**: Search the web for current information
- **scrape_url(url)**: Extract content from a specific URL
- **crawl_site(base_url)**: Crawl multiple pages from a site
- **map_domain(domain)**: Discover URLs from a sitemap

## Guidelines

1. Start with search_web to find relevant sources
2. Use scrape_url to get detailed content from promising results
3. Use crawl_site for comprehensive coverage of a single source
4. Always cite your sources with URLs
5. Synthesize findings, don't just list them

## Output Format

```
## Summary
[2-3 sentence overview]

## Key Findings
- Finding 1 [source URL]
- Finding 2 [source URL]
- Finding 3 [source URL]

## Sources
1. Title: URL
2. Title: URL
```

Be thorough but efficient. Focus on quality over quantity.""",
        "model": model,
        "tools": tools,
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_quick_search_specialist(
    store: BaseStore,
    session_id: str,
) -> dict[str, Any]:
    """Create a quick search specialist for simple queries.

    Only has search_web tool, no scraping. Good for fast lookups.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        Subagent definition dict for quick search
    """
    from context_engine.config import get_model
    from context_engine.store.tools import create_search_vectors_tool

    model = get_model("fast")

    return {
        "name": "quick-search",
        "description": (
            "Quick web search for simple questions. "
            "Returns top search results with snippets. "
            "Use for fast lookups, not deep research."
        ),
        "system_prompt": """You are a Quick Search specialist.

Use search_web to find quick answers. Return concise results.

Format:
- Answer: [direct answer]
- Sources: [top 2-3 URLs]""",
        "model": model,
        "tools": [
            create_search_vectors_tool(store, session_id),
            create_mcp_web_search_tool(),
        ],
    }
