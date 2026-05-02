"""Orchestrator SDK bridge for the Deep Research Engine.

Wraps the orch-sdk Python package to provide async-compatible functions
that match DRE's existing tool interface. Routes web search, scrape,
and research calls through the orchestrator's /api/tools/exec endpoint.

The orchestrator handles MCP tool routing, sandbox management, and
provides access to 37+ tools (web research, media analysis, etc.)

Usage:
    from deep_research_engine.shared.orch_client import get_orch, orch_search, orch_scrape

    # Direct function calls (async-safe via thread pool)
    results = await orch_search("quantum computing advances")
    content = await orch_scrape("https://example.com")

    # Or use the client directly for more control
    orch = get_orch()
    result = orch.mcp_research("deep topic", max_results=5)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from loguru import logger

# Orchestrator SDK
from orch import OrchestratorClient, OrchestratorError, ToolResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORCH_BASE_URL = os.getenv("ORCH_BASE_URL", "http://localhost:3847")
ORCH_TIMEOUT = int(os.getenv("ORCH_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: OrchestratorClient | None = None


def get_orch() -> OrchestratorClient:
    """Get or create the global OrchestratorClient instance."""
    global _client
    if _client is None:
        _client = OrchestratorClient(base_url=ORCH_BASE_URL, timeout=ORCH_TIMEOUT)
        healthy = _client.is_healthy()
        if healthy:
            tools = _client.list_tools()
            logger.info(
                f"Orchestrator client connected ({ORCH_BASE_URL}), "
                f"{len(tools)} tools available"
            )
        else:
            logger.warning(f"Orchestrator not reachable at {ORCH_BASE_URL}")
    return _client


# ---------------------------------------------------------------------------
# Async wrappers (DRE is async, SDK is sync — bridge via to_thread)
# ---------------------------------------------------------------------------


async def orch_search(query: str, max_results: int = 5) -> str:
    """Search the web via orchestrator MCP.

    Args:
        query: Search query string.
        max_results: Number of results to return.

    Returns:
        JSON string with search results.
    """
    try:
        orch = get_orch()
        result = await asyncio.to_thread(
            orch.exec_tool, "search", {"query": query, "top_k": max_results}
        )
        if result.success:
            return result.text
        logger.warning(f"orch_search failed: {result.error}")
        return f"Search error: {result.error}"
    except Exception as e:
        logger.error(f"orch_search exception: {e}")
        return f"Search error: {e}"


async def orch_scrape(url: str) -> str:
    """Scrape a URL via orchestrator MCP.

    Args:
        url: URL to scrape.

    Returns:
        Markdown content of the page.
    """
    try:
        orch = get_orch()
        result = await asyncio.to_thread(
            orch.exec_tool, "scrape", {"url": url}
        )
        if result.success:
            return result.text
        logger.warning(f"orch_scrape failed for {url}: {result.error}")
        return f"Scrape error: {result.error}"
    except Exception as e:
        logger.error(f"orch_scrape exception: {e}")
        return f"Scrape error: {e}"


async def orch_research(query: str, max_results: int = 3) -> str:
    """Research a topic: search + scrape top results via orchestrator MCP.

    This is the most powerful tool — it searches, ranks, and scrapes
    the top results in one call. Use this for deep research queries.

    Args:
        query: Research query.
        max_results: Number of results to fully scrape.

    Returns:
        Combined research findings with page content.
    """
    try:
        orch = get_orch()
        result = await asyncio.to_thread(
            orch.exec_tool, "research",
            {"query": query, "max_results": max_results}
        )
        if result.success:
            return result.text
        logger.warning(f"orch_research failed: {result.error}")
        return f"Research error: {result.error}"
    except Exception as e:
        logger.error(f"orch_research exception: {e}")
        return f"Research error: {e}"


async def orch_crawl(url: str, max_depth: int = 2, max_pages: int = 20) -> str:
    """Deep crawl a site following links via orchestrator MCP.

    Args:
        url: Starting URL.
        max_depth: Maximum crawl depth.
        max_pages: Maximum pages to crawl.

    Returns:
        Combined crawled page content.
    """
    try:
        orch = get_orch()
        result = await asyncio.to_thread(
            orch.exec_tool, "crawl",
            {"url": url, "max_depth": max_depth, "max_pages": max_pages}
        )
        if result.success:
            return result.text
        logger.warning(f"orch_crawl failed: {result.error}")
        return f"Crawl error: {result.error}"
    except Exception as e:
        logger.error(f"orch_crawl exception: {e}")
        return f"Crawl error: {e}"


async def orch_extract(url: str, schema_type: str = "news") -> str:
    """Extract structured data from a page via orchestrator MCP.

    Args:
        url: URL to extract from.
        schema_type: Schema type (ecommerce, news, jobs, blog, social).

    Returns:
        Structured JSON data.
    """
    try:
        orch = get_orch()
        result = await asyncio.to_thread(
            orch.exec_tool, "extract",
            {"url": url, "schema_type": schema_type}
        )
        if result.success:
            return result.text
        logger.warning(f"orch_extract failed: {result.error}")
        return f"Extract error: {result.error}"
    except Exception as e:
        logger.error(f"orch_extract exception: {e}")
        return f"Extract error: {e}"


# ---------------------------------------------------------------------------
# Sandbox tools (for research that needs code execution)
# ---------------------------------------------------------------------------


async def orch_bash(sandbox_id: str, command: str, timeout: int = 60) -> str:
    """Execute a bash command in a sandbox via orchestrator.

    Args:
        sandbox_id: Sandbox container ID.
        command: Bash command to execute.
        timeout: Command timeout in seconds.

    Returns:
        Command output.
    """
    try:
        orch = get_orch()
        result = await asyncio.to_thread(
            orch.bash, sandbox_id, command, timeout
        )
        if result.success:
            return result.text
        logger.warning(f"orch_bash failed: {result.error}")
        return f"Bash error: {result.error}"
    except Exception as e:
        logger.error(f"orch_bash exception: {e}")
        return f"Bash error: {e}"


# ---------------------------------------------------------------------------
# Convenience: research with automatic search → scrape pipeline
# ---------------------------------------------------------------------------


async def orch_deep_research(query: str, max_results: int = 5) -> str:
    """Multi-step deep research using orchestrator tools.

    1. Research the query (search + scrape top results)
    2. Extract additional structured data from top URLs
    3. Combine into comprehensive findings

    Args:
        query: Research query.
        max_results: Number of sources to investigate.

    Returns:
        Combined research findings.
    """
    try:
        orch = get_orch()

        # Step 1: Primary research (search + scrape)
        logger.info(f"orch_deep_research: starting research for '{query}'")
        primary = await orch_research(query, max_results=min(max_results, 3))

        # Step 2: Extract URLs from primary results for additional scraping
        import re
        urls = re.findall(r'https?://[^\s<>"\]]+', primary)
        unique_urls = list(dict.fromkeys(urls))[:max_results]

        if len(unique_urls) > 1:
            # Scrape top unique URLs for more depth
            scrape_tasks = [orch_scrape(url) for url in unique_urls[1:4]]
            scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

            additional = []
            for i, r in enumerate(scrape_results):
                if isinstance(r, str) and not r.startswith("Scrape error"):
                    # Truncate each scrape to avoid context bloat
                    additional.append(f"### Source: {unique_urls[i+1]}\n\n{r[:3000]}")

            if additional:
                return f"{primary}\n\n## Additional Sources\n\n" + "\n\n".join(additional)

        return primary

    except Exception as e:
        logger.error(f"orch_deep_research exception: {e}")
        # Fallback to basic research
        return await orch_research(query, max_results)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Check if the orchestrator is reachable."""
    try:
        return get_orch().is_healthy()
    except Exception:
        return False


__all__ = [
    "get_orch",
    "is_available",
    "orch_search",
    "orch_scrape",
    "orch_research",
    "orch_crawl",
    "orch_extract",
    "orch_bash",
    "orch_deep_research",
    "ORCH_BASE_URL",
]
