"""WebSearch tool using Command API pattern (LangGraph V1).

This is the NEW pattern following LangChain handoffs documentation:
- Uses @tool decorator instead of BaseTool class
- Returns Command(update={...}) to update state
- No in-place state mutations
- State flows naturally through the graph
- Uses MCP server for web search (replaces direct SearXNG calls)

Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""

import asyncio
from src.shared.logging_config import get_logger
import re
from typing import Literal, Optional, Type, Dict, Any

from pydantic import Field
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

# Import base class
from src.shared._base_models import BaseWebSearchInput

logger = get_logger(__name__)


class WebSearchInput(BaseWebSearchInput):
    """Input for web search with validation."""
    # Inherits all fields from BaseWebSearchInput:
    # - query, time_range, category, num_results, location, pages
    pass


# Internal semaphore for rate limiting (shared across all tool calls)
_search_semaphore: Optional[asyncio.Semaphore] = None
_concurrency_limit: int = 10


def _get_semaphore():
    """Get or create the shared semaphore."""
    global _search_semaphore
    if _search_semaphore is None:
        _search_semaphore = asyncio.Semaphore(_concurrency_limit)
    return _search_semaphore


async def _execute_search_impl(
    query: str,
    time_range: Literal["day", "week", "month", "year", "noLimit"],
    category: str,
    num_results: int,
    location: str,
    pages: int,
) -> str:
    """Execute web search via MCP server.

    Uses MCP search_web tool for web search.
    This replaces SearXNG with the MCP server via Tailscale Funnel.
    """
    import json
    from src.shared.tools.logic.web.custom_mcp import mcp_web_search

    logger.debug(f"[WEB_SEARCH] Starting MCP search: query='{query[:50]}...', pages={pages}")

    try:
        # MCP search returns JSON results
        raw_result = await mcp_web_search(
            query=query,
            num_results=num_results * pages,  # Multiply by pages for more results
        )

        # Parse the MCP response
        data = json.loads(raw_result)

        # Extract the results list from the response structure
        if isinstance(data, dict) and "results" in data:
            results_list = data["results"]
        elif isinstance(data, list):
            results_list = data
        else:
            # Unknown format, return as-is
            logger.warning(f"[WEB_SEARCH] Unexpected MCP response format: {type(data)}")
            return raw_result

        # Format results in the expected output format
        output_lines = []
        for r in results_list[:num_results]:
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("snippet", "")[:300]

            output_lines.append(f"**{title}**")
            output_lines.append(f"URL: {url}")
            if snippet:
                output_lines.append(f"{snippet}...")
            output_lines.append("")

        result = "\n".join(output_lines)
        result_length = len(result)
        logger.debug(f"[WEB_SEARCH] MCP complete: query='{query[:50]}...', returned {result_length} chars")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[WEB_SEARCH] Failed to parse MCP response: {e}")
        raise
    except Exception as e:
        logger.error(f"[WEB_SEARCH] MCP search failed: {e}")
        raise


async def _collect_sources_from_search(
    search_result: str,
    query: str,
    num_results: int,
    worker_id: str = "",
) -> Dict[str, Any]:
    """Collect sources from search results (does NOT mutate state).

    This extracts URLs from the formatted search output, scrapes their content,
    and returns a dict of sources ready to be added to raw_sources.

    Args:
        search_result: The formatted search result string
        query: The original search query
        num_results: Number of results returned
        worker_id: Worker identifier for tracking

    Returns:
        Dict of source_id -> source_data
    """
    # Import helper functions (Note: tier system removed - using Context Engine instead)
    from src.extensions.content.deep_research.storage.schema import (
        generate_source_id_short,
    )
    from src.shared.tools.logic.web.reader import web_read
    from datetime import datetime

    # Parse URLs from search result format: **Title**\nURL: <url>\n...
    url_pattern = re.compile(r'^URL:\s*(https?://[^\s]+)$', re.MULTILINE)
    title_pattern = re.compile(r'^\*\*([^*]+)\*\*$', re.MULTILINE)

    urls = url_pattern.findall(search_result)
    titles = title_pattern.findall(search_result)

    if not urls:
        logger.debug("[WEB_SEARCH] No URLs found in search results")
        return {}

    sources = {}
    logger.debug(f"[WEB_SEARCH] Collecting {len(urls)} sources from search")

    # Scrape content for each URL in parallel (limit concurrency)
    async def scrape_and_store(url: str, rank: int) -> Optional[tuple[str, Dict]]:
        try:
            # Check for duplicates
            source_id = generate_source_id_short(url)
            if source_id in sources:
                return None

            # Scrape content
            content = await web_read(url=url)

            # Extract title from results or use URL
            title = titles[rank] if rank < len(titles) else url

            # Calculate relevance score (decay based on rank)
            relevance_score = max(0.1, 1.0 - (rank * 0.05))

            # Create snippet
            snippet = content[:300] + "..." if len(content) > 300 else content

            # Create source data (Note: tier system removed)
            source_data = {
                "source_id": source_id,
                "url": url,
                "title": title,
                "content": content,
                "snippet": snippet,
                "summary": "",  # Will be populated by async summarizer
                "relevance_score": relevance_score,
                "worker_id": worker_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "query": query,
                    "rank": rank,
                    "total_sources": len(urls)
                }
            }

            logger.debug(
                f"[WEB_SEARCH] Collected source: {title[:50]}... "
                f"({len(content)} chars, relevance: {relevance_score:.2f})"
            )
            return (source_id, source_data)

        except Exception as e:
            logger.warning(f"[WEB_SEARCH] Failed to scrape {url}: {e}")
            return None

    # Limit concurrency for scraping
    semaphore = asyncio.Semaphore(5)
    async def scrape_with_sem(url, rank):
        async with semaphore:
            return await scrape_and_store(url, rank)

    # Run scraping in parallel
    tasks = [scrape_with_sem(url, rank) for rank, url in enumerate(urls)]
    results = await asyncio.gather(*tasks)

    # Collect successful results
    added_count = 0
    for result in results:
        if result is not None:
            source_id, source_data = result
            sources[source_id] = source_data
            added_count += 1

    logger.debug(f"[WEB_SEARCH] Collected {added_count}/{len(urls)} sources")
    return sources


@tool
async def web_search(
    query: str,
    time_range: Literal["day", "week", "month", "year", "noLimit"] = "noLimit",
    category: str = "general",
    num_results: int = 5,
    location: str = "us",
    pages: int = 10,
    # TODO: runtime should be required, not Optional - this is legacy code
    # Tools should only be called from LangGraph with proper runtime context
    # When called programmatically (e.g., from web_agent), runtime=None is passed
    # This is a design flaw that should be fixed by making web_agent a proper LangGraph node
    runtime: Optional[ToolRuntime] = None,
) -> Command:
    """Search the web for current information, news, facts, or research.

    Uses parallel pagination to fetch multiple pages (default 10) and
    FlashRank reranking to return the most relevant results.

    IMPORTANT: Use category='news' when the user asks for news, current events,
    headlines, or breaking stories.

    Args:
        query: The search query string
        time_range: Time filter (day, week, month, year, noLimit)
        category: SearXNG category (general, news, images, videos, files, map, social media, it, science)
        num_results: Number of top results (1-50)
        location: Location for region-specific results (cn or us)
        pages: Number of pages to fetch (1-20)
        runtime: ToolRuntime context for accessing state

    Returns:
        Command with ToolMessage (content) and state updates (raw_sources)
    """
    semaphore = _get_semaphore()

    logger.info(
        f"[WEB_SEARCH] Acquiring semaphore (concurrency_limit={_concurrency_limit})..."
    )

    # Acquire semaphore - this queues requests when limit is reached
    async with semaphore:
        logger.debug(f"[WEB_SEARCH] Semaphore acquired, executing search: query={query[:50]}...")

        # Execute search
        result = await _execute_search_impl(
            query=query,
            time_range=time_range,
            category=category,
            num_results=num_results,
            location=location,
            pages=pages,
        )

        # Collect sources (if runtime is available)
        new_sources = {}
        if runtime is not None:
            worker_id = runtime.state.get("current_worker", "")
            new_sources = await _collect_sources_from_search(
                search_result=result,
                query=query,
                num_results=num_results,
                worker_id=worker_id
            )

            # Merge new sources with existing raw_sources
            current_sources = runtime.state.get("raw_sources", {})
            merged_sources = {**current_sources, **new_sources}

            logger.debug(f"[WEB_SEARCH] Adding {len(new_sources)} sources to state")

            # Return Command with both content and state update
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=result,
                            tool_call_id=runtime.tool_call_id
                        )
                    ],
                    "raw_sources": merged_sources,  # State update
                }
            )

        # Fallback: no runtime available, return result as string
        # Note: This is expected when tools are called outside LangGraph runtime
        logger.debug("[WEB_SEARCH] No runtime context - returning result without state update")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=runtime.tool_call_id if runtime else ""
                    )
                ]
            }
        )


# Export the tool function
__all__ = ["web_search"]
