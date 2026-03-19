"""Global concurrency control for parallel tool execution.

This module provides semaphores to prevent system overload when
executing multiple tools in parallel across all workers.

The DDoS Risk:
- 1 Worker calls web_search 3x in parallel = 30 browsers (if web_search spawns 10)
- 5 Workers running in parallel = 150 browsers!
- Solution: Global semaphore limits concurrent heavy operations
"""

import asyncio
from typing import Optional

# =============================================================================
# GLOBAL SEMAPHORES
# =============================================================================

# Heavy operations: scraping, browsers, etc.
# Limit to 10 concurrent across ALL workers and ALL parallel tool calls
HEAVY_OPERATION_SEMAPHORE = asyncio.Semaphore(10)

# Light operations: simple API calls, database lookups
# Can be higher limit as these are less resource-intensive
LIGHT_OPERATION_SEMAPHORE = asyncio.Semaphore(50)

# Web search specific (often spawns browsers)
WEB_SEARCH_SEMAPHORE = asyncio.Semaphore(8)

# Community scraping (Reddit, etc.)
COMMUNITY_SCRAPE_SEMAPHORE = asyncio.Semaphore(5)

# Web reader (fetching single URLs)
WEB_READER_SEMAPHORE = asyncio.Semaphore(15)


# =============================================================================
# TOOL CLASSIFICATION
# =============================================================================

HEAVY_TOOLS = {
    "web_search",          # Often spawns browsers
    "scrape_google_scholar",
    "scrape_arxiv",
}

LIGHT_TOOLS = {
    "list_collections",    # Database lookup
    "get_collection",      # Database lookup
    "vector_search",        # RAG query
}

WEB_SEARCH_TOOLS = {
    "web_search",
}

COMMUNITY_SCRAPE_TOOLS = {
    "community_scrape",
}

WEB_READER_TOOLS = {
    "web_reader",
    "web_get",
}


def get_semaphore_for_tool(tool_name: str) -> Optional[asyncio.Semaphore]:
    """Get the appropriate semaphore for a given tool.

    Args:
        tool_name: Name of the tool being called

    Returns:
        Semaphore to use, or None if no limit needed
    """
    tool_lower = tool_name.lower()

    if tool_lower in WEB_SEARCH_TOOLS:
        return WEB_SEARCH_SEMAPHORE
    elif tool_lower in COMMUNITY_SCRAPE_TOOLS:
        return COMMUNITY_SCRAPE_SEMAPHORE
    elif tool_lower in WEB_READER_TOOLS:
        return WEB_READER_SEMAPHORE
    elif tool_lower in HEAVY_TOOLS:
        return HEAVY_OPERATION_SEMAPHORE
    elif tool_lower in LIGHT_TOOLS:
        return LIGHT_OPERATION_SEMAPHORE
    else:
        # Default to light operation limit for unknown tools
        return LIGHT_OPERATION_SEMAPHORE


async def execute_with_semaphore(
    tool_func,
    tool_name: str,
    *args,
    **kwargs
):
    """Execute a tool function with appropriate semaphore limiting.

    Args:
        tool_func: The async tool function to execute
        tool_name: Name of the tool (for semaphore selection)
        *args: Positional args for the tool function
        **kwargs: Keyword args for the tool function

    Returns:
        Result of the tool function
    """
    semaphore = get_semaphore_for_tool(tool_name)

    if semaphore is None:
        # No limit - execute directly
        return await tool_func(*args, **kwargs)
    else:
        # Acquire semaphore before executing
        async with semaphore:
            return await tool_func(*args, **kwargs)
