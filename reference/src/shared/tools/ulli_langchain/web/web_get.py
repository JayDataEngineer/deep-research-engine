"""WebGet tool using Command API pattern (LangGraph V1).

This is the NEW pattern following LangChain handoffs documentation:
- Uses @tool decorator instead of BaseTool class
- Returns Command(update={...}) to update state
- No in-place state mutations
- State flows naturally through the graph

Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""

import asyncio
from urllib.parse import urlparse
from .logging_config import get_logger
from typing import Optional, Dict, Any

from pydantic import Field
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

# Import shared blocklist
from .blocklist import is_url_blocklisted

# Import base class
from ._base_models import BaseWebToolInput

logger = get_logger(__name__)


class WebGetInput(BaseWebToolInput):
    """Input for web page reading."""
    # Inherits 'url' field from BaseWebToolInput
    pass


# Internal semaphore for rate limiting (shared across all tool calls)
_semaphore: Optional[asyncio.Semaphore] = None
_concurrency_limit: int = 10


def _get_semaphore():
    """Get or create the shared semaphore."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_concurrency_limit)
    return _semaphore


async def _collect_source_from_get(url: str, content: str, worker_id: str = "") -> Dict[str, Any]:
    """Collect source data from web_get result (does NOT mutate state).

    Since web_get is a direct fetch (not a search), these sources
    are tracked with high relevance_score (1.0) for prioritization.

    Args:
        url: The URL that was fetched
        content: The scraped content
        worker_id: Worker identifier for tracking

    Returns:
        Dict with source_id -> source_data
    """
    from src.extensions.content.deep_research.storage.schema import (
        generate_source_id_short,
    )
    from datetime import datetime

    # Extract title from content or use URL
    title = url
    # Try to extract a title from the content (first line or markdown heading)
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            title = line.lstrip('#').strip()
            break
        elif line and not line.startswith('[') and len(line) < 100:
            title = line
            break

    # Create snippet
    snippet = content[:300] + "..." if len(content) > 300 else content

    # Generate source ID
    source_id = generate_source_id_short(url)

    # Create source data (Note: tier system removed - using Context Engine instead)
    source_data = {
        "source_id": source_id,
        "url": url,
        "title": title,
        "content": content,
        "snippet": snippet,
        "summary": "",  # Will be populated by async summarizer
        "relevance_score": 1.0,  # Explicitly requested
        "worker_id": worker_id,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "fetch_type": "direct",
        }
    }

    logger.debug(
        f"[WEB_GET] Collected source: {title[:50]}... "
        f"({len(content)} chars)"
    )

    return {source_id: source_data}


async def _execute_fetch_impl(url: str) -> str:
    """Execute the actual fetch call (protected by semaphore).

    This is the core fetch logic - extracted so it can be reused.

    Args:
        url: The URL to fetch

    Returns:
        Page content with metadata header
    """
    # Check blocklist FIRST to save resources
    is_blocked, block_reason = is_url_blocklisted(url)
    if is_blocked:
        logger.info(f"[WEB_GET] Skipped blocklisted URL: {url[:60]}... ({block_reason})")
        return f"Skipped: {block_reason}. This URL cannot be scraped."

    # CRITICAL FIX: Validate URL before attempting to fetch
    # Skip malformed URLs that look like IPv6 but aren't
    try:
        # Basic URL validation
        parsed = urlparse(url)

        # Check if URL has a valid scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"[WEB_GET] Invalid URL (missing scheme or netloc): {url[:50]}...")
            return f"Invalid URL: {url}. Please provide a valid http:// or https:// URL."

        # Check for malformed IPv6-style URLs that aren't actually IPv6
        if '[' in url and '](' not in url:
            logger.warning(f"[WEB_GET] Malformed IPv6-style URL detected: {url[:50]}...")
            return f"Malformed URL detected. Skipping this URL: {url[:50]}..."

    except Exception as validation_error:
        logger.warning(f"[WEB_GET] URL validation failed: {validation_error}, url: {url[:50]}...")
        return f"URL validation failed for: {url[:50]}..."

    # Import shared lang_tools
    from logic.web import web_read as lang_tool_read

    logger.info(f"[WEB_GET] Starting: url='{url[:60]}...'")

    # Fetch content
    result = await lang_tool_read(url=url)

    # Add metadata header
    output = [f"Content from: {url}\n"]
    output.append(result)

    final_result = "\n".join(output)
    logger.info(f"[WEB_GET] Complete: url='{url[:60]}...', {len(final_result)} chars")

    return final_result


@tool
async def web_get(
    url: str,
    runtime: Optional[ToolRuntime] = None,
) -> Command:
    """Fetch and read web pages, extracting main content.

    Use for reading articles, documentation, or extracting data from web pages.
    Returns page content in markdown format by default.

    Args:
        url: The URL of the web page to fetch. Must be a valid http:// or https:// URL.
        runtime: ToolRuntime context for accessing state

    Returns:
        Command with ToolMessage (content) and state updates (raw_sources)
    """
    semaphore = _get_semaphore()

    logger.info(
        f"[WEB_GET] Acquiring semaphore (concurrency_limit={_concurrency_limit})..."
    )

    # Acquire semaphore - this queues requests when limit is reached
    async with semaphore:
        logger.info(f"[WEB_GET] Semaphore acquired, executing fetch: url={url[:50]}...")

        # Execute fetch
        result = await _execute_fetch_impl(url)

        # Collect source (if runtime is available)
        new_sources = {}
        if runtime is not None:
            worker_id = runtime.state.get("current_worker", "")
            # Check for duplicate
            current_sources = runtime.state.get("raw_sources", {})
            from src.extensions.content.deep_research.storage.schema import (
                generate_source_id_short,
            )
            source_id = generate_source_id_short(url)

            if source_id not in current_sources:
                # Extract content from result (remove the header line)
                content_lines = result.split('\n')
                if content_lines and content_lines[0].startswith("Content from:"):
                    content = '\n'.join(content_lines[1:])
                else:
                    content = result

                # Collect source data
                new_sources = await _collect_source_from_get(
                    url=url,
                    content=content,
                    worker_id=worker_id
                )

            # Merge new sources with existing raw_sources
            merged_sources = {**current_sources, **new_sources}

            if new_sources:
                logger.info(f"[WEB_GET] Adding {len(new_sources)} source to state")

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
        logger.debug("[WEB_GET] No runtime context - returning result without state update")
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
__all__ = ["web_get"]
