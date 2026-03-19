"""Community scraping tool using Command API pattern (LangGraph V1).

This follows LangChain handoffs documentation:
- Uses @tool decorator instead of BaseTool class
- Returns Command(update={...}) to update state
- No in-place state mutations
- State flows naturally through the graph

Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""

import asyncio
from src.shared.logging_config import get_logger
from typing import Literal, Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

logger = get_logger(__name__)


class CommunityScrapeInput(BaseModel):
    """Input for community scraping."""
    platform: Literal["reddit", "4chan", "generic"] = Field(
        ...,
        description="Platform to scrape: 'reddit', '4chan', or 'generic'"
    )
    community: Optional[str] = Field(
        default=None,
        description="REQUIRED for Reddit. Subreddit name (e.g., 'r/politics' or 'politics')"
    )
    board: Optional[str] = Field(
        default=None,
        description="REQUIRED for 4chan. Board name (e.g., 'pol', 'v', 'g')"
    )
    url: Optional[str] = Field(
        default=None,
        description="REQUIRED for generic. Full URL of page to scrape"
    )
    query: str = Field(
        default="",
        description="Optional search query to filter posts (Reddit only)"
    )
    limit: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of posts/threads to retrieve (default: 30)"
    )


# Internal semaphore for rate limiting (shared across all tool calls)
_semaphore: Optional[asyncio.Semaphore] = None
_concurrency_limit: int = 10


def _get_semaphore():
    """Get or create the shared semaphore."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_concurrency_limit)
    return _semaphore


async def _execute_scrape_impl(
    platform: Literal["reddit", "4chan", "generic"],
    community: Optional[str],
    board: Optional[str],
    url: Optional[str],
    query: str,
    limit: int,
) -> str:
    """Execute the actual scrape call (protected by semaphore).

    This is the core scrape logic - extracted so it can be reused.

    Args:
        platform: Platform to scrape
        community: Subreddit for Reddit
        board: Board for 4chan
        url: URL for generic
        query: Search query for Reddit
        limit: Number of posts

    Returns:
        Formatted markdown with scraped content
    """
    # Import and call the logic layer
    from src.shared.tools.logic.web.community import community_scrape as logic_community_scrape

    logger.info(f"[COMMUNITY_SCRAPE] Starting: platform={platform}, community={community}, board={board}")

    result = await logic_community_scrape(
        platform=platform,
        community=community,
        board=board,
        url=url,
        query=query,
        limit=limit
    )

    logger.info(f"[COMMUNITY_SCRAPE] Complete: {len(result)} chars")

    return result


@tool
async def community_scrape(
    platform: Literal["reddit", "4chan", "generic"],
    community: Optional[str] = None,
    board: Optional[str] = None,
    url: Optional[str] = None,
    query: str = "",
    limit: int = 30,
    runtime: Optional[ToolRuntime] = None,
) -> Command:
    """Scrape authentic community discussions from online platforms.

    Use this tool to gather REAL USER SENTIMENT from online communities, not media takes.
    This provides authentic reactions from actual people discussing topics.

    REDDIT (platform="reddit"):
    - Set community="r/subreddit_name" (e.g., "r/politics", "r/PoliticalDiscussion")
    - Optionally set query="search terms" to filter posts
    - Best for: General political discussion, progressive/leftist sentiment

    4CHAN (platform="4chan"):
    - Set board="board_name" (e.g., "pol", "v", "g")
    - Best for: Raw, unfiltered, accelerationist/right-wing sentiment

    GENERIC (platform="generic"):
    - Set url="https://full-url"
    - Use for: Other forums, imageboards, or discussion platforms

    Args:
        platform: Platform to scrape (reddit, 4chan, generic)
        community: Subreddit name for Reddit (e.g., "r/politics" or "politics")
        board: Board name for 4chan (e.g., "pol", "v", "g")
        url: Full URL for generic scraping
        query: Optional search query to filter Reddit posts
        limit: Number of posts to retrieve (1-100, default: 30)
        runtime: ToolRuntime context for accessing state

    Returns:
        Command with ToolMessage (content)
    """
    semaphore = _get_semaphore()

    logger.info(
        f"[COMMUNITY_SCRAPE] Acquiring semaphore (concurrency_limit={_concurrency_limit})..."
    )

    # Acquire semaphore - this queues requests when limit is reached
    async with semaphore:
        logger.info(f"[COMMUNITY_SCRAPE] Semaphore acquired, executing scrape: platform={platform}")

        try:
            # Execute scrape
            result = await _execute_scrape_impl(
                platform=platform,
                community=community,
                board=board,
                url=url,
                query=query,
                limit=limit
            )

            logger.info(f"[COMMUNITY_SCRAPE] Adding {len(result)} chars to response")

            # Return Command with content
            # Note: community_scrape doesn't update raw_sources, so we only return messages
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

        except Exception as e:
            logger.error(f"[COMMUNITY_SCRAPE] Error: {e}", exc_info=True)
            error_msg = f"Community scraping failed: {str(e)}"

            if runtime is not None:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=runtime.tool_call_id
                            )
                        ]
                    }
                )

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=""
                        )
                    ]
                }
            )


# Export the tool function
__all__ = ["community_scrape"]
