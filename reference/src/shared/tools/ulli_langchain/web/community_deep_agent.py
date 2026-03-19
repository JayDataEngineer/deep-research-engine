"""Community Deep Agent - Autonomous community sentiment researcher.

This agent follows the "Draft then Extract" pattern:
1. Agent runs freely with community_scrape tool
2. Returns raw markdown findings
3. Caller uses extract_structured_output() for validated data

The agent is designed to:
- Understand which platform to use based on the query
- Make multiple calls if needed to gather comprehensive sentiment
- Return raw markdown for extraction layer

Usage:
    from src.shared.tools.ulli_langchain.web.community_deep_agent import (
        community_deep_agent,
        extract_community_sentiment
    )

    # Get raw findings
    result = await community_deep_agent.ainvoke({
        "query": "What does Reddit think about gym motivation?",
        "platform_hint": "reddit",
        "limit": 30
    })

    # Extract structured data
    posts = await extract_community_sentiment(
        raw_markdown=result,
        yaml_config=config
    )

Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""

import asyncio
from src.shared.logging_config import get_logger
import time
from typing import Literal, Optional, Dict, Any, List, Type

from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

# Import base class
from src.shared._base_models import BaseWebAgentInput

logger = get_logger(__name__)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class CommunityDeepAgentInput(BaseWebAgentInput):
    """Input for community deep agent."""
    # Inherits: query, time_range, category, depth, location from BaseWebAgentInput
    platform_hint: Optional[Literal["reddit", "4chan", "hackernews", "lemmy", "auto"]] = Field(
        default="auto",
        description="Platform hint. 'auto' lets the agent decide based on query context."
    )
    community: Optional[str] = Field(
        default=None,
        description="Specific community/subreddit (e.g., 'r/technology', 'technology@lemmy.world')"
    )
    limit: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of posts to retrieve per platform"
    )
    max_platforms: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum platforms to query (default: 2)"
    )


class CommunityDeepAgentOutput(BaseModel):
    """Output from community deep agent."""
    query: str
    platforms_queried: List[str]
    raw_findings: str
    total_posts: int
    elapsed_seconds: float


# =============================================================================
# INTERNAL SEMAPHORE
# =============================================================================

_agent_semaphore: Optional[asyncio.Semaphore] = None
_concurrency_limit: int = 5  # Community scraping is expensive


def _get_semaphore():
    """Get or create the shared semaphore."""
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(_concurrency_limit)
    return _agent_semaphore


# =============================================================================
# PLATFORM SELECTION LOGIC
# =============================================================================

def _infer_platform(query: str) -> str:
    """Infer the best platform based on query content.

    Args:
        query: The research query

    Returns:
        Platform name string
    """
    query_lower = query.lower()

    # Tech/startup keywords -> Hacker News
    tech_keywords = ["startup", "programming", "software", "developer", "code", "api",
                     "silicon valley", "tech industry", "y combinator", "vc", "founder"]
    if any(kw in query_lower for kw in tech_keywords):
        return "hackernews"

    # Gaming keywords -> Reddit or 4chan
    gaming_keywords = ["game", "gaming", "video game", "console", "pc gaming", "esports"]
    if any(kw in query_lower for kw in gaming_keywords):
        return "reddit"  # Reddit has better gaming communities

    # Political/social keywords
    political_keywords = ["politics", "election", "democrat", "republican", "trump",
                          "biden", "congress", "policy", "government"]
    if any(kw in query_lower for kw in political_keywords):
        return "reddit"  # Reddit for general political discussion

    # Privacy/foss keywords -> Lemmy
    foss_keywords = ["privacy", "foss", "open source", "linux", "decentralized",
                     "federated", "self-hosted", "matrix", "mastodon"]
    if any(kw in query_lower for kw in foss_keywords):
        return "lemmy"

    # Fitness/lifestyle -> Reddit
    lifestyle_keywords = ["gym", "fitness", "workout", "diet", "health", "motivation",
                          "self-improvement", "dating", "relationship"]
    if any(kw in query_lower for kw in lifestyle_keywords):
        return "reddit"

    # Default to Reddit (most general)
    return "reddit"


def _infer_community(query: str, platform: str) -> Optional[str]:
    """Infer the best community based on query content.

    Args:
        query: The research query
        platform: Target platform

    Returns:
        Community name or None
    """
    query_lower = query.lower()

    if platform == "reddit":
        # Map keywords to subreddits
        subreddit_map = {
            "gym": "fitness",
            "fitness": "fitness",
            "workout": "fitness",
            "gaming": "gaming",
            "video game": "gaming",
            "politics": "politics",
            "election": "politics",
            "technology": "technology",
            "tech": "technology",
            "programming": "programming",
            "coding": "programming",
            "startup": "startups",
            "finance": "personalfinance",
            "money": "personalfinance",
            "investing": "investing",
            "crypto": "cryptocurrency",
            "bitcoin": "bitcoin",
            "motivation": "getmotivated",
            "self-improvement": "selfimprovement",
            "dating": "dating",
            "relationship": "relationships",
        }
        for keyword, subreddit in subreddit_map.items():
            if keyword in query_lower:
                return f"r/{subreddit}"
        return None

    elif platform == "lemmy":
        # Map keywords to Lemmy communities
        lemmy_map = {
            "privacy": "privacy@lemmy.world",
            "linux": "linux@lemmy.ml",
            "technology": "technology@lemmy.world",
            "gaming": "gaming@lemmy.ml",
            "opensource": "opensource@lemmy.ml",
        }
        for keyword, community in lemmy_map.items():
            if keyword in query_lower:
                return community
        return None

    return None


# =============================================================================
# CORE RESEARCH FUNCTION
# =============================================================================

async def _execute_community_research(
    query: str,
    platform_hint: str = "auto",
    community: Optional[str] = None,
    limit: int = 30,
    max_platforms: int = 2,
) -> Dict[str, Any]:
    """Execute community research across one or more platforms.

    Args:
        query: Research query
        platform_hint: Platform hint or "auto"
        community: Specific community to target
        limit: Posts per platform
        max_platforms: Maximum platforms to query

    Returns:
        Dict with raw_findings, platforms_queried, total_posts
    """
    from src.shared.tools.logic.web.community import community_scrape

    start_time = time.time()
    findings = []
    platforms_queried = []
    total_posts = 0

    # Determine which platforms to use
    if platform_hint == "auto":
        primary_platform = _infer_platform(query)
        # Add a secondary platform for broader coverage
        secondary_map = {
            "reddit": "hackernews",
            "hackernews": "reddit",
            "lemmy": "hackernews",
            "4chan": "reddit",
        }
        platforms = [primary_platform]
        if max_platforms > 1 and primary_platform in secondary_map:
            platforms.append(secondary_map[primary_platform])
    else:
        platforms = [platform_hint]

    # Limit to max_platforms
    platforms = platforms[:max_platforms]

    logger.info(f"[COMMUNITY_DEEP_AGENT] Querying platforms: {platforms}")

    # Query each platform
    for platform in platforms:
        try:
            # Infer community if not specified
            target_community = community or _infer_community(query, platform)

            logger.info(
                f"[COMMUNITY_DEEP_AGENT] Scraping {platform} "
                f"(community={target_community}, limit={limit})"
            )

            # First try with search query
            use_query = query if platform in ["reddit", "hackernews", "lemmy"] else ""
            result = await community_scrape(
                platform=platform,
                community=target_community,
                query=use_query,
                limit=limit
            )

            # Check for errors
            if result.startswith("Error:") or result.startswith("No "):
                logger.warning(f"[COMMUNITY_DEEP_AGENT] {platform} returned: {result[:100]}")
                continue

            findings.append(f"## Platform: {platform.upper()}\n\n{result}")
            platforms_queried.append(platform)

            # Rough estimate of posts
            total_posts += result.count("### [")

        except Exception as e:
            logger.error(f"[COMMUNITY_DEEP_AGENT] Failed to scrape {platform}: {e}")
            continue

    elapsed = time.time() - start_time

    # Combine all findings
    raw_findings = "\n\n---\n\n".join(findings) if findings else "No community data found."

    logger.info(
        f"[COMMUNITY_DEEP_AGENT] Complete: {len(platforms_queried)} platforms, "
        f"~{total_posts} posts, {elapsed:.2f}s"
    )

    return {
        "raw_findings": raw_findings,
        "platforms_queried": platforms_queried,
        "total_posts": total_posts,
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# TOOL DEFINITION
# =============================================================================

@tool
async def community_deep_agent(
    query: str,
    platform_hint: Optional[Literal["reddit", "4chan", "hackernews", "lemmy", "auto"]] = "auto",
    community: Optional[str] = None,
    limit: int = 30,
    max_platforms: int = 2,
    runtime: Optional[ToolRuntime] = None,
) -> Command:
    """Research community sentiment across multiple platforms.

    This agent autonomously:
    1. Analyzes your query to determine the best platform(s)
    2. Scrapes relevant communities for posts
    3. Returns raw markdown with posts from all platforms

    Use this to understand REAL USER SENTIMENT from online communities.
    This provides authentic reactions from actual people, not media takes.

    PLATFORM SELECTION (auto mode):
    - Tech/startup topics → Hacker News + Reddit
    - Gaming topics → Reddit (r/gaming)
    - Privacy/FOSS topics → Lemmy + Hacker News
    - Lifestyle/fitness → Reddit (r/fitness, etc.)
    - Political topics → Reddit (r/politics)

    PLATFORM HINTS:
    - "reddit": General discussions, most topics
    - "hackernews": Tech industry, startups, programming
    - "lemmy": Privacy-focused, federated, FOSS
    - "4chan": Raw, unfiltered (use carefully)
    - "auto": Let agent decide (recommended)

    Args:
        query: Research question about community sentiment
        platform_hint: Which platform to prioritize (default: auto)
        community: Specific community (e.g., "r/technology", "technology@lemmy.world")
        limit: Posts to retrieve per platform (default: 30)
        max_platforms: Maximum platforms to query (default: 2)
        runtime: ToolRuntime context

    Returns:
        Command with ToolMessage containing raw markdown findings
    """
    semaphore = _get_semaphore()

    logger.info(f"[COMMUNITY_DEEP_AGENT] Starting: query={query[:50]}..., platform_hint={platform_hint}")

    async with semaphore:
        try:
            result = await _execute_community_research(
                query=query,
                platform_hint=platform_hint or "auto",
                community=community,
                limit=limit,
                max_platforms=max_platforms,
            )

            output = f"""# Community Research: {query}

**Platforms:** {', '.join(result['platforms_queried'])}
**Posts Found:** ~{result['total_posts']}
**Time:** {result['elapsed_seconds']:.2f}s

---

{result['raw_findings']}
"""

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=output,
                            tool_call_id=runtime.tool_call_id if runtime else ""
                        )
                    ]
                }
            )

        except Exception as e:
            logger.error(f"[COMMUNITY_DEEP_AGENT] Error: {e}", exc_info=True)
            error_msg = f"Community research failed: {str(e)}"

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=runtime.tool_call_id if runtime else ""
                        )
                    ]
                }
            )


# =============================================================================
# STRUCTURED EXTRACTION HELPER
# =============================================================================

async def extract_community_sentiment(
    raw_markdown: str,
    query: str = "",
    platforms: Optional[List[str]] = None,
    llm: Optional[Any] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Extract structured CommunityPosts from raw agent output.

    This is the "Extract" part of "Draft then Extract":
    1. community_deep_agent returns raw markdown
    2. This function extracts validated structured data using LLM

    Args:
        raw_markdown: Raw markdown output from community_deep_agent
        query: Original query (for metadata)
        platforms: List of platforms queried (for metadata)
        llm: Optional LLM instance
        yaml_config: Config for creating LLM

    Returns:
        CommunityPostsModel with list of structured posts
    """
    from src.shared.tools.logic.web.community import (
        extract_community_posts,
        get_community_posts_model
    )

    # Determine primary platform from content
    platform = "unknown"
    if platforms:
        platform = platforms[0] if len(platforms) == 1 else "multi"
    elif "REDDIT" in raw_markdown.upper():
        platform = "reddit"
    elif "HACKER NEWS" in raw_markdown.upper():
        platform = "hackernews"
    elif "LEMMY" in raw_markdown.upper():
        platform = "lemmy"
    elif "4CHAN" in raw_markdown.upper():
        platform = "4chan"

    return await extract_community_posts(
        raw_markdown=raw_markdown,
        platform=platform,
        query=query,
        llm=llm,
        yaml_config=yaml_config,
    )


# =============================================================================
# SIMPLE RESEARCH FUNCTION (for direct use without agent)
# =============================================================================

async def simple_community_research(
    query: str,
    platform: str = "auto",
    community: Optional[str] = None,
    limit: int = 30,
) -> tuple[str, Dict[str, Any]]:
    """Simple community research function for direct use.

    Use this when you don't need the agent overhead - just want
    to quickly grab community posts.

    Args:
        query: Research query
        platform: Platform hint ("auto" for inference)
        community: Specific community
        limit: Posts to retrieve

    Returns:
        Tuple of (raw_findings, metadata_dict)
    """
    result = await _execute_community_research(
        query=query,
        platform_hint=platform,
        community=community,
        limit=limit,
        max_platforms=1,
    )

    return result["raw_findings"], {
        "platforms": result["platforms_queried"],
        "total_posts": result["total_posts"],
        "elapsed": result["elapsed_seconds"],
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "community_deep_agent",
    "extract_community_sentiment",
    "simple_community_research",
    "CommunityDeepAgentInput",
    "CommunityDeepAgentOutput",
]
