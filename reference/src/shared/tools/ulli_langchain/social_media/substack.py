"""Substack scraping tool for LangChain.

This tool provides access to Substack newsletters and articles via RSS feeds.

Supports:
- Author-based RSS feed scraping
- Article content extraction
- Favorite authors configuration

Usage:
    substack_scrape(
        author="matttaibbi",
        limit=20
    )
"""

import asyncio
from src.shared.logging_config import get_logger
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class SubstackScrapeInput(BaseModel):
    """Input for Substack scraping."""
    author: str = Field(
        ...,
        description="Substack author name (e.g., 'matttaibbi', 'greenroom')"
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of articles to retrieve (default: 20, max: 100)"
    )
    full_content: bool = Field(
        default=False,
        description="If True, fetch full article content. If False, RSS summary only."
    )


class SubstackScrapeTool(BaseTool):
    """Substack newsletter scraping tool via RSS feeds.

    Uses Substack's public RSS feeds to fetch article metadata and content.
    Can optionally fetch full article content via web scraping.
    """

    name: str = "substack_scrape"
    description: str = """Scrape articles from Substack newsletters to gather author perspectives.

Use this tool to get JOURNALIST and AUTHOR OPINIONS from Substack, not just social media chatter.

RECOMMENDED AUTHORS BY PERSPECTIVE:

Progressive/Left:
- "matttaibbi" - Matt Taibbi (investigative journalism)
- "greenroom" - The Green Room (climate politics)
- "publicnotice" - Public Notice (policy analysis)

Conservative/Right:
- "walterkirn" - Walter Kirn (cultural commentary)
- "andrewsorkin" - Andrew Ross Sorkin (financial/business)

Independent/Contrarian:
- "balajis" - Balaji Srinivasan (tech/cryptocurrency)
- "paulg" - Paul Graham (essays on tech/startups)

FEATURES:
- RSS feed parsing for article discovery
- Optional full content fetching via web scraping
- Publication dates and metadata
- Clean markdown output

Best for:
- Long-form journalism and analysis
- Expert opinion pieces
- Independent media perspectives
- Deep-dive research topics
"""
    args_schema: Type[BaseModel] = SubstackScrapeInput

    def _run(self, **kwargs):
        """Synchronous run - not supported."""
        raise NotImplementedError("Use async for Substack scraping")

    async def _arun(
        self,
        author: str,
        limit: int = 20,
        full_content: bool = False,
    ) -> str:
        """Async run.

        Args:
            author: Substack author name
            limit: Number of articles to retrieve
            full_content: Whether to fetch full article content

        Returns:
            Formatted markdown with articles
        """
        logger.info(f"[SUBSTACK_SCRAP] Scraping author '{author}', limit={limit}, full_content={full_content}")

        try:
            # Import and call the logic layer
            from src.shared.tools.logic.web.social_media.substack import substack_scrape as logic_substack_scrape

            result = await logic_substack_scrape(
                author=author,
                limit=limit,
                full_content=full_content,
            )

            logger.info(f"[SUBSTACK_SCRAP] Completed successfully, returning {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"[SUBSTACK_SCRAP] Error: {e}", exc_info=True)
            return f"Substack scraping failed: {str(e)}"


# =============================================================================
# SHARED INSTANCE
# =============================================================================

_shared_substack_scrape_tool: Optional[SubstackScrapeTool] = None


def get_substack_scrape_tool() -> SubstackScrapeTool:
    """Get or create the shared Substack scrape tool instance."""
    global _shared_substack_scrape_tool

    if _shared_substack_scrape_tool is None:
        _shared_substack_scrape_tool = SubstackScrapeTool()
        logger.info("[SUBSTACK_SCRAP] Created shared instance")

    return _shared_substack_scrape_tool


# Backwards compatibility - export as function
async def substack_scrape(
    author: str,
    limit: int = 20,
    full_content: bool = False,
) -> str:
    """Backwards-compatible function wrapper."""
    tool = get_substack_scrape_tool()
    return await tool._arun(
        author=author,
        limit=limit,
        full_content=full_content,
    )


# Add tool-like attributes for registry
substack_scrape.name = "substack_scrape"
substack_scrape.args_schema = SubstackScrapeInput
substack_scrape.description = """Scrape articles from Substack newsletters to gather author perspectives.

Use this tool to get JOURNALIST and AUTHOR OPINIONS from Substack, not just social media chatter.

RECOMMENDED AUTHORS BY PERSPECTIVE:

Progressive/Left:
- "matttaibbi" - Matt Taibbi (investigative journalism)
- "greenroom" - The Green Room (climate politics)
- "publicnotice" - Public Notice (policy analysis)

Conservative/Right:
- "walterkirn" - Walter Kirn (cultural commentary)
- "andrewsorkin" - Andrew Ross Sorkin (financial/business)

Independent/Contrarian:
- "balajis" - Balaji Srinivasan (tech/cryptocurrency)
- "paulg" - Paul Graham (essays on tech/startups)

FEATURES:
- RSS feed parsing for article discovery
- Optional full content fetching via web scraping
- Publication dates and metadata
- Clean markdown output

Best for:
- Long-form journalism and analysis
- Expert opinion pieces
- Independent media perspectives
- Deep-dive research topics
"""


__all__ = [
    "SubstackScrapeTool",
    "SubstackScrapeInput",
    "get_substack_scrape_tool",
    "substack_scrape",
]
