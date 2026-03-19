"""Telegram channel scraping tool for LangChain.

This tool provides access to Telegram channels for gathering authentic
community sentiment from Telegram groups and channels.

Supports:
- Channel message scraping (public channels via web, or via Telethon API)
- Favorite channels configuration
- Message filtering and search

Usage:
    telegram_scrape(
        channel="durov",
        limit=50,
        search_query="crypto"
    )
"""

import asyncio
from src.shared.logging_config import get_logger
import os
from typing import Literal, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class TelegramScrapeInput(BaseModel):
    """Input for Telegram channel scraping."""
    channel: str = Field(
        ...,
        description="Telegram channel username (without @) or invite link"
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Number of messages to retrieve (default: 50, max: 200)"
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Optional search query to filter messages"
    )
    method: Literal["web", "api"] = Field(
        default="web",
        description="Method: 'web' (via t.me, no auth) or 'api' (via Telethon, requires credentials)"
    )


class TelegramScrapeTool(BaseTool):
    """Telegram channel scraping tool.

    Supports two modes:
    1. Web mode - Scrapes public t.me pages (no auth required, rate limited)
    2. API mode - Uses Telethon client (requires TELEGRAM_API_ID and TELEGRAM_API_HASH)
    """

    name: str = "telegram_scrape"
    description: str = """Scrape messages from Telegram channels to gather authentic community sentiment.

Use this tool to get REAL USER DISCUSSIONS from Telegram channels, not media takes.

WEB METHOD (method="web", default):
- Set channel="channel_name" (e.g., "durov", "telegram")
- No authentication required
- Uses public t.me web interface
- Best for: Quick scraping of public channels without setup

API METHOD (method="api"):
- Requires TELEGRAM_API_ID and TELEGRAM_API_HASH in .env
- Get credentials from https://my.telegram.org
- More reliable and faster than web scraping
- Can access private channels if bot is member

Best for:
- Niche community sentiment (crypto, politics, etc.)
- Breaking news and real-time reactions
- International perspectives (non-English sources)
- Alternative social media platform data
"""
    args_schema: Type[BaseModel] = TelegramScrapeInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check for API credentials
        self.api_id = os.environ.get("TELEGRAM_API_ID")
        self.api_hash = os.environ.get("TELEGRAM_API_HASH")
        self._telethon_client = None

        if self.api_id and self.api_hash:
            logger.info("[TELEGRAM] API credentials found, API mode available")
        else:
            logger.info("[TELEGRAM] No API credentials, web mode only")

    def _run(self, **kwargs):
        """Synchronous run - not supported."""
        raise NotImplementedError("Use async for Telegram scraping")

    async def _arun(
        self,
        channel: str,
        limit: int = 50,
        search_query: Optional[str] = None,
        method: Literal["web", "api"] = "web",
    ) -> str:
        """Async run with rate limiting.

        Args:
            channel: Channel username (without @)
            limit: Number of messages to retrieve
            search_query: Optional search query
            method: "web" or "api"

        Returns:
            Formatted markdown with scraped content
        """
        logger.info(f"[TELEGRAM_SCRAP] Acquiring semaphore, method={method}...")

        try:
            # Import and call the logic layer
            from src.shared.tools.logic.web.social_media.telegram import telegram_scrape as logic_telegram_scrape

            result = await logic_telegram_scrape(
                channel=channel,
                limit=limit,
                search_query=search_query,
                method=method,
                api_id=self.api_id,
                api_hash=self.api_hash,
            )

            logger.info(f"[TELEGRAM_SCRAP] Completed successfully, returning {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"[TELEGRAM_SCRAP] Error: {e}", exc_info=True)
            return f"Telegram scraping failed: {str(e)}"


# =============================================================================
# SHARED INSTANCE
# =============================================================================

_shared_telegram_scrape_tool: Optional[TelegramScrapeTool] = None


def get_telegram_scrape_tool() -> TelegramScrapeTool:
    """Get or create the shared Telegram scrape tool instance."""
    global _shared_telegram_scrape_tool

    if _shared_telegram_scrape_tool is None:
        _shared_telegram_scrape_tool = TelegramScrapeTool()
        logger.info("[TELEGRAM_SCRAP] Created shared instance")

    return _shared_telegram_scrape_tool


# Backwards compatibility - export as function
async def telegram_scrape(
    channel: str,
    limit: int = 50,
    search_query: Optional[str] = None,
    method: Literal["web", "api"] = "web",
) -> str:
    """Backwards-compatible function wrapper."""
    tool = get_telegram_scrape_tool()
    return await tool._arun(
        channel=channel,
        limit=limit,
        search_query=search_query,
        method=method,
    )


# Add tool-like attributes for registry
telegram_scrape.name = "telegram_scrape"
telegram_scrape.args_schema = TelegramScrapeInput
telegram_scrape.description = """Scrape messages from Telegram channels to gather authentic community sentiment.

Use this tool to get REAL USER DISCUSSIONS from Telegram channels, not media takes.

WEB METHOD (method="web", default):
- Set channel="channel_name" (e.g., "durov", "telegram")
- No authentication required
- Uses public t.me web interface

API METHOD (method="api"):
- Requires TELEGRAM_API_ID and TELEGRAM_API_HASH in .env
- Get credentials from https://my.telegram.org
- More reliable and faster than web scraping

Best for:
- Niche community sentiment (crypto, politics, etc.)
- Breaking news and real-time reactions
- International perspectives (non-English sources)
"""


__all__ = [
    "TelegramScrapeTool",
    "TelegramScrapeInput",
    "get_telegram_scrape_tool",
    "telegram_scrape",
]
