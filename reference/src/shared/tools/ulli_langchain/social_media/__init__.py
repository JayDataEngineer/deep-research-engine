"""Social media scraping tools.

This module contains tools for scraping various social media platforms
to gather authentic community sentiment and user discussions.

Available tools:
- telegram_scrape: Telegram channel scraping
- substack_scrape: Substack newsletter scraping
- community_scrape: Reddit, 4chan, and generic forum scraping
"""

from .telegram import telegram_scrape, TelegramScrapeTool, TelegramScrapeInput
from .substack import substack_scrape, SubstackScrapeTool, SubstackScrapeInput

# Import community_scrape from the web module
from ..web.community_scrape import community_scrape

__all__ = [
    "telegram_scrape",
    "TelegramScrapeTool",
    "TelegramScrapeInput",
    "substack_scrape",
    "SubstackScrapeTool",
    "SubstackScrapeInput",
    "community_scrape",
]
