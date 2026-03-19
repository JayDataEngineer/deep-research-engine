"""Social media scraping logic layer.

This module contains the logic implementations for social media scraping.
The LangChain layer calls these functions.

Available scrapers:
- telegram_scrape: Telegram channel scraping
- substack_scrape: Substack newsletter scraping
"""

from .telegram import telegram_scrape
from .substack import substack_scrape

__all__ = [
    "telegram_scrape",
    "substack_scrape",
]
