"""Community scraping - re-export from ulli_langchain/web.

This module re-exports the LangChain @tool decorated community_scrape
from ulli_langchain.web to avoid code duplication.

Usage:
    from logic.web import community_scrape
    content = await community_scrape(platform="reddit", community="r/politics", limit=50)
"""

from src.shared.tools.ulli_langchain.web.community_scrape import community_scrape

__all__ = ["community_scrape"]
