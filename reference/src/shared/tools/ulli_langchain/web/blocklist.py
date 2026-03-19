"""Shared URL blocklist for web scraping.

This module provides a single source of truth for domains and URL patterns
that should be excluded from scraping. Used by:

1. MCP web search result filtering (before LLM selection)
2. MCP web scrape (before scraping)

Keeping this DRY prevents:
- Wasted LLM calls on known-bad URLs
- Wasted MCP scrape requests on domains that always fail
"""

from typing import Set, List, Tuple
from urllib.parse import urlparse


# =============================================================================
# BLOCKLISTED DOMAINS
# =============================================================================

# Domains that always return 500 errors or are not scrapable
BLOCKLIST_DOMAINS: Set[str] = {
    # Academic/Paywalled - require login, always fail
    "researchgate.net",
    "sciencedirect.com",
    "nature.com",
    "frontiersin.org",
    "tandfonline.com",
    "psycnet.apa.org",
    "link.springer.com",
    "emerald.com",
    "journals.sagepub.com",
    "search.proquest.com",
    "mdpi.com",
    "jstor.org",
    "ieee.org",
    "acm.org",
    "dl.acm.org",
    "scholar.google.com",  # Just links to other sites
    "pubs.aip.org",  # Academic publisher - paywall/blocking
    "link.aps.org",  # Academic publisher - paywall/blocking

    # Social media/App stores - anti-bot, always fail
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "linkedin.com",
    "lemon8-app.com",
    "alibaba.com",
    "pinterest.com",
    "reddit.com",  # Often rate limits
    "stackoverflow.com",  # Anti-bot protection, Crawl4AI fails
    "stackexchange.com",  # Same as stackoverflow
    "zhihu.com",  # Chinese Q&A - heavy anti-bot

    # PDF/Binary hosts - not HTML, will fail
    "pdfs.semanticscholar.org",
    "arxiv.org",  # PDFs, use abstract URL instead

    # Video sites - can't scrape video content
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv",

    # App stores
    "play.google.com",
    "apps.apple.com",

    # Known problematic sites
    "thefreedictionary.com",  # Returns minimal content
    "dictionary.cambridge.org",  # Just definitions
    "merriam-webster.com",
    "wikipedia.org",  # Often scraped but rarely useful for research
    "azure.microsoft.com",  # Anti-bot protection
    "math.stackexchange.com",  # Requires JS/login
    "phys.org",  # News site with anti-bot
}

# Path patterns that indicate non-scrapable content
BLOCKLIST_PATHS: List[str] = [
    "/downloadpdf",
    "/pdf/",
    "/uploads/",
    "/wp-content/uploads/",
    "/download/",
    "/file/",
    ".pdf",
    ".zip",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
]


def is_url_blocklisted(url: str) -> Tuple[bool, str]:
    """Check if URL should be skipped.

    Args:
        url: The URL to check

    Returns:
        (is_blocklisted, reason) - True if URL should be skipped
    """
    if not url:
        return False, ""

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Remove www. prefix for matching
        if domain.startswith("www."):
            domain = domain[4:]

        # Check domain blocklist
        for blocked in BLOCKLIST_DOMAINS:
            if blocked in domain or domain in blocked:
                return True, f"blocklisted_domain:{domain}"

        # Check path patterns
        for pattern in BLOCKLIST_PATHS:
            if pattern in path:
                return True, f"blocklisted_path:{pattern}"

        return False, ""

    except Exception:
        return False, ""


def filter_urls(urls: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Filter a list of URLs, removing blocklisted ones.

    Args:
        urls: List of URLs to filter

    Returns:
        (filtered_urls, removed) where removed is list of (url, reason)
    """
    filtered = []
    removed = []

    for url in urls:
        is_blocked, reason = is_url_blocklisted(url)
        if is_blocked:
            removed.append((url, reason))
        else:
            filtered.append(url)

    return filtered, removed


def filter_search_results(results: List[dict]) -> Tuple[List[dict], int]:
    """Filter search results, removing blocklisted URLs.

    Args:
        results: List of search result dicts with 'url' or 'link' key

    Returns:
        (filtered_results, removed_count)
    """
    filtered = []
    removed_count = 0

    for result in results:
        url = result.get("url") or result.get("link") or ""
        is_blocked, reason = is_url_blocklisted(url)

        if is_blocked:
            removed_count += 1
        else:
            filtered.append(result)

    return filtered, removed_count
