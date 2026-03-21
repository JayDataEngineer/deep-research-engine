"""Substack scraping logic via RSS feeds.

Supports:
- RSS feed parsing for article discovery
- Full article content fetching via web scraping
- Author-based subscriptions

Usage:
    from .tools.logic.web.social_media.substack import substack_scrape

    results = await substack_scrape(
        author="matttaibbi",
        limit=20,
        full_content=True
    )
"""

import asyncio
import html
import sys
from datetime import datetime
from typing import Optional

import httpx

from ..reader import web_read


# =============================================================================
# WORKER CONTEXT HELPER
# =============================================================================

def _get_worker_prefix() -> str:
    """Get the worker prefix for logging."""
    from .. import get_worker_context
    worker_id = get_worker_context()
    return f"[{worker_id}] " if worker_id else ""


# =============================================================================
# RSS PARSING
# =============================================================================

async def _fetch_substack_rss(author: str, limit: int = 20) -> list:
    """Fetch and parse Substack RSS feed.

    Args:
        author: Substack author name
        limit: Maximum number of articles to return

    Returns:
        List of article dictionaries with title, link, date, summary
    """
    prefix = _get_worker_prefix()
    rss_url = f"https://{author}.substack.com/feed"

    print(f"{prefix}[substack] Fetching RSS feed from {rss_url}", file=sys.stderr)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(rss_url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
            })
            response.raise_for_status()

        # Parse RSS XML
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.content)

        # RSS items are in <channel><item> structure
        items = []
        channel = root.find('.//channel')

        if channel is None:
            # Try Atom feed format
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry')[:limit]:
                title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                link_elem = entry.find('{http://www.w3.org/2005/Atom}link')
                date_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                content_elem = entry.find('{http://www.w3.org/2005/Atom}content')

                title = title_elem.text if title_elem is not None else "No title"
                link = link_elem.get('href') if link_elem is not None else ""
                date = date_elem.text if date_elem is not None else ""
                summary = content_elem.text if content_elem is not None else ""

                items.append({
                    'title': title,
                    'link': link,
                    'date': date,
                    'summary': summary,
                })
        else:
            # RSS format
            for item in channel.findall('item')[:limit]:
                title = item.find('title')
                link = item.find('link')
                date = item.find('pubDate')
                summary = item.find('description')

                items.append({
                    'title': title.text if title is not None else "No title",
                    'link': link.text if link is not None else "",
                    'date': date.text if date is not None else "",
                    'summary': summary.text if summary is not None else "",
                })

        print(f"{prefix}[substack] Parsed {len(items)} articles from RSS feed", file=sys.stderr)
        return items

    except Exception as e:
        print(f"{prefix}[substack] RSS fetch error: {e}", file=sys.stderr)
        return []


# =============================================================================
# FULL CONTENT FETCHING
# =============================================================================

async def _fetch_article_content(url: str, timeout: float = 15.0) -> str:
    """Fetch full article content via web_read.

    Args:
        url: Article URL
        timeout: Request timeout

    Returns:
        Article content as markdown
    """
    try:
        content = await web_read(url)

        # Limit content length
        if len(content) > 8000:
            content = content[:8000] + "\n\n... (content truncated)"

        return content

    except Exception as e:
        return f"Error fetching article: {str(e)}"


# =============================================================================
# UNIFIED SUBSTACK SCRAPE FUNCTION
# =============================================================================

async def substack_scrape(
    author: str,
    limit: int = 20,
    full_content: bool = False,
) -> str:
    """Scrape articles from a Substack author.

    **WHEN TO USE THIS TOOL:**
    - Use to gather long-form journalism and expert analysis
    - Use for independent media perspectives not found in mainstream outlets
    - Use when you need well-researched, thoughtful content on complex topics
    - Use to understand specific authors' ongoing coverage of issues

    **POPULAR SUBSTACK AUTHORS BY PERSPECTIVE:**

    Progressive/Left:
    - "matttaibbi" - Matt Taibbi (investigative journalism, political commentary)
    - "greenroom" - The Green Room (climate, energy politics)
    - "publicnotice" - Public Notice (legal/policy analysis)

    Conservative/Right:
    - "walterkirn" - Walter Kirn (cultural commentary, politics)
    - "bariweiss" - The Free Press (former NYT, contrarian views)

    Independent/Contrarian:
    - "balajis" - Balaji Srinivasan (tech, crypto, future of society)
    - "paulg" - Paul Graham (essays on startups, technology)

    Args:
        author: Substack author name (e.g., "matttaibbi")
        limit: Number of articles to retrieve (default: 20, max: 100)
        full_content: If True, fetch full article content via web scraping.
                     If False, use RSS summary only (faster, less detailed).

    Returns:
        Formatted markdown with article titles, dates, links, and content

    Examples:
        # Get article summaries (fast)
        await substack_scrape("matttaibbi", limit=10)

        # Get full article content (slower but more complete)
        await substack_scrape("matttaibbi", limit=5, full_content=True)
    """
    prefix = _get_worker_prefix()
    print(f"{prefix}[substack] Author={author}, limit={limit}, full_content={full_content}", file=sys.stderr)

    try:
        # Fetch RSS feed
        articles = await _fetch_substack_rss(author, limit)

        if not articles:
            return f"No articles found for Substack author '{author}'. Check the author name."

        # Build output
        output = [f"## Substack: @{author}\n"]
        output.append(f"**Feed:** https://{author}.substack.com/feed\n")
        output.append(f"**Articles:** {len(articles)}\n")

        for i, article in enumerate(articles[:limit], 1):
            title = html.unescape(article['title'])
            link = article['link']
            date_str = article['date']

            # Parse and format date
            try:
                if date_str:
                    # Try parsing RFC 2822 date
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(date_str)
                    date_formatted = dt.strftime('%Y-%m-%d')
                else:
                    date_formatted = "Unknown date"
            except:
                date_formatted = date_str[:20] if date_str else "Unknown date"

            output.append(f"\n### [{i}] {title}")
            output.append(f"**Published:** {date_formatted}")
            output.append(f"**Link:** {link}")

            if full_content:
                print(f"{prefix}[substack]   Fetching full content for: {title[:50]}...", file=sys.stderr)
                content = await _fetch_article_content(link)
                output.append(f"\n{content}")
            else:
                # Use RSS summary
                summary = article.get('summary', '')
                if summary:
                    # Strip HTML tags from summary
                    import re
                    summary = re.sub(r'<[^>]+>', '', summary)
                    summary = html.unescape(summary)

                    if len(summary) > 1000:
                        summary = summary[:1000] + "..."

                    output.append(f"\n{summary}")

            output.append("")  # Blank line between articles

        output.append(f"\n\n**Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        print(f"{prefix}[substack] COMPLETE: {len(articles[:limit])} articles from @{author}", file=sys.stderr)
        return "\n".join(output)

    except Exception as e:
        print(f"{prefix}[substack] Error: {e}", file=sys.stderr)
        return f"Substack scraping error: {str(e)}"


__all__ = ["substack_scrape"]
