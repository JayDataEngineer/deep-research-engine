"""Telegram channel scraping logic.

Supports two methods:
1. Web scraping via t.me public pages (no auth)
2. API via Telethon (requires credentials)

Usage:
    from .tools.logic.web.social_media.telegram import telegram_scrape

    results = await telegram_scrape(
        channel="durov",
        limit=50,
        method="web"
    )
"""

import asyncio
import html
import sys
from typing import Literal, Optional
from datetime import datetime

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
# WEB METHOD (t.me scraping)
# =============================================================================

async def _scrape_telegram_web(
    channel: str,
    limit: int = 50,
) -> str:
    """Scrape Telegram channel via t.me public web interface.

    Args:
        channel: Channel username (without @)
        limit: Number of messages to retrieve

    Returns:
        Formatted markdown with channel messages
    """
    prefix = _get_worker_prefix()
    print(f"{prefix}[telegram:web] Scraping channel @{channel} via t.me web interface", file=sys.stderr)

    # Build t.me URL
    url = f"https://t.me/s/{channel}"

    try:
        # Use web_read to fetch the page
        content = await web_read(url)

        # Parse the content for messages
        # t.me web view uses specific HTML structure
        output = [f"## Telegram: @{channel}\n"]
        output.append(f"**Source:** https://t.me/{channel}\n")
        output.append(f"**Method:** Web scraping (t.me)\n")

        # The web_read returns markdown, but we need to parse messages
        # Look for message patterns in the returned content
        lines = content.split('\n')
        messages_found = 0
        current_message = []

        for line in lines:
            # t.me web messages typically have specific patterns
            # This is a simplified parser - you may need to adjust based on actual output
            if line.strip() and messages_found < limit:
                # Skip header/footer lines
                if any(skip in line.lower() for skip in ['log in', 'sign up', 'download', 'Telegram', 'preview']):
                    continue
                if len(line.strip()) > 10:  # Only substantive content
                    current_message.append(line.strip())

                    # Check if this looks like a complete message
                    if len(line.strip()) < 200 or line.endswith(('.', '!', '?',)):
                        if current_message:
                            msg_text = '\n'.join(current_message)
                            output.append(f"\n### Message {messages_found + 1}")
                            output.append(msg_text)
                            messages_found += 1
                            current_message = []

        if messages_found == 0:
            # Fallback: return the raw content with truncation
            output.append(f"\n### Channel Content (first {limit} lines)\n")
            output.append(content[:5000] + "\n\n... (content truncated)")
            print(f"{prefix}[telegram:web] Parsed content, {len(content)} chars", file=sys.stderr)
        else:
            print(f"{prefix}[telegram:web] Found {messages_found} messages", file=sys.stderr)

        output.append(f"\n\n**Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return "\n".join(output)

    except Exception as e:
        print(f"{prefix}[telegram:web] Error: {e}", file=sys.stderr)
        return f"Telegram web scraping error: {str(e)}"


# =============================================================================
# API METHOD (Telethon)
# =============================================================================

async def _scrape_telegram_api(
    channel: str,
    limit: int = 50,
    api_id: Optional[str] = None,
    api_hash: Optional[str] = None,
) -> str:
    """Scrape Telegram channel via Telethon API.

    Args:
        channel: Channel username (without @)
        limit: Number of messages to retrieve
        api_id: Telegram API ID (from my.telegram.org)
        api_hash: Telegram API Hash (from my.telegram.org)

    Returns:
        Formatted markdown with channel messages
    """
    prefix = _get_worker_prefix()

    if not api_id or not api_hash:
        return "Error: API method requires TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables"

    print(f"{prefix}[telegram:api] Scraping channel @{channel} via Telethon API", file=sys.stderr)

    try:
        # Import Telethon
        from telethon import TelegramClient
        from telethon.errors import ChannelPrivateError, ChannelInvalidError

        # Create a temporary session file
        session_file = f"/tmp/telegram_scraper_{channel}.session"

        # Initialize client
        client = TelegramClient(session_file, int(api_id), api_hash)

        await client.start()

        output = [f"## Telegram: @{channel}\n"]
        output.append(f"**Source:** https://t.me/{channel}\n")
        output.append(f"**Method:** Telethon API\n")

        # Get channel entity
        try:
            entity = await client.get_entity(channel)
        except (ChannelPrivateError, ChannelInvalidError) as e:
            await client.disconnect()
            return f"Error: Cannot access channel @{channel}. It may be private or invalid."

        # Fetch messages
        messages = []
        async for message in client.iter_messages(entity, limit=limit):
            if message.message:
                messages.append({
                    'text': message.message,
                    'date': message.date,
                    'views': getattr(message, 'views', None),
                    'forwards': getattr(message, 'forwards', None),
                })

        # Format output
        for i, msg in enumerate(messages[:limit], 1):
            output.append(f"\n### Message {i}")
            output.append(f"**Date:** {msg['date'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            if msg['views']:
                output.append(f"**Views:** {msg['views']}")
            if msg['forwards']:
                output.append(f"**Forwards:** {msg['forwards']}")
            output.append(f"\n{html.unescape(msg['text'])}")

        output.append(f"\n\n**Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        output.append(f"**Total messages:** {len(messages)}")

        # Disconnect client
        await client.disconnect()

        # Clean up session file
        import os
        try:
            os.remove(session_file)
            os.remove(session_file + '-journal')
        except:
            pass

        print(f"{prefix}[telegram:api] Retrieved {len(messages)} messages", file=sys.stderr)
        return "\n".join(output)

    except ImportError:
        return "Error: Telethon not installed. Install with: uv add telethon"
    except Exception as e:
        print(f"{prefix}[telegram:api] Error: {e}", file=sys.stderr)
        return f"Telegram API error: {str(e)}"


# =============================================================================
# UNIFIED TELEGRAM SCRAPE FUNCTION
# =============================================================================

async def telegram_scrape(
    channel: str,
    limit: int = 50,
    search_query: Optional[str] = None,
    method: Literal["web", "api"] = "web",
    api_id: Optional[str] = None,
    api_hash: Optional[str] = None,
) -> str:
    """Scrape messages from a Telegram channel.

    **WHEN TO USE THIS TOOL:**
    - Use to gather authentic community sentiment from Telegram
    - Use for niche topics not well-covered on Reddit/4chan
    - Use for international/non-English perspectives
    - Use for real-time breaking news reactions

    **METHOD SELECTION:**

    Web Method (method="web", default):
    - No authentication required
    - Works on any public channel
    - May be rate limited
    - Best for: Quick scraping without setup

    API Method (method="api"):
    - Requires TELEGRAM_API_ID and TELEGRAM_API_HASH in .env
    - Get credentials from https://my.telegram.org
    - More reliable and faster
    - Can access private channels if you're a member
    - Best for: Production use, high-volume scraping

    Args:
        channel: Channel username (without @ symbol)
        limit: Number of messages to retrieve (default: 50, max: 200)
        search_query: Optional search to filter messages (API method only)
        method: "web" or "api" (default: "web")
        api_id: Telegram API ID (from environment, for API method)
        api_hash: Telegram API Hash (from environment, for API method)

    Returns:
        Formatted markdown with channel messages, dates, and metadata

    Examples:
        # Scrape Telegram's official channel
        await telegram_scrape("telegram", limit=30)

        # Scrape via API with credentials
        await telegram_scrape("durov", limit=50, method="api")
    """
    prefix = _get_worker_prefix()
    print(f"{prefix}[telegram] Channel=@{channel}, method={method}, limit={limit}", file=sys.stderr)

    # Normalize channel name
    channel = channel.lstrip('@')

    if method == "api":
        return await _scrape_telegram_api(
            channel=channel,
            limit=limit,
            api_id=api_id,
            api_hash=api_hash,
        )
    else:
        return await _scrape_telegram_web(
            channel=channel,
            limit=limit,
        )


__all__ = ["telegram_scrape"]
