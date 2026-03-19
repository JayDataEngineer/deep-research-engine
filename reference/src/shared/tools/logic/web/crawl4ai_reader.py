"""Two-tiered web scraping using crawl4ai Docker API.

Architecture:
- Tier 1: Stealth mode (enable_stealth=True) - fast, playwright-stealth
- Tier 2: Undetected browser (UndetectedAdapter) - slower but bypasses more detection

The crawl4ai Docker container runs as a REST API on port 11235.
"""

import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse
import time
import httpx

logger = logging.getLogger(__name__)

# Crawl4AI Docker API endpoint
CRAWL4AI_API_URL = "http://localhost:11235"

# Domain failure tracking
_DOMAIN_FAILURE_CACHE: dict = {}
_MAX_FAILURES_BEFORE_BLOCKLIST = 3
_FAILURE_EXPIRY_SECONDS = 3600  # 1 hour


def _get_domain(url: str) -> str:
    """Extract domain from URL for failure tracking."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"


def _should_skip_domain(domain: str) -> bool:
    """Check if domain has been permanently blocklisted."""
    if domain not in _DOMAIN_FAILURE_CACHE:
        return False

    entry = _DOMAIN_FAILURE_CACHE[domain]

    # Check if failures have expired
    if time.time() - entry.get("last_failure", 0) > _FAILURE_EXPIRY_SECONDS:
        del _DOMAIN_FAILURE_CACHE[domain]
        return False

    # If domain failed many times, skip it
    return entry.get("failures", 0) >= _MAX_FAILURES_BEFORE_BLOCKLIST


def _record_failure(domain: str, mode: str):
    """Record a scraping failure for a domain."""
    if domain not in _DOMAIN_FAILURE_CACHE:
        _DOMAIN_FAILURE_CACHE[domain] = {"failures": 0, "last_failure": 0, "mode": mode}

    _DOMAIN_FAILURE_CACHE[domain]["failures"] += 1
    _DOMAIN_FAILURE_CACHE[domain]["last_failure"] = time.time()
    _DOMAIN_FAILURE_CACHE[domain]["mode"] = mode

    failures = _DOMAIN_FAILURE_CACHE[domain]["failures"]
    logger.warning(f"[CRAWL4AI] Domain {domain} failure #{failures} (mode={mode})")


def _clear_failures(domain: str):
    """Clear failures for a domain (e.g., after successful scrape)."""
    if domain in _DOMAIN_FAILURE_CACHE:
        del _DOMAIN_FAILURE_CACHE[domain]


def _is_empty_result(content: str) -> bool:
    """Check if scraped content is empty or effectively empty."""
    if not content or len(content.strip()) < 50:
        return True
    # Check for error messages - more specific patterns
    error_indicators = [
        "access denied",
        "you don't have permission",
        "403 forbidden",
        "404 not found",
        "captcha",
        "please complete the security check",
        "enable javascript and cookies to continue",
        "you've been blocked",
        "cloudflare challenge",
    ]
    content_lower = content.lower()
    for indicator in error_indicators:
        if indicator in content_lower:
            return True
    return False


async def _scrape_with_mode(
    url: str,
    mode: str = "stealth",  # "stealth" or "undetected"
    timeout: float = 30.0,
) -> str:
    """Scrape URL using crawl4ai Docker API with specific mode.

    Args:
        url: URL to scrape
        mode: "stealth" for enable_stealth=True, "undetected" for UndetectedAdapter
        timeout: Timeout in seconds

    Returns:
        Scraped content as markdown string
    """
    domain = _get_domain(url)

    # Build request payload based on mode
    payload = {
        "urls": [url],  # API expects "urls" array
        "markdown": True,
        "waitFor": "networkidle",
        "screenshot": False,
    }

    # Set browser config based on mode
    if mode == "stealth":
        payload["browserConfig"] = {
            "enableStealth": True,
            "headless": True,
        }
    elif mode == "undetected":
        payload["browserConfig"] = {
            "headless": True,
        }
        payload["useUndetectedAdapter"] = True

    async with httpx.AsyncClient(timeout=timeout + 10.0) as client:
        response = await client.post(
            f"{CRAWL4AI_API_URL}/crawl",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            _record_failure(domain, mode)
            return f"[ERROR] Crawl4AI API returned {response.status_code}: {response.text[:200]}"

        data = response.json()

        # Check for success
        if not data.get("success", False):
            error = data.get("error", "Unknown error")
            _record_failure(domain, mode)
            return f"[FAILED] {error}"

        # Extract markdown content from results array
        results = data.get("results", [])
        if not results:
            _record_failure(domain, mode)
            return f"[FAILED] No results returned from {mode} mode"

        result = results[0]
        markdown_data = result.get("markdown", {})
        markdown = markdown_data.get("raw_markdown", "")

        if not markdown or _is_empty_result(markdown):
            _record_failure(domain, mode)
            return f"[FAILED] Empty or blocked content from {mode} mode"

        _clear_failures(domain)
        return markdown


async def scrape_with_crawl4ai(
    url: str,
    timeout: float = 30.0,
    user_agent: Optional[str] = None,
    headless: bool = True,
) -> str:
    """Scrape URL using two-tiered crawl4ai approach.

    Tier 1: Stealth mode (enable_stealth=True) - fast
    Tier 2: Undetected browser (UndetectedAdapter) - slower but more robust

    Args:
        url: URL to scrape
        timeout: Timeout in seconds
        user_agent: Optional custom user agent (not currently supported by API)
        headless: Run browser in headless mode (always true for API)

    Returns:
        Scraped content as markdown string
    """
    domain = _get_domain(url)

    # Check if domain is permanently blocklisted
    if _should_skip_domain(domain):
        return f"[BLOCKLISTED] Domain {domain} permanently blocklisted after {_MAX_FAILURES_BEFORE_BLOCKLIST} consecutive failures"

    start_time = time.time()

    # ============================================================================
    # TIER 1: Stealth Mode
    # ============================================================================
    logger.debug(f"[CRAWL4AI] Tier 1: Trying stealth mode for {url[:60]}...")

    try:
        result = await _scrape_with_mode(url, mode="stealth", timeout=timeout)
        if not result.startswith("[ERROR]") and not result.startswith("[FAILED]"):
            logger.info(f"[CRAWL4AI] Tier 1 SUCCESS: {url[:50]}... ({len(result)} chars in {time.time()-start_time:.1f}s)")
            return result
        else:
            logger.warning(f"[CRAWL4AI] Tier 1 failed: {result[:50]}, trying Tier 2")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[CRAWL4AI] Tier 1 exception: {e}, trying Tier 2")

    # ============================================================================
    # TIER 2: Undetected Browser
    # ============================================================================
    logger.debug(f"[CRAWL4AI] Tier 2: Trying undetected browser for {url[:60]}...")

    try:
        result = await _scrape_with_mode(url, mode="undetected", timeout=timeout)
        if not result.startswith("[ERROR]") and not result.startswith("[FAILED]"):
            logger.info(f"[CRAWL4AI] Tier 2 SUCCESS: {url[:50]}... ({len(result)} chars in {time.time()-start_time:.1f}s)")
            return result
        else:
            logger.error(f"[CRAWL4AI] Tier 2 failed: {result[:50]}")
            return result
    except Exception as e:
        logging.getLogger(__name__).error(f"[CRAWL4AI] Tier 2 exception: {e}")
        return f"[ERROR] Crawl4AI scraping failed: {str(e)}"


async def web_read_crawl4ai(
    url: str,
    timeout: float = 30.0,
    user_agent: Optional[str] = None,
    headless: bool = True,
) -> str:
    """Main entry point for crawl4ai-based web reading.

    Args:
        url: URL to scrape
        timeout: Timeout in seconds
        user_agent: Optional custom user agent
        headless: Run browser in headless mode

    Returns:
        Scraped content as markdown string
    """
    result = await scrape_with_crawl4ai(url, timeout, user_agent, headless)
    return result


def get_domain_failure_stats() -> dict:
    """Get statistics on domain failures for monitoring."""
    return {
        "total_blocked_domains": len([d for d, e in _DOMAIN_FAILURE_CACHE.items() if e.get("failures", 0) >= _MAX_FAILURES_BEFORE_BLOCKLIST]),
        "domains_with_failures": len(_DOMAIN_FAILURE_CACHE),
        "details": {d: e for d, e in _DOMAIN_FAILURE_CACHE.items() if e.get("failures", 0) >= 2},
    }


def clear_domain_cache():
    """Clear all domain failure tracking."""
    global _DOMAIN_FAILURE_CACHE
    _DOMAIN_FAILURE_CACHE.clear()
    logger.info("[CRAWL4AI] Domain failure cache cleared")
