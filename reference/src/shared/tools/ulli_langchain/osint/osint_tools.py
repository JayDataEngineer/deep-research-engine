"""OSINT tools for open-source intelligence investigations.

Tools for:
- Username search across platforms
- Email investigation
- Domain WHOIS lookup
- Breach directory lookup
- Google dork generation
"""

from src.shared.logging_config import get_logger
import os
from typing import Optional, List

from langchain_core.tools import tool

logger = get_logger(__name__)

# =============================================================================
# ENV VAR CACHE (performance optimization)
# =============================================================================

_RAPIDAPI_KEY: Optional[str] = os.environ.get("RAPIDAPI_KEY")

# =============================================================================
# PLATFORM URLS FOR USERNAME SEARCH
# =============================================================================

PLATFORM_URLS = {
    "github": "https://github.com/{}",
    "reddit": "https://www.reddit.com/user/{}",
    "twitter": "https://twitter.com/{}",
    "instagram": "https://www.instagram.com/{}",
    "tiktok": "https://www.tiktok.com/@{}",
    "linkedin": "https://www.linkedin.com/in/{}",
    "youtube": "https://www.youtube.com/@{}",
    "facebook": "https://www.facebook.com/{}",
    "hackernews": "https://news.ycombinator.com/user?id={}",
}


@tool
async def osint_username_search(
    username: str,
    platforms: Optional[List[str]] = None,
) -> str:
    """Search for a username across multiple social platforms.

    Use this tool to:
    - Find user profiles across GitHub, Reddit, Twitter, Instagram, TikTok, LinkedIn, YouTube, Facebook, HackerNews
    - Investigate digital footprints
    - Cross-reference username presence across platforms

    Args:
        username: Username to search for (without @ symbol)
        platforms: Optional list of specific platforms to check. Default is all platforms.
                   Available: github, reddit, twitter, instagram, tiktok, linkedin, youtube, facebook, hackernews

    Returns:
        JSON with found profile URLs and generated Google dork queries
    """
    import json

    if platforms is None:
        platforms = list(PLATFORM_URLS.keys())

    results = {
        "username": username,
        "profiles": {},
        "dorks": []
    }

    # Generate direct URLs for requested platforms
    for platform in platforms:
        if platform in PLATFORM_URLS:
            url = PLATFORM_URLS[platform].format(username)
            results["profiles"][platform] = url

    # Generate Google dorks for additional investigation
    dork_queries = [
        f'"{username}" site:github.com',
        f'"{username}" site:reddit.com',
        f'"{username}" site:twitter.com',
        f'intext:"{username}"',
    ]
    results["dorks"] = dork_queries

    return json.dumps(results, indent=2)


@tool
async def osint_email_investigation(
    email: str,
) -> str:
    """Investigate an email address for breach data and domain information.

    Use this tool to:
    - Check if email appears in data breaches
    - Get domain registration info
    - Generate investigation dorks

    Args:
        email: Email address to investigate

    Returns:
        JSON with breach check results, domain WHOIS info, and investigation dorks
    """
    import json

    results = {
        "email": email,
        "breach_check": None,
        "domain_info": None,
        "dorks": []
    }

    # Extract domain from email
    domain = email.split("@")[-1] if "@" in email else None

    # Breach check (if API key available)
    if _RAPIDAPI_KEY:
        results["breach_check"] = await _breach_lookup(email, "email")
    else:
        results["breach_check"] = {"note": "RAPIDAPI_KEY required for breach lookup"}

    # Domain WHOIS
    if domain:
        results["domain_info"] = await _domain_whois(domain)

    # Generate investigation dorks
    results["dorks"] = [
        f'"{email}"',
        f'intext:"{email}" filetype:txt',
        f'"{email}" site:linkedin.com',
        f'"{email}" site:github.com',
    ]

    return json.dumps(results, indent=2, default=str)


@tool
async def osint_domain_whois(
    domain: str,
) -> str:
    """Perform WHOIS lookup for a domain.

    Use this tool to:
    - Get domain registration information
    - Find registrar and name servers
    - Investigate website ownership

    Args:
        domain: Domain name to lookup (e.g., example.com)

    Returns:
        JSON with WHOIS information including registrar, creation date, and name servers
    """
    return await _domain_whois(domain)


@tool
async def osint_generate_dorks(
    search_term: str,
    dork_type: str = "general",
) -> str:
    """Generate Google dork queries for advanced searching.

    Use this tool to create targeted search queries for:
    - Email addresses
    - Phone numbers
    - Usernames
    - General searches

    Args:
        search_term: Term to search for
        dork_type: Type of dork to generate - "email", "phone", "username", or "general" (default)

    Returns:
        JSON with list of Google dork queries
    """
    import json

    dorks = []

    if dork_type == "email":
        dorks = [
            f'intext:"{search_term}"',
            f'"{search_term}" filetype:txt',
            f'"{search_term}" filetype:csv',
            f'"{search_term}" filetype:xls',
        ]
    elif dork_type == "phone":
        dorks = [
            f'"{search_term}"',
            f'"{search_term}" filetype:txt',
        ]
    elif dork_type == "username":
        dorks = [
            f'"{search_term}" site:github.com',
            f'"{search_term}" site:reddit.com',
            f'"{search_term}" site:twitter.com',
            f'intext:"{search_term}"',
        ]
    else:  # general
        dorks = [
            f'"{search_term}"',
            f'intext:"{search_term}"',
            f'intitle:"{search_term}"',
            f'{search_term} filetype:pdf',
        ]

    return json.dumps({
        "search_term": search_term,
        "dork_type": dork_type,
        "dorks": dorks,
    }, indent=2)


@tool
async def osint_breach_lookup(
    term: str,
    term_type: str = "email",
) -> str:
    """Check breach directory for email or username.

    Use this tool to:
    - Check if credentials appeared in data breaches
    - Search breach databases for compromised accounts

    Args:
        term: Email or username to check
        term_type: Type of term - "email" or "username" (default: "email")

    Returns:
        JSON with breach check results

    Note: Requires RAPIDAPI_KEY environment variable
    """
    import json

    if not _RAPIDAPI_KEY:
        return json.dumps({
            "error": "RAPIDAPI_KEY required for breach lookup",
            "term": term,
            "type": term_type
        }, indent=2)

    result = await _breach_lookup(term, term_type)
    return json.dumps(result, indent=2, default=str)


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

async def _domain_whois(domain: str) -> str:
    """Internal WHOIS lookup helper."""
    import json

    # Simplified WHOIS (for production, use whoisxmlapi.com or similar)
    results = {
        "domain": domain,
        "registrar": "Unknown (API required for full WHOIS)",
        "created_date": "Unknown",
        "name_servers": [],
        "status": "Active (assumed)",
        "note": "Full WHOIS data requires a paid API service"
    }

    return json.dumps(results, indent=2)


async def _breach_lookup(term: str, term_type: str) -> dict:
    """Internal breach lookup via RapidAPI."""
    import httpx

    try:
        url = "https://breachdirectory.p.rapidapi.com/"
        params = {"func": "auto", "term": term}
        headers = {"X-RapidAPI-Key": _RAPIDAPI_KEY}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"Breach lookup error for '{term[:30]}...': {e}")
        return {"error": str(e), "found": False, "breaches": []}
