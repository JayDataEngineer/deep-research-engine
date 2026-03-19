"""Fact Check Tool - Verifies claims against web sources.

This tool provides automated fact-checking capabilities by:
1. Searching for relevant sources about a claim
2. Analyzing multiple sources for verification
3. Returning a structured fact-check report with citations

Usage:
    from src.shared.tools.ulli_langchain.web.fact_check import fact_check

    result = await fact_check(
        claim="ICE shot two people in Minneapolis in January 2026",
        confidence_threshold=0.7
    )
"""

import asyncio
from src.shared.logging_config import get_logger
from typing import Literal, Optional
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import base class
from src.shared._base_models import BaseWebSearchInput

logger = get_logger(__name__)


class FactCheckInput(BaseWebSearchInput):
    """Input for fact-checking with validation."""

    claim: str = Field(
        ...,
        description="The claim or statement to fact-check. Be specific and include key details like names, dates, locations."
    )
    context: str = Field(
        default="",
        description="Additional context about the claim (optional, but helps accuracy)"
    )
    search_depth: Literal["quick", "standard", "deep"] = Field(
        default="standard",
        description="How thoroughly to search: quick (5 results), standard (10 results), deep (20 results)"
    )
    # Inherits: time_range, category, location from BaseWebSearchInput


class FactCheckResult(BaseModel):
    """Structured result from fact-checking."""

    claim: str = Field(description="The claim that was checked")
    verdict: Literal["TRUE", "MOSTLY_TRUE", "MIXED", "MOSTLY_FALSE", "FALSE", "UNVERIFIABLE"] = Field(
        description="Overall verdict based on sources"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (0-1)"
    )
    summary: str = Field(description="Summary of findings")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting the claim"
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence contradicting the claim"
    )
    sources: list[dict] = Field(
        default_factory=list,
        description="Sources consulted with URLs and snippets"
    )
    fact_checks_found: list[dict] = Field(
        default_factory=list,
        description="Professional fact-checks found (Snopes, PolitiFact, etc.)"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Any caveats or limitations in the assessment"
    )
    checked_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="When the fact-check was performed"
    )


@tool(args_schema=FactCheckInput)
async def fact_check(
    claim: str,
    context: str = "",
    search_depth: Literal["quick", "standard", "deep"] = "standard",
    time_range: Literal["day", "week", "month", "year", "noLimit"] = "month",
    category: str = "news",
    location: str = "us",
) -> str:
    """Fact-check a claim by searching for and analyzing web sources.

    This tool searches multiple sources to verify the accuracy of a claim,
    providing a structured assessment with supporting evidence.

    Best practices:
    - Be specific in your claim (include names, dates, locations)
    - Use shorter time_range for recent events (day, week)
    - Use "news" category for current events and breaking stories
    - Use "general" category for historical or scientific claims

    Args:
        claim: The claim or statement to fact-check
        context: Additional context (helps accuracy)
        search_depth: How thoroughly to search (quick/standard/deep)
        time_range: Time range for search (day/week/month/year/noLimit)
        category: Content category (news/general/science/etc.)
        location: Location for region-specific results (cn or us)

    Returns:
        Structured fact-check report with verdict, evidence, and sources
    """
    try:
        # Import shared lang_tools for web search
        from logic.web import web_search as lang_tool_search

        # Determine number of results based on search depth
        num_results_map = {"quick": 5, "standard": 10, "deep": 20}
        num_results = num_results_map.get(search_depth, 10)

        # Build comprehensive search query
        search_query = claim
        if context:
            search_query = f"{claim} {context}"

        logger.info(f"[FACT_CHECK] Checking claim: {claim[:80]}...")
        logger.info(f"[FACT_CHECK] Search depth: {search_depth}, time_range: {time_range}")

        # Perform web search
        search_results = await lang_tool_search(
            query=search_query,
            recency=time_range if time_range != "noLimit" else None,
            category=category,
            num_results=num_results,
            location=location,
            pages=min(5, num_results)  # Don't over-fetch
        )

        # Also try a "fact check" specific search
        factcheck_query = f'fact check "{claim}"'
        if context:
            factcheck_query = f'fact check "{claim}" {context}'

        factcheck_results = await lang_tool_search(
            query=factcheck_query,
            recency=time_range if time_range != "noLimit" else None,
            category=category,
            num_results=max(3, num_results // 2),
            location=location,
            pages=3
        )

        # Combine results
        all_results = search_results + factcheck_results

        # Analyze results to determine verdict
        result = _analyze_fact_check_results(claim, all_results, context)

        # Return as formatted string (LangChain tools return strings)
        return _format_fact_check_result(result)

    except Exception as e:
        logger.error(f"[FACT_CHECK] Error fact-checking claim: {e}")
        # Return error as structured result
        error_result = FactCheckResult(
            claim=claim,
            verdict="UNVERIFIABLE",
            confidence=0.0,
            summary=f"Unable to verify claim due to error: {str(e)}",
            caveats=["Error occurred during fact-checking process"]
        )
        return _format_fact_check_result(error_result)


def _analyze_fact_check_results(
    claim: str,
    search_results: str,
    context: str
) -> FactCheckResult:
    """Analyze search results to determine fact-check verdict.

    Args:
        claim: The claim being checked
        search_results: Raw search results from web search
        context: Additional context

    Returns:
        FactCheckResult with verdict and evidence
    """
    # Parse search results
    supporting = []
    contradicting = []
    sources = []
    fact_checks = []
    caveats = []

    # Parse individual results from the formatted string
    lines = search_results.split('\n')
    current_result = {}

    for line in lines:
        line = line.strip()
        if not line:
            if current_result:
                sources.append(current_result)
                current_result = {}
            continue

        if line.startswith("Title:"):
            if current_result:
                sources.append(current_result)
            current_result = {"title": line[6:].strip()}
        elif line.startswith("URL:"):
            current_result["url"] = line[4:].strip()
        elif line.startswith("Snippet:"):
            current_result["snippet"] = line[8:].strip()

    # Add the last result
    if current_result:
        sources.append(current_result)

    # Analyze sources for supporting/contradicting evidence
    claim_lower = claim.lower()

    for source in sources:
        snippet = source.get("snippet", "").lower()
        title = source.get("title", "").lower()
        combined = f"{title} {snippet}"

        # Check for fact-checking sites
        domain = source.get("url", "").lower()
        if any(site in domain for site in ["snopes", "politifact", "factcheck", "reuters", "apnews"]):
            fact_checks.append(source)

        # Simple keyword-based analysis (in production, use LLM for this)
        # Supporting indicators
        if any(word in combined for word in ["confirmed", "verified", "true", "reported", "according to"]):
            if claim_lower.split()[:3] & set(combined.split()):
                supporting.append(source.get("snippet", ""))

        # Contradicting indicators
        if any(word in combined for word in ["false", "debunked", "untrue", "incorrect", "misleading"]):
            contradicting.append(source.get("snippet", ""))

    # Determine verdict
    if len(supporting) > len(contradicting) * 2:
        verdict = "TRUE"
        confidence = min(0.9, 0.5 + (len(supporting) / 10))
    elif len(supporting) > len(contradicting):
        verdict = "MOSTLY_TRUE"
        confidence = 0.7
    elif len(contradicting) > len(supporting) * 2:
        verdict = "FALSE"
        confidence = min(0.9, 0.5 + (len(contradicting) / 10))
    elif len(contradicting) > len(supporting):
        verdict = "MOSTLY_FALSE"
        confidence = 0.7
    elif len(supporting) == 0 and len(contradicting) == 0:
        verdict = "UNVERIFIABLE"
        confidence = 0.3
        caveats.append("No direct sources found to confirm or deny the claim")
    else:
        verdict = "MIXED"
        confidence = 0.5
        caveats.append("Sources provide conflicting information")

    # Build summary
    summary_parts = []
    if supporting:
        summary_parts.append(f"Found {len(supporting)} source(s) supporting the claim")
    if contradicting:
        summary_parts.append(f"Found {len(contradicting)} source(s) contradicting the claim")
    if fact_checks:
        summary_parts.append(f"Found {len(fact_checks)} professional fact-check(s)")

    summary = ". ".join(summary_parts) if summary_parts else "Insufficient evidence to verify claim"

    return FactCheckResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        summary=summary,
        supporting_evidence=supporting[:5],  # Limit to top 5
        contradicting_evidence=contradicting[:5],
        sources=sources[:10],  # Limit to top 10
        fact_checks_found=fact_checks,
        caveats=caveats
    )


def _format_fact_check_result(result: FactCheckResult) -> str:
    """Format fact-check result as readable string.

    Args:
        result: FactCheckResult to format

    Returns:
        Formatted string representation
    """
    lines = [
        "=" * 60,
        f"FACT CHECK RESULT: {result.verdict}",
        f"Confidence: {result.confidence:.0%}",
        "=" * 60,
        "",
        f"Claim: {result.claim}",
        "",
        f"Summary: {result.summary}",
        ""
    ]

    if result.supporting_evidence:
        lines.extend([
            "Supporting Evidence:",
            "-" * 40
        ])
        for i, evidence in enumerate(result.supporting_evidence, 1):
            lines.append(f"{i}. {evidence[:200]}...")
        lines.append("")

    if result.contradicting_evidence:
        lines.extend([
            "Contradicting Evidence:",
            "-" * 40
        ])
        for i, evidence in enumerate(result.contradicting_evidence, 1):
            lines.append(f"{i}. {evidence[:200]}...")
        lines.append("")

    if result.fact_checks_found:
        lines.extend([
            "Professional Fact-Checks:",
            "-" * 40
        ])
        for fc in result.fact_checks_found[:3]:
            lines.append(f"- {fc.get('title', 'Unknown')}")
            lines.append(f"  {fc.get('url', 'No URL')}")
        lines.append("")

    if result.sources:
        lines.extend([
            "Sources Consulted:",
            "-" * 40
        ])
        for source in result.sources[:5]:
            lines.append(f"- {source.get('title', 'Unknown')}")
            if source.get('url'):
                lines.append(f"  {source['url']}")
        lines.append("")

    if result.caveats:
        lines.extend([
            "Caveats:",
            "-" * 40
        ])
        for caveat in result.caveats:
            lines.append(f"- {caveat}")
        lines.append("")

    lines.extend([
        f"Checked at: {result.checked_at}",
        "=" * 60
    ])

    return "\n".join(lines)


# Export utility function for direct use (outside of tool context)
async def check_claim(
    claim: str,
    **kwargs
) -> FactCheckResult:
    """Check a claim and return structured result.

    Utility function for using fact-check outside of LangChain tool context.

    Args:
        claim: The claim to check
        **kwargs: Additional arguments passed to fact_check

    Returns:
        FactCheckResult with full details
    """
    result_str = await fact_check(claim, **kwargs)
    # Parse the result back into a FactCheckResult
    # In production, you might want to return the result object directly
    # instead of serializing to string
    return _parse_fact_check_result(result_str)


def _parse_fact_check_result(result_str: str) -> FactCheckResult:
    """Parse fact-check result string back into FactCheckResult.

    Args:
        result_str: Formatted result string

    Returns:
        FactCheckResult object
    """
    # Simple parsing - in production, use proper serialization
    lines = result_str.split('\n')

    verdict = "UNVERIFIABLE"
    confidence = 0.5
    summary = ""
    sources = []

    for line in lines:
        if line.startswith("FACT CHECK RESULT:"):
            verdict = line.split(":")[1].strip()
        elif line.startswith("Confidence:"):
            confidence_str = line.split(":")[1].strip().rstrip('%')
            confidence = float(confidence_str) / 100
        elif line.startswith("Summary:"):
            summary = line.split(":", 1)[1].strip()

    return FactCheckResult(
        claim="Unknown",  # Would need proper parsing
        verdict=verdict,
        confidence=confidence,
        summary=summary,
        sources=sources
    )
