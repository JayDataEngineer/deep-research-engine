"""Report Synthesis - Generates structured reports from context.

Reads from LangGraph Store and generates a comprehensive report.
This is an OPTIONAL separate CLI command, NOT part of the main pipeline.

Usage:
    uv run context-engine synthesize --session-id abc123 --format report
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field
from loguru import logger


def _get_model(tier: str):
    """Lazy import for model."""
    from context_engine.config import get_model
    return get_model(tier)


async def _get_store():
    """Lazy import for store."""
    from context_engine.store.setup import get_store
    return await get_store()


# =============================================================================
# SCHEMAS
# =============================================================================


class ReportSection(BaseModel):
    """A section of the report."""

    title: str = Field(
        description="Section title"
    )
    content: str = Field(
        description="Section content (markdown supported)"
    )
    subsections: list["ReportSection"] = Field(
        default_factory=list,
        description="Nested subsections",
    )


class Report(BaseModel):
    """A comprehensive research report."""

    title: str = Field(
        description="Report title"
    )
    executive_summary: str = Field(
        description="Executive summary (2-3 paragraphs)",
        max_length=1000,
    )
    sections: list[ReportSection] = Field(
        description="Main report sections",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Key findings as bullet points",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations based on findings",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source references",
    )
    generated_at: str = Field(
        default="",
        description="Generation timestamp",
    )


# =============================================================================
# PROMPT
# =============================================================================

REPORT_SYNTHESIS_PROMPT = """You are an expert research analyst. Create a comprehensive report based on the provided context.

CONTEXT:
{context}

OUTPUT FORMAT (JSON):
{{
  "title": "Report Title",
  "executive_summary": "2-3 paragraph summary of key insights",
  "sections": [
    {{
      "title": "Section Title",
      "content": "Section content with markdown formatting",
      "subsections": []
    }}
  ],
  "key_findings": [
    "Finding 1",
    "Finding 2"
  ],
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ],
  "sources": ["Source references if mentioned"]
}}

REQUIREMENTS:
- Professional tone
- Clear structure with headers
- Use markdown for formatting
- Include specific data and examples from context
- Executive summary should be 2-3 paragraphs
- 3-5 key findings
- Actionable recommendations

SECTIONS TO INCLUDE:
1. Introduction/Overview
2. Analysis (break into logical subsections)
3. Discussion
4. Conclusions

JSON OUTPUT:"""


# =============================================================================
# SYNTHESIS FUNCTION
# =============================================================================


async def synthesize_report(
    session_id: str,
    max_items: int = 30,
    report_type: str = "research",
) -> Report:
    """Synthesize a comprehensive report from session context.

    Args:
        session_id: Session ID to read context from
        max_items: Maximum items to include from store
        report_type: Type of report (research, analysis, summary)

    Returns:
        Report with synthesized content

    Raises:
        ValueError: If no content found or synthesis fails
    """
    logger.info(f"Synthesizing report for session: {session_id}")

    # Get store and read context (lazy imports)
    store = await _get_store()
    context = await _get_context_for_synthesis(store, session_id, max_items)

    if not context:
        raise ValueError(f"No content found for session {session_id}")

    # Get model (lazy import)
    model = _get_model("smart")

    # Build prompt
    prompt = REPORT_SYNTHESIS_PROMPT.format(
        context=context,
        report_type=report_type,
    )

    # Invoke model
    response = await model.ainvoke(prompt)
    content = response.content

    # Parse JSON
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        raise ValueError("No JSON found in synthesis response")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse synthesis JSON: {e}")

    # Create Report
    from datetime import datetime

    report = Report(
        title=data.get("title", "Research Report"),
        executive_summary=data.get("executive_summary", ""),
        sections=[
            ReportSection(
                title=s.get("title", ""),
                content=s.get("content", ""),
                subsections=[
                    ReportSection(
                        title=ss.get("title", ""),
                        content=ss.get("content", ""),
                    )
                    for ss in s.get("subsections", [])
                ],
            )
            for s in data.get("sections", [])
        ],
        key_findings=data.get("key_findings", []),
        recommendations=data.get("recommendations", []),
        sources=data.get("sources", []),
        generated_at=datetime.utcnow().isoformat(),
    )

    total_sections = len(report.sections)
    logger.info(
        f"Synthesized report: {total_sections} sections, "
        f"{len(report.key_findings)} findings"
    )

    return report


async def _get_context_for_synthesis(
    store: Any,
    session_id: str,
    max_items: int,
) -> str:
    """Get context from store for synthesis."""
    namespaces = [
        (session_id, "memories"),
        (session_id, "input"),
        (session_id, "summary"),
    ]

    all_content = []

    for namespace in namespaces:
        try:
            results = await store.asearch(namespace, query="", limit=max_items)
            for result in results:
                content = result.value.get("content", "")
                if content:
                    all_content.append(f"[{result.key}]\n{content}\n")
        except Exception as e:
            logger.debug(f"No content in {namespace}: {e}")

    if not all_content:
        return ""

    # More context for comprehensive reports
    context = "\n---\n".join(all_content)
    max_chars = 25000
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[...content truncated...]"

    return context


def format_report_output(report: Report) -> str:
    """Format report for display (Markdown).

    Args:
        report: Report to format

    Returns:
        Formatted markdown string
    """
    lines = []

    # Title
    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(f"*Generated: {report.generated_at}*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.executive_summary)
    lines.append("")

    # Sections
    for section in report.sections:
        lines.append(f"## {section.title}")
        lines.append("")
        lines.append(section.content)
        lines.append("")

        for subsection in section.subsections:
            lines.append(f"### {subsection.title}")
            lines.append("")
            lines.append(subsection.content)
            lines.append("")

    # Key Findings
    if report.key_findings:
        lines.append("## Key Findings")
        lines.append("")
        for finding in report.key_findings:
            lines.append(f"- {finding}")
        lines.append("")

    # Recommendations
    if report.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    # Sources
    if report.sources:
        lines.append("## Sources")
        lines.append("")
        for source in report.sources:
            lines.append(f"- {source}")
        lines.append("")

    return "\n".join(lines)
