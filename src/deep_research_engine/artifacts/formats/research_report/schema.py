"""Research Report Schema - Pydantic model for structured research reports.

This schema defines the structure for a comprehensive research report
with sections, citations, and executive summary.
"""

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation/reference in the research report."""

    id: str = Field(description="Unique citation ID (e.g., '1', '2', or 'smith2024')")
    title: str = Field(description="Title of the source")
    url: str | None = Field(default=None, description="URL of the source")
    authors: list[str] | None = Field(default=None, description="Authors of the source")
    date: str | None = Field(default=None, description="Publication date")
    source_type: str | None = Field(
        default="web", description="Type of source (web, paper, book, etc.)"
    )


class Section(BaseModel):
    """A section in the research report."""

    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown format")
    subsections: list["Section"] = Field(
        default_factory=list, description="Optional nested subsections"
    )
    citations: list[str] = Field(
        default_factory=list, description="List of citation IDs referenced in this section"
    )


class KeyFinding(BaseModel):
    """A key finding from the research."""

    finding: str = Field(description="The key finding or insight")
    confidence: float = Field(description="Confidence level 0.0-1.0", ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(
        default_factory=list, description="List of evidence supporting this finding"
    )
    citations: list[str] = Field(
        default_factory=list, description="Citation IDs supporting this finding"
    )


class ResearchReport(BaseModel):
    """A comprehensive research report with structured sections."""

    title: str = Field(description="Report title")
    executive_summary: str = Field(
        description="Executive summary (2-4 paragraphs)", max_length=2000
    )
    key_findings: list[KeyFinding] = Field(description="List of key findings from the research")
    sections: list[Section] = Field(description="Main report sections in order")
    conclusion: str = Field(description="Conclusion and recommendations")
    citations: list[Citation] = Field(description="List of all citations referenced in the report")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata (research date, methodology, etc.)"
    )
