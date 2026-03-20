"""Structured JSON Schema - Pydantic model for JSON research output.

Schema for structured JSON output suitable for programmatic consumption.
"""

from typing import Any

from pydantic import BaseModel, Field


class JSONFinding(BaseModel):
    """A single research finding."""

    title: str = Field(description="Brief title for this finding")
    description: str = Field(description="Detailed description of the finding")
    confidence: float = Field(description="Confidence level 0.0-1.0", ge=0.0, le=1.0)
    sources: list[str] = Field(
        default_factory=list, description="List of source IDs supporting this finding"
    )
    tags: list[str] = Field(default_factory=list, description="Tags or categories for this finding")


class JSONSource(BaseModel):
    """A source/reference."""

    id: str = Field(description="Unique source ID")
    title: str = Field(description="Title of the source")
    url: str | None = Field(default=None, description="URL of the source")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata (author, date, etc.)"
    )


class StructuredJSON(BaseModel):
    """Structured JSON research output."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Brief summary of the research")
    findings: list[JSONFinding] = Field(description="List of research findings")
    sources: list[JSONSource] = Field(description="List of sources")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata (query, timestamp, etc.)"
    )
