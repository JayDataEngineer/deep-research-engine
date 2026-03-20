"""Markdown Output Schema - Pydantic model for markdown research output.

Simple schema for markdown-formatted research output with optional citations.
"""

from typing import Any

from pydantic import BaseModel, Field


class MarkdownSource(BaseModel):
    """A source/citation in the markdown output."""

    id: str = Field(description="Unique source ID")
    title: str = Field(description="Title of the source")
    url: str | None = Field(default=None, description="URL of the source")
    snippet: str | None = Field(default=None, description="Brief excerpt or description")


class MarkdownOutput(BaseModel):
    """Markdown-formatted research output."""

    title: str = Field(description="Document title")
    content: str = Field(description="Full markdown content of the research output")
    sources: list[MarkdownSource] = Field(
        default_factory=list, description="List of sources cited in the document"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata (tags, category, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Research Findings",
                "content": "# Research Findings\n\n## Summary\n\nThis report presents...",
                "sources": [
                    {
                        "id": "1",
                        "title": "Example Source",
                        "url": "https://example.com",
                        "snippet": "Key information...",
                    }
                ],
            }
        }
