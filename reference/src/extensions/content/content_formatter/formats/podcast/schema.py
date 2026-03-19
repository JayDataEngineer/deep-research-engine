"""Podcast Script Pydantic Schema."""

from pydantic import BaseModel, Field
from typing import List, Optional


class DialogueLine(BaseModel):
    """A single line of dialogue in the podcast."""

    speaker: str = Field(
        description="Speaker name (e.g., 'Olivia', 'Emma', 'Host')"
    )
    text: str = Field(
        description="The spoken dialogue"
    )
    emotion: Optional[str] = Field(
        default="neutral",
        description="Emotional tone for TTS (happy, sad, angry, calm, etc.)"
    )
    stage_direction: Optional[str] = Field(
        default=None,
        description="Optional stage direction or action note"
    )


class PodcastSegment(BaseModel):
    """A segment of the podcast (e.g., intro, discussion, conclusion)."""

    segment_type: str = Field(
        description="Type of segment (intro, discussion, interview, conclusion, etc.)"
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional title for this segment"
    )
    dialogue: List[DialogueLine] = Field(
        description="Lines of dialogue in this segment"
    )


class PodcastScript(BaseModel):
    """A complete podcast script with character-driven dialogue."""

    title: str = Field(
        description="Podcast episode title"
    )
    summary: str = Field(
        description="Brief summary of the episode (1-2 sentences)",
        max_length=500
    )
    segments: List[PodcastSegment] = Field(
        description="List of podcast segments in order"
    )
    total_duration_estimate: Optional[str] = Field(
        default=None,
        description="Estimated duration (e.g., '15-20 minutes')"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes for production or TTS processing"
    )
