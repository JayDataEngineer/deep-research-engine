"""Podcast Script Synthesis - Generates podcast scripts from context.

Reads from LangGraph Store and generates a podcast script with dialogue.
This is an OPTIONAL separate CLI command, NOT part of the main pipeline.

Usage:
    uv run context-engine synthesize --session-id abc123 --format podcast

Based on content_formatter from unified-llm-interface.
"""

import json
import re
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
        description="Emotional tone for TTS (happy, sad, angry, calm, etc.)",
    )
    stage_direction: Optional[str] = Field(
        default=None,
        description="Optional stage direction or action note",
    )


class PodcastSegment(BaseModel):
    """A segment of the podcast (e.g., intro, discussion, conclusion)."""

    segment_type: str = Field(
        description="Type of segment (intro, discussion, interview, conclusion, etc.)"
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional title for this segment",
    )
    dialogue: list[DialogueLine] = Field(
        description="Lines of dialogue in this segment",
    )


class PodcastScript(BaseModel):
    """A complete podcast script with character-driven dialogue."""

    title: str = Field(
        description="Podcast episode title"
    )
    summary: str = Field(
        description="Brief summary of the episode (1-2 sentences)",
        max_length=500,
    )
    segments: list[PodcastSegment] = Field(
        description="List of podcast segments in order",
    )
    total_duration_estimate: Optional[str] = Field(
        default=None,
        description="Estimated duration (e.g., '15-20 minutes')",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes for production or TTS processing",
    )


# =============================================================================
# PROMPT
# =============================================================================

PODCAST_SYNTHESIS_PROMPT = """You are an expert podcast script writer. Create an engaging podcast script based on the research context.

The podcast should:
1. Have two hosts (Alex and Sam) with distinct personalities
2. Alex: More analytical, asks clarifying questions
3. Sam: More enthusiastic, provides examples and stories
4. Feel natural and conversational
5. Break down complex topics for general audience
6. Include emotional variety (curiosity, excitement, thoughtfulness)

CONTEXT:
{context}

OUTPUT FORMAT (JSON):
{{
  "title": "Episode Title",
  "summary": "Brief episode summary",
  "segments": [
    {{
      "segment_type": "intro|discussion|conclusion",
      "title": "Optional segment title",
      "dialogue": [
        {{
          "speaker": "Alex|Sam",
          "text": "Spoken dialogue",
          "emotion": "neutral|happy|curious|excited|thoughtful",
          "stage_direction": "Optional direction"
        }}
      ]
    }}
  ],
  "total_duration_estimate": "15-20 minutes",
  "notes": "Production notes"
}}

STRUCTURE:
1. INTRO (2-3 min): Hook, introduce topic, why it matters
2. DISCUSSION (10-15 min): Main content, back and forth between hosts
3. CONCLUSION (2-3 min): Key takeaways, call to action

REQUIREMENTS:
- Natural dialogue with interruptions and reactions
- Mix of explanations, questions, and examples
- Emotional variety for engaging TTS
- Total runtime: 15-20 minutes

JSON OUTPUT:"""


# =============================================================================
# SYNTHESIS FUNCTION
# =============================================================================


async def synthesize_podcast(
    session_id: str,
    max_items: int = 20,
    hosts: Optional[dict[str, str]] = None,
) -> PodcastScript:
    """Synthesize a podcast script from session context.

    Args:
        session_id: Session ID to read context from
        max_items: Maximum items to include from store
        hosts: Optional custom host names/personalities

    Returns:
        PodcastScript with synthesized content

    Raises:
        ValueError: If no content found or synthesis fails
    """
    logger.info(f"Synthesizing podcast for session: {session_id}")

    # Get store and read context (lazy imports)
    store = await _get_store()
    context = await _get_context_for_synthesis(store, session_id, max_items)

    if not context:
        raise ValueError(f"No content found for session {session_id}")

    # Get model (lazy import)
    model = _get_model("smart")

    # Build prompt
    prompt = PODCAST_SYNTHESIS_PROMPT.format(context=context)

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

    # Create PodcastScript
    script = PodcastScript(
        title=data.get("title", "Research Podcast"),
        summary=data.get("summary", ""),
        segments=[
            PodcastSegment(
                segment_type=s.get("segment_type", "discussion"),
                title=s.get("title"),
                dialogue=[
                    DialogueLine(
                        speaker=d.get("speaker", "Host"),
                        text=d.get("text", ""),
                        emotion=d.get("emotion", "neutral"),
                        stage_direction=d.get("stage_direction"),
                    )
                    for d in s.get("dialogue", [])
                ],
            )
            for s in data.get("segments", [])
        ],
        total_duration_estimate=data.get("total_duration_estimate"),
        notes=data.get("notes"),
    )

    total_lines = sum(len(seg.dialogue) for seg in script.segments)
    logger.info(
        f"Synthesized podcast: {len(script.segments)} segments, {total_lines} dialogue lines"
    )

    return script


async def _get_context_for_synthesis(
    store: Any,
    session_id: str,
    max_items: int,
) -> str:
    """Get context from store for synthesis."""
    # Try different namespaces
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

    # Limit total context length
    context = "\n---\n".join(all_content)
    max_chars = 20000  # More context for longer podcast
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[...content truncated...]"

    return context


def format_podcast_output(script: PodcastScript) -> str:
    """Format podcast script for display.

    Args:
        script: PodcastScript to format

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"PODCAST: {script.title}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Summary: {script.summary}")
    lines.append(f"Estimated Duration: {script.total_duration_estimate or 'N/A'}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    for segment in script.segments:
        lines.append(f"[{segment.segment_type.upper()}]" + (
            f" - {segment.title}" if segment.title else ""
        ))
        lines.append("-" * 40)
        lines.append("")

        for line in segment.dialogue:
            if line.stage_direction:
                lines.append(f"  [{line.stage_direction}]")
            lines.append(f"{line.speaker.upper()}: {line.text}")
            lines.append("")

        lines.append("")

    if script.notes:
        lines.append("=" * 60)
        lines.append("PRODUCTION NOTES:")
        lines.append(script.notes)
        lines.append("=" * 60)

    return "\n".join(lines)
