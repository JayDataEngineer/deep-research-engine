"""Podcast output formatter for research extension.

Formats LLM responses as podcast scripts with dialogue validation
and segment breaks for parallel TTS processing.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR PODCAST OUTPUT VALIDATION
# =============================================================================

class DialogueLine(BaseModel):
    """A single line of dialogue in a podcast script."""

    character: str = Field(description="Character name speaking")
    emotion: Optional[str] = Field(default=None, description="Emotional tone (e.g., 'excited', 'deadpan', 'sarcastic')")
    dialogue: str = Field(description="The spoken dialogue text")
    action: Optional[str] = Field(default=None, description="Stage direction or action (e.g., 'laughs', 'pauses')")


class PodcastSegment(BaseModel):
    """A segment of a podcast (intro, story discussion, outro, etc.)."""

    name: str = Field(description="Segment name (e.g., 'Intro', 'Story 1: Economy', 'Outro')")
    scene_direction: Optional[str] = Field(default=None, description="Scene setting in [brackets]")
    dialogue: List[DialogueLine] = Field(description="Lines of dialogue in this segment")


class PodcastOutput(BaseModel):
    """Podcast script output with structured dialogue."""

    title: str = Field(description="Episode title")
    date: str = Field(description="Episode date (YYYY-MM-DD)")
    duration_estimate: Optional[str] = Field(default=None, description="Estimated duration (e.g., '30 minutes')")
    word_count: Optional[int] = Field(default=None, description="Total word count")
    hosts: List[str] = Field(description="Host character names")
    segments: List[PodcastSegment] = Field(description="Podcast segments in order")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# PODCAST FORMATTER
# =============================================================================

class PodcastFormatter:
    """Format output as a podcast script with dialogue validation.

    This formatter handles:
    - Podcast script parsing from markdown format
    - Structured dialogue validation
    - Segment break markers for parallel TTS processing
    """

    def format(self, response: str, config: Dict[str, Any]) -> str:
        """Format response as podcast script.

        Args:
            response: LLM response text (should be script format)
            config: Output configuration

        Returns:
            Formatted podcast script
        """
        # If validation is enabled, try to parse as structured script
        if config.get("validate", False):
            # Try to parse as JSON first (structured output)
            try:
                import json
                script_data = json.loads(response)
                output = PodcastOutput(**script_data)
                return self._render(output)
            except (json.JSONDecodeError, TypeError):
                # Parse as markdown-style script
                output = self._parse_script_format(response, config)
                return self._render(output)
        else:
            # Simple rendering without validation
            return self._simple_render(response, config)

    def _parse_script_format(self, response: str, config: Dict[str, Any]) -> PodcastOutput:
        """Parse markdown-style script into structured PodcastOutput.

        Expected format:
        ## Episode: Title

        **Date:** YYYY-MM-DD
        **Hosts:** Host A, Host B

        ### Intro

        [SCENE: Studio description]

        HOST A (emotion): Dialogue line
        HOST B (emotion): Dialogue line
        """
        lines = response.strip().split('\n')
        segments = []
        current_segment = None
        current_dialogue = []

        # Extract metadata
        title = config.get("title", "Untitled Episode")
        date = datetime.now().strftime("%Y-%m-%d")
        hosts = []

        for i, line in enumerate(lines):
            # Episode title
            if line.startswith("## Episode:") or line.startswith("## Episode:"):
                title = line.split(":", 1)[1].strip()

            # Date
            elif line.startswith("**Date:**"):
                date = line.split(":", 1)[1].strip()

            # Hosts
            elif line.startswith("**Hosts:**"):
                hosts_str = line.split(":", 1)[1].strip()
                hosts = [h.strip() for h in hosts_str.split(",")]

            # Segment header
            elif line.startswith("###"):
                # Save previous segment
                if current_segment and current_dialogue:
                    segments.append(PodcastSegment(
                        name=current_segment,
                        dialogue=current_dialogue
                    ))

                # Start new segment
                current_segment = line.strip("#").strip()
                current_dialogue = []

            # Scene direction
            elif line.strip().startswith("[SCENE:") or line.strip().startswith("["):
                if current_dialogue:
                    # Add scene direction to last dialogue or create placeholder
                    pass

            # Dialogue line
            elif ":" in line and not line.strip().startswith("#"):
                # Parse "CHARACTER (emotion): dialogue"
                dialogue_line = self._parse_dialogue_line(line)
                if dialogue_line:
                    current_dialogue.append(dialogue_line)

        # Don't forget the last segment
        if current_segment and current_dialogue:
            segments.append(PodcastSegment(
                name=current_segment,
                dialogue=current_dialogue
            ))

        return PodcastOutput(
            title=title,
            date=date,
            duration_estimate=config.get("target_duration"),
            word_count=len(response.split()),
            hosts=hosts,
            segments=segments
        )

    def _parse_dialogue_line(self, line: str) -> Optional[DialogueLine]:
        """Parse a dialogue line.

        Expected formats:
        - CHARACTER (emotion): dialogue
        - CHARACTER: dialogue
        """
        # Match pattern: CHARACTER (optional emotion): dialogue
        match = re.match(r'^\s*([A-Z\s/]+?)\s*(?:\(([^)]+)\))?\s*:\s*(.+)$', line)
        if match:
            character = match.group(1).strip()
            emotion = match.group(2).strip() if match.group(2) else None
            dialogue = match.group(3).strip()
            return DialogueLine(
                character=character,
                emotion=emotion,
                dialogue=dialogue
            )
        return None

    def _render(self, output: PodcastOutput) -> str:
        """Render validated podcast output."""
        lines = []

        # Header
        lines.append(f"## Episode: {output.title}")
        lines.append("")
        lines.append(f"**Date:** {output.date}")
        if output.duration_estimate:
            lines.append(f"**Duration:** ~{output.duration_estimate}")
        if output.hosts:
            lines.append(f"**Hosts:** {', '.join(output.hosts)}")
        lines.append("")

        # Word count
        if output.word_count:
            lines.append(f"*Word count: {output.word_count:,}*")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Segments - with unique section break marker for parallel processing
        for i, segment in enumerate(output.segments):
            # UNIQUE SECTION BREAK MARKER for IndexTTS parallel processing
            lines.append("<<<SEGMENT_BREAK>>>")
            lines.append(f"### {segment.name}")
            lines.append("")

            if segment.scene_direction:
                lines.append(segment.scene_direction)
                lines.append("")

            for dialogue_line in segment.dialogue:
                # Format: CHARACTER (emotion): dialogue
                parts = [dialogue_line.character]
                if dialogue_line.emotion:
                    parts.append(f"({dialogue_line.emotion})")
                parts.append(f": {dialogue_line.dialogue}")

                line_str = " ".join(parts)
                lines.append(line_str)

                if dialogue_line.action:
                    lines.append(f"*{dialogue_line.action}*")

            lines.append("")

        return "\n".join(lines)

    def _simple_render(self, response: str, config: Dict[str, Any]) -> str:
        """Simple rendering without validation."""
        # Just add metadata header if not present
        if not response.startswith("##"):
            lines = []
            lines.append(f"## Episode: {config.get('title', 'Untitled')}")
            lines.append("")
            lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
            lines.append("")
            lines.append("---")
            lines.append("")

            # Add section break markers before each ### or ## SEGMENT header for IndexTTS parallel processing
            for line in response.split('\n'):
                if line.strip().startswith('###') or (line.strip().startswith('##') and 'SEGMENT' in line.upper()):
                    lines.append("<<<SEGMENT_BREAK>>>")
                lines.append(line)

            return "\n".join(lines)
        else:
            # Response already has metadata, just add section breaks
            lines = []
            for line in response.split('\n'):
                if line.strip().startswith('###') or (line.strip().startswith('##') and 'SEGMENT' in line.upper()):
                    lines.append("<<<SEGMENT_BREAK>>>")
                lines.append(line)
            return "\n".join(lines)
