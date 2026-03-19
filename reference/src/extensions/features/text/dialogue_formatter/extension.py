"""Dialogue Formatter Extension - Parse text and apply emotion tagging for TTS.

This extension provides reusable functionality for:
- Parsing text into dialogue lines
- Applying emotion tagging via LLM
- Returning formatted dialogue with emotion vectors

It works with ANY text input, not just podcasts.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from src.extensions.core.base import LogicExtension

# Import from centralized content_formatter emotion module
from src.extensions.content.content_formatter.emotion import (
    DialogueLine,
    EmotionVector,
    detect_emotions,
)

logger = logging.getLogger(__name__)


class DialogueFormatterExtension(LogicExtension):
    """Dialogue formatter extension with emotion tagging."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)

    async def initialize(self, context):
        """Initialize the dialogue formatter extension."""
        await super().initialize(context)
        logger.info("Dialogue formatter extension initialized")

    async def shutdown(self):
        """Cleanup resources."""
        pass

    async def parse(
        self,
        text: str,
        patterns: Optional[List[str]] = None,
        supported_speakers: Optional[List[str]] = None,
    ) -> List[DialogueLine]:
        """Parse text into dialogue lines.

        Args:
            text: Text to parse
            patterns: Optional custom regex patterns (uses config patterns if None)
            supported_speakers: Optional list of valid speakers (auto-detect if None)

        Returns:
            List of DialogueLine objects
        """
        config_parsing = self.config.get("processing", {}).get("parsing", {})

        # Use provided patterns or config patterns
        if patterns is None:
            patterns = config_parsing.get("patterns", [])

        # Use provided speakers or config speakers
        if supported_speakers is None:
            supported_speakers = config_parsing.get("supported_speakers", [])

        dialogue_lines = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try each pattern
            for pattern in patterns:
                # Convert pattern to regex
                # Pattern format: "**{speaker}:** {text}" -> r"^\*\*(?P<speaker>[^(:]+)\*\*\s*:\*\*\s*(?P<text>.+)$"
                regex_pattern = self._pattern_to_regex(pattern)

                match = re.match(regex_pattern, line)
                if match:
                    speaker = match.group("speaker").strip()
                    direction = match.group("direction") if "direction" in match.groupdict() else ""
                    text = match.group("text").strip() if "text" in match.groupdict() else ""

                    # Filter by supported speakers if specified
                    if supported_speakers and speaker not in supported_speakers:
                        continue

                    # Skip empty lines or stage directions
                    if text and text not in ['[...]', '(silence)', '(pause)', '(scene change)']:
                        dialogue_lines.append(DialogueLine(speaker, text, direction))
                    break

        logger.info(f"Parsed {len(dialogue_lines)} dialogue lines")
        return dialogue_lines

    def _pattern_to_regex(self, pattern: str) -> str:
        """Convert a simple pattern to regex.

        Pattern format: "**{speaker}:** {text}"
        Regex format: r"^\*\*(?P<speaker>[^:]+)\*\*\s*:\*\*\s*(?P<text>.+)$"
        """
        # Replace placeholders with named groups
        regex = pattern

        # Escape special regex characters except for our placeholders
        for char in r"[]().*+?{}|^$":
            regex = regex.replace(char, "\\" + char)

        # Replace {placeholder} with named groups
        regex = regex.replace(r"\{speaker\}", r"(?P<speaker>[^:(]+)")
        regex = regex.replace(r"\{direction\}", r"(?P<direction>\([^)]*\))")
        regex = regex.replace(r"\{text\}", r"(?P<text>.+)")

        # Unescape necessary characters
        regex = regex.replace(r"\*", r"*")

        return regex

    async def tag_emotions(self, dialogue_lines: List[DialogueLine]) -> List[DialogueLine]:
        """Apply emotion tagging to dialogue lines using centralized content_formatter.

        Args:
            dialogue_lines: List of dialogue lines

        Returns:
            List of dialogue lines with emotion vectors
        """
        emotion_config = self.config.get("processing", {}).get("emotion_detection", {})
        enabled = emotion_config.get("enabled", True)

        if not enabled:
            logger.info("Emotion detection disabled, using neutral emotions")
            for line in dialogue_lines:
                line.emotion = EmotionVector.neutral()
            return dialogue_lines

        logger.info(f"Tagging emotions for {len(dialogue_lines)} dialogue lines using content_formatter...")

        try:
            # Use centralized emotion detection
            detected_lines, llm_info = await detect_emotions(dialogue_lines)
            return detected_lines

        except Exception as e:
            logger.warning(f"Emotion tagging failed: {e}, using neutral emotions")
            for line in dialogue_lines:
                line.emotion = EmotionVector.neutral()
            return dialogue_lines

    async def process(self, input_data: dict, **kwargs):
        """Process text to format as dialogue with emotions.

        Args:
            input_data: Dict with 'text' key containing the text to process
            **kwargs: Additional parameters

        Yields:
            Progress updates and final result
        """
        text = input_data.get("text")
        if not text:
            raise ValueError("input_data must contain 'text'")

        yield json.dumps({"status": "parsing", "message": "Parsing dialogue..."})

        # Parse dialogue
        dialogue_lines = await self.parse(text)

        # Tag emotions
        yield json.dumps({"status": "tagging", "message": f"Tagging emotions for {len(dialogue_lines)} lines..."})

        dialogue_lines = await self.tag_emotions(dialogue_lines)

        # Format output
        output_format = self.config.get("output", {}).get("format", "object")

        if output_format == "json":
            result = {
                "status": "complete",
                "dialogue_lines": [line.to_dict() for line in dialogue_lines],
                "count": len(dialogue_lines),
            }
        elif output_format == "yaml":
            import yaml
            result = {
                "status": "complete",
                "dialogue_yaml": yaml.dump([line.to_dict() for line in dialogue_lines]),
                "count": len(dialogue_lines),
            }
        else:  # object
            result = {
                "status": "complete",
                "dialogue_lines": dialogue_lines,
                "count": len(dialogue_lines),
            }

        yield json.dumps(result)

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(DialogueFormatterExtension)
