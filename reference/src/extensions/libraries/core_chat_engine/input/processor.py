"""
Input Processing Layer

Processes raw user input into structured format.
Detects special commands, OOC, multimodal content.

NOTE: SillyTavern special command detection is OPTIONAL.
The brain works without it - this is for enhanced RP mode.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from src.extensions.libraries.core_chat_engine.state.brain_state import ProcessedInput


# Optional SillyTavern special commands detection
_detect_special_commands = None
try:
    from ..sillytavern.special_commands import detect_special_commands as _st_detect
    _detect_special_commands = _st_detect
except ImportError:
    pass  # SillyTavern features not available


class InputProcessor:
    """Process raw user input into structured format."""

    def __init__(
        self,
        ooc_markers: List[str] = None,
        story_marker: str = ">",
        slash_command_prefix: str = "/",
    ):
        """
        Initialize input processor.

        Args:
            ooc_markers: Markers for out-of-character content (default: ["((", "))"])
            story_marker: Marker for story mode actions
            slash_command_prefix: Prefix for slash commands
        """
        self.ooc_markers = ooc_markers or ["((", "))"]
        self.story_marker = story_marker
        self.slash_command_prefix = slash_command_prefix

    def process(self, raw_input: str, context: Any = None) -> ProcessedInput:
        """
        Process raw user input into structured format.

        Args:
            raw_input: Raw user input text
            context: Optional conversation context

        Returns:
            ProcessedInput with structured data
        """
        # Initialize result
        result = ProcessedInput(raw=raw_input, clean=raw_input)

        # Check for story mode
        if raw_input.strip().startswith(self.story_marker):
            result.story_mode = True
            # Remove story marker for clean text
            result.clean = raw_input.strip()[1:].strip()

        # Check for OOC content
        ooc_content = self._extract_ooc(raw_input)
        if ooc_content:
            result.ooc_content = ooc_content
            # Remove OOC from clean text
            result.clean = self._remove_ooc(raw_input)

        # Check for slash commands
        if raw_input.strip().startswith(self.slash_command_prefix):
            result.commands = self._extract_commands(raw_input)
            # Don't modify clean for commands - they'll be processed separately

        # Detect special SillyTavern-style commands (OPTIONAL)
        if _detect_special_commands is not None:
            special_result = _detect_special_commands(raw_input)
            if special_result:
                result.metadata["special_commands"] = special_result

        # Extract mentions, keywords, etc.
        result.metadata["mentions"] = self._extract_mentions(raw_input)
        result.metadata["keywords"] = self._extract_keywords(raw_input)
        result.metadata["emotion_indicators"] = self._detect_emotion(raw_input)

        return result

    def _extract_ooc(self, text: str) -> Optional[str]:
        """Extract out-of-character content."""
        if len(self.ooc_markers) < 2:
            return None

        start_marker, end_marker = self.ooc_markers[0], self.ooc_markers[1]

        # Check for content wrapped in OOC markers
        pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Check if starts with OOC marker
        if text.strip().startswith(start_marker):
            # Find end of OOC (either explicit end marker or end of string)
            if end_marker in text:
                return text[text.index(start_marker) + len(start_marker):text.index(end_marker)].strip()
            else:
                return text[text.index(start_marker) + len(start_marker):].strip()

        return None

    def _remove_ooc(self, text: str) -> str:
        """Remove OOC content from text."""
        if len(self.ooc_markers) < 2:
            return text

        start_marker, end_marker = self.ooc_markers[0], self.ooc_markers[1]

        # Remove wrapped OOC content
        pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
        text = re.sub(pattern, "", text, flags=re.DOTALL)

        # Remove standalone OOC
        if text.strip().startswith(start_marker):
            if end_marker in text:
                text = text[text.index(end_marker) + len(end_marker):].strip()
            else:
                text = ""

        return text.strip()

    def _extract_commands(self, text: str) -> List[str]:
        """Extract slash commands from text."""
        commands = []

        # Split by whitespace and find commands
        words = text.strip().split()
        for word in words:
            if word.startswith(self.slash_command_prefix):
                commands.append(word)

        return commands

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        pattern = r"@(\w+)"
        return re.findall(pattern, text)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction (could be enhanced with NLP)
        # Remove common words and extract remaining terms
        common_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "can",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "this", "that", "these", "those", "my", "your", "his", "their",
            "what", "which", "who", "when", "where", "why", "how",
        }

        words = re.findall(r"\b\w+\b", text.lower())
        keywords = [w for w in words if w not in common_words and len(w) > 2]

        # Return unique keywords
        return list(set(keywords))

    def _detect_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotional indicators in text."""
        emotion_indicators = {
            "positive": 0.0,
            "negative": 0.0,
            "excitement": 0.0,
            "confusion": 0.0,
            "urgency": 0.0,
        }

        # Positive indicators
        positive_words = [
            "good", "great", "awesome", "happy", "love", "excellent",
            "thanks", "thank", "excited", "yay", "woohoo", ":)", ":D",
        ]
        for word in positive_words:
            if word.lower() in text.lower():
                emotion_indicators["positive"] += 1

        # Negative indicators
        negative_words = [
            "bad", "terrible", "awful", "hate", "angry", "sad", "upset",
            "frustrated", "annoying", "worst", "sucks", ":(", ":(",
        ]
        for word in negative_words:
            if word.lower() in text.lower():
                emotion_indicators["negative"] += 1

        # Excitement indicators
        excitement_words = ["!", "wow", "omg", "incredible", "amazing"]
        for word in excitement_words:
            if word in text.lower():
                emotion_indicators["excitement"] += 1

        # Confusion indicators
        confusion_words = ["?", "confused", "don't understand", "huh", "what"]
        for word in confusion_words:
            if word.lower() in text.lower():
                emotion_indicators["confusion"] += 1

        # Urgency indicators
        urgency_words = ["urgent", "asap", "immediately", "hurry", "quick", "help"]
        for word in urgency_words:
            if word.lower() in text.lower():
                emotion_indicators["urgency"] += 1

        # Normalize
        total = sum(emotion_indicators.values())
        if total > 0:
            for key in emotion_indicators:
                emotion_indicators[key] /= total

        return emotion_indicators


def create_input_processor(
    ooc_markers: List[str] = None,
    story_marker: str = ">",
) -> InputProcessor:
    """Factory function to create InputProcessor."""
    return InputProcessor(
        ooc_markers=ooc_markers,
        story_marker=story_marker,
    )
