"""
Emotion Tagger Service - AI-powered emotion detection and tagging.

Uses GLM-4.7 via existing LLM factory to analyze text and
suggest emotion tags for expressive TTS.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.shared.factory import get_llm
from src.constants import ZAI_DEFAULT_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class EmotionTag:
    """An emotion tag with position and intensity."""
    emotion: str
    position: int  # Character position in text
    intensity: float = 1.0
    reason: str = ""  # Why this emotion was suggested


@dataclass
class TaggedText:
    """Text with emotion tags applied."""
    original: str
    tagged: str
    tags: list[EmotionTag]


# Available emotions
EMOTIONS = {
    "neutral": "Default speaking style, no particular emotion",
    "happy": "Cheerful, upbeat, positive tone",
    "sad": "Melancholic, somber, downcast tone",
    "angry": "Aggressive, intense, frustrated tone",
    "surprise": "Shocked, amazed, unexpected tone",
    "fear": "Anxious, scared, nervous tone",
    "disgust": "Revolted, repulsed tone",
    "excited": "Eager, enthusiastic, energetic tone",
    "calm": "Peaceful, serene, relaxed tone",
    "confident": "Self-assured, certain, assertive tone",
    "gentle": "Soft, tender, compassionate tone",
    "serious": "Grave, earnest, solemn tone",
}


class EmotionTagger:
    """
    AI-powered emotion tagger using GLM-4.7.

    Analyzes text to suggest appropriate emotion tags for expressive TTS.
    Uses existing LLM factory.
    """

    def __init__(self):
        """Initialize the emotion tagger with existing LLM factory."""
        # LLM will be initialized lazily on first use
        self._llm = None

    async def _get_llm(self):
        """Get LLM instance (lazy initialization)."""
        if self._llm is None:
            self._llm = await get_llm(temperature=0.3)
        return self._llm

    async def suggest_emotions(
        self,
        text: str,
        max_suggestions: int = 5
    ) -> list[EmotionTag]:
        """
        Suggest emotion tags for the given text using GLM-4.7.

        Args:
            text: Text to analyze
            max_suggestions: Maximum number of emotion suggestions

        Returns:
            List of EmotionTag suggestions
        """
        if not text or not text.strip():
            return []

        try:
            return await self._ai_suggest_emotions(text, max_suggestions)
        except Exception as e:
            logger.error(f"AI emotion tagging FAILED: {e}")
            raise

    async def _ai_suggest_emotions(
        self,
        text: str,
        max_suggestions: int
    ) -> list[EmotionTag]:
        """Use existing LLM to suggest emotion tags."""
        # SIMPLIFIED: Just ask for the main emotion
        emotions_list = ", ".join(EMOTIONS.keys())

        prompt = f"""Analyze the emotion of this text and respond with ONLY one emotion name from this list:
{emotions_list}

Text: \"{text[:200]}\"

Respond with ONLY the emotion name, nothing else."""

        # Use existing LLM factory
        from langchain_core.messages import HumanMessage

        llm = await self._get_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        emotion = response.content.strip().lower()

        # Validate emotion
        if emotion not in EMOTIONS:
            emotion = "neutral"

        # Return single tag
        return [
            EmotionTag(
                emotion=emotion,
                position=0,
                intensity=1.0,
                reason=f"Detected {emotion} emotion in text"
            )
        ]


    def _rule_based_suggest_emotions(self, text: str) -> list[EmotionTag]:
        """Fallback rule-based emotion detection."""
        tags = []
        text_lower = text.lower()

        # Emotion keywords
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "wonderful", "great", "love", "smile", "laugh"],
            "sad": ["sad", "sorry", "cry", "miss", "loss", "grief", "tragic", "heartbreak"],
            "angry": ["angry", "furious", "hate", "rage", "mad", "outrage"],
            "surprise": ["surprise", "shock", "amazing", "unbelievable", "wow", "sudden"],
            "fear": ["scared", "afraid", "fear", "terrified", "horror", "anxious"],
            "excited": ["excited", "thrilled", "can't wait", "eager", "looking forward"],
            "calm": ["calm", "peaceful", "relax", "serene", "tranquil"],
            "confident": ["confident", "certain", "sure", "believe", "know"],
        }

        # Check for emotion keywords
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                pos = text_lower.find(keyword)
                if pos != -1:
                    tags.append(EmotionTag(
                        emotion=emotion,
                        position=pos,
                        intensity=1.0,
                        reason=f"Found keyword: {keyword}"
                    ))
                    break  # Only add once per emotion

        # Sort by position
        tags.sort(key=lambda t: t.position)

        return tags

    def apply_tags(self, text: str, tags: list[EmotionTag]) -> TaggedText:
        """
        Apply emotion tags to text.

        Args:
            text: Original text
            tags: List of emotion tags to apply

        Returns:
            TaggedText with tagged version
        """
        # Sort tags by position (reverse) to insert from end
        sorted_tags = sorted(tags, key=lambda t: t.position, reverse=True)

        # Insert tags into text
        tagged_text = text
        for tag in sorted_tags:
            pos = min(tag.position, len(tagged_text))
            # Format: [emotion] or [emotion:intensity]
            if tag.intensity != 1.0:
                tag_str = f"[{tag.emotion}:{tag.intensity}]"
            else:
                tag_str = f"[{tag.emotion}]"
            tagged_text = tagged_text[:pos] + tag_str + " " + tagged_text[pos:]

        return TaggedText(
            original=text,
            tagged=tagged_text.strip(),
            tags=sorted(tags, key=lambda t: t.position)
        )

    def remove_tags(self, text: str) -> str:
        """
        Remove all emotion tags from text.

        Args:
            text: Text with emotion tags

        Returns:
            Clean text without tags
        """
        # Remove [emotion] and [emotion:intensity] tags
        cleaned = re.sub(r'\[\w+(:\d+\.?\d*)?\]', '', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def parse_tags(self, text: str) -> list[EmotionTag]:
        """
        Parse emotion tags from text.

        Args:
            text: Text with emotion tags

        Returns:
            List of parsed EmotionTag objects
        """
        tags = []
        pattern = r'\[(\w+)(?::(\d+\.?\d*))?\]'

        for match in re.finditer(pattern, text):
            emotion = match.group(1)
            intensity = float(match.group(2)) if match.group(2) else 1.0

            # Calculate position (before this tag in original text)
            # Subtract count of tags from position
            tags_before = len(re.findall(pattern, text[:match.start()]))
            position = match.start() - (tags_before * 10)  # Approximate

            tags.append(EmotionTag(
                emotion=emotion,
                position=max(0, position),
                intensity=intensity
            ))

        return tags

    async def auto_tag_text(
        self,
        text: str,
        max_tags: int = 3
    ) -> TaggedText:
        """
        Automatically tag text with suggested emotions.

        Args:
            text: Text to tag
            max_tags: Maximum number of tags to add

        Returns:
            TaggedText with applied tags
        """
        # Get suggestions
        suggestions = await self.suggest_emotions(text, max_tags)

        # Apply tags
        return self.apply_tags(text, suggestions)

    def get_available_emotions(self) -> dict:
        """Get dictionary of available emotions."""
        return EMOTIONS.copy()


# Singleton instance (lazy initialization)
_emotion_tagger_instance: Optional["EmotionTagger"] = None


def get_emotion_tagger() -> "EmotionTagger":
    """Get or create the singleton EmotionTagger instance."""
    global _emotion_tagger_instance
    if _emotion_tagger_instance is None:
        _emotion_tagger_instance = EmotionTagger()
    return _emotion_tagger_instance


async def auto_tag_text(text: str, max_tags: int = 3) -> TaggedText:
    """Convenience function to auto-tag text."""
    return await get_emotion_tagger().auto_tag_text(text, max_tags)


async def suggest_emotions(text: str, max_suggestions: int = 5) -> list[EmotionTag]:
    """Convenience function to get emotion suggestions."""
    return await get_emotion_tagger().suggest_emotions(text, max_suggestions)
