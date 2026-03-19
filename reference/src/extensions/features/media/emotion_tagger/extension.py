"""Emotion Tagger Extension

Analyzes emotions in text using LLM (host environment) for TTS processing.
This extension runs in the host environment to access the main project's LLM.
"""

import asyncio
import httpx
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.extensions.core.base import LogicExtension
from src.shared.factory import get_llm

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class EmotionVector:
    """8-dimensional emotion vector for IndexTTS."""
    valence: float = 0.5      # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.5      # 0.0 (calm) to 1.0 (excited)
    dominance: float = 0.5    # 0.0 (submissive) to 1.0 (dominant)
    joy: float = 0.0          # 0.0 to 1.0
    sadness: float = 0.0      # 0.0 to 1.0
    anger: float = 0.0        # 0.0 to 1.0
    fear: float = 0.0         # 0.0 to 1.0
    surprise: float = 0.0     # 0.0 to 1.0

    def __init__(
        self,
        valence: float = 0.5,
        arousal: float = 0.5,
        dominance: float = 0.5,
        joy: float = 0.0,
        sadness: float = 0.0,
        anger: float = 0.0,
        fear: float = 0.0,
        surprise: float = 0.0,
    ):
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
        self.joy = joy
        self.sadness = sadness
        self.anger = anger
        self.fear = fear
        self.surprise = surprise

    def to_list(self) -> List[float]:
        """Convert to list for API calls."""
        return [
            self.valence,
            self.arousal,
            self.dominance,
            self.joy,
            self.sadness,
            self.anger,
            self.fear,
            self.surprise,
        ]

    @classmethod
    def from_list(cls, values: List[float]) -> 'EmotionVector':
        """Create from list of 8 floats."""
        if len(values) != 8:
            raise ValueError(f"EmotionVector requires 8 values, got {len(values)}")
        return cls(*values)


class SentenceEmotion:
    """Emotion analysis result for a single sentence."""
    text: str
    emotion: EmotionVector
    valence: float
    arousal: float
    description: str


# =============================================================================
# Emotion Tagger Extension
# =============================================================================

class EmotionTaggerExtension(LogicExtension):
    """Emotion tagger extension using host LLM.

    This extension runs in the host environment to access the main project's
    LLM configuration for emotion analysis. The IndexTTS extension (isolated
    environment) calls this extension for emotion analysis.
    """

    EMOTION_ANALYSIS_PROMPT = """Analyze the emotional tone of this text and provide an 8-dimensional emotion vector.

Text: "{text}"

Context: This is from a {context_type}. {context_info}

Provide emotion values as JSON with these keys:
- valence: -1.0 (very negative) to 1.0 (very positive)
- arousal: 0.0 (very calm) to 1.0 (very excited)
- dominance: 0.0 (very submissive) to 1.0 (very dominant)
- joy: 0.0 to 1.0 (intensity of joy)
- sadness: 0.0 to 1.0 (intensity of sadness)
- anger: 0.0 to 1.0 (intensity of anger)
- fear: 0.0 to 1.0 (intensity of fear)
- surprise: 0.0 to 1.0 (intensity of surprise)

Respond ONLY with valid JSON, no markdown."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._llm = None
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self, context):
        """Initialize the emotion tagger with LLM from host environment."""
        await super().initialize(context)
        # Get LLM factory from context
        self._llm = await get_llm()
        logger.info("Emotion tagger extension initialized with host LLM")

    async def shutdown(self):
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def analyze_emotions(
        self,
        text: str,
        context_type: str = "conversation",
        context_info: str = "",
    ) -> dict:
        """Analyze emotions in text, returning sentence-by-sentence results.

        Args:
            text: Text to analyze
            context_type: Type of content (conversation, narrative, etc.)
            context_info: Additional context

        Returns:
            Dict with 'sentences' list and 'overall_emotion' list
        """
        # Split text into sentences
        sentences = self._split_sentences(text)

        sentence_emotions = []
        overall_vals = [0.0] * 8

        for sentence in sentences:
            emotion = await self._analyze_sentence(
                sentence,
                context_type,
                context_info,
            )

            emotion_list = emotion.to_list()
            sentence_emotions.append(SentenceEmotion(
                text=sentence,
                emotion=emotion,
                valence=emotion.valence,
                arousal=emotion.arousal,
                description=self._describe_emotion(emotion),
            ))

            # Accumulate for average
            for i, val in enumerate(emotion_list):
                overall_vals[i] += val

        # Calculate overall average
        if sentences:
            overall_vals = [v / len(sentences) for v in overall_vals]

        return {
            "sentences": [
                {
                    "sentence": se.text,
                    "emotion": se.emotion.to_list(),
                    "valence": se.valence,
                    "arousal": se.arousal,
                    "description": se.description,
                }
                for se in sentence_emotions
            ],
            "overall_emotion": overall_vals,
        }

    async def _analyze_sentence(
        self,
        text: str,
        context_type: str,
        context_info: str,
    ) -> EmotionVector:
        """Analyze emotion of a single sentence using LLM.

        Args:
            text: Text to analyze
            context_type: Type of content
            context_info: Additional context

        Returns:
            EmotionVector with 8 dimensions
        """
        # Build prompt
        prompt = self.EMOTION_ANALYSIS_PROMPT.format(
            text=text,
            context_type=context_type,
            context_info=context_info,
        )

        try:
            # Use the LLM from host environment
            # We'll call the LLM via the chat completions API
            response = await self._llm.ainvoke([
                {"role": "user", "content": prompt}
            ])

            # Parse response
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.split("```")[0].strip()

            emotion_data = json.loads(content)

            return EmotionVector(
                valence=emotion_data.get("valence", 0.0),
                arousal=emotion_data.get("arousal", 0.5),
                dominance=emotion_data.get("dominance", 0.5),
                joy=emotion_data.get("joy", 0.0),
                sadness=emotion_data.get("sadness", 0.0),
                anger=emotion_data.get("anger", 0.0),
                fear=emotion_data.get("fear", 0.0),
                surprise=emotion_data.get("surprise", 0.0),
            )

        except Exception as e:
            logger.warning(f"[EMOTION ANALYSIS] Failed to analyze emotion: {e}")
            # Return neutral emotion on failure
            return EmotionVector()

    def _split_sentences(self, text: str, max_length: int = 200) -> List[str]:
        """Split text into sentences."""
        # Sentence-ending punctuation
        SENTENCE_ENDINGS = {'.', '!', '?', '。', '！', '？', '…', '...'}
        # Clause boundaries for better emotion detection
        CLAUSE_SEPARATORS = {',', ';', ':', '，', '；', '：', '—', '–'}

        # Clean up text
        text = text.strip()
        if not text:
            return []

        sentences = []
        current = ""

        # Split by sentence endings
        i = 0
        while i < len(text):
            char = text[i]
            current += char

            # Check for sentence ending
            if char in SENTENCE_ENDINGS:
                # Look ahead for ellipsis or multiple endings
                if i + 1 < len(text) and text[i + 1] in SENTENCE_ENDINGS:
                    continue  # Part of multi-char ending

                sentence = current.strip()
                if sentence:
                    # Split long sentences
                    if len(sentence) > max_length:
                        sentences.extend(self._split_long(sentence, max_length))
                    else:
                        sentences.append(sentence)
                current = ""

            i += 1

        # Add remaining text
        if current.strip():
            remaining = current.strip()
            if len(remaining) > max_length:
                sentences.extend(self._split_long(remaining, max_length))
            else:
                sentences.append(remaining)

        return sentences

    def _split_long(self, text: str, max_length: int) -> List[str]:
        """Split a long sentence by clause boundaries."""
        CLAUSE_SEPARATORS = {',', ';', ':', '，', '；', '：', '—', '–'}

        parts = []
        current = ""

        for char in text:
            current += char

            if char in CLAUSE_SEPARATORS and len(current) > max_length // 2:
                parts.append(current.strip())
                current = ""

            if len(current) >= max_length:
                parts.append(current.strip())
                current = ""

        if current.strip():
            parts.append(current.strip())

        return parts

    def _describe_emotion(self, emotion: EmotionVector) -> str:
        """Generate human-readable emotion description."""
        parts = []

        # Valence
        if emotion.valence > 0.5:
            parts.append("positive")
        elif emotion.valence < -0.5:
            parts.append("negative")

        # Arousal
        if emotion.arousal > 0.7:
            parts.append("excited")
        elif emotion.arousal < 0.3:
            parts.append("calm")

        # Specific emotions (highest intensity)
        emotions = [
            ("joyful", emotion.joy),
            ("sad", emotion.sadness),
            ("angry", emotion.anger),
            ("fearful", emotion.fear),
            ("surprised", emotion.surprise),
        ]
        emotions.sort(key=lambda x: x[1], reverse=True)

        if emotions[0][1] > 0.5:
            parts.append(emotions[0][0])

        return " ".join(parts) if parts else "neutral"

    # The process method for JobExtension compatibility
    async def process(self, input_data: dict, **kwargs):
        """
        Process text to add emotion tags.

        Args:
            input_data: Dict with 'text' key
            **kwargs: Additional parameters like context_type

        Yields:
            Tagged text or emotion analysis results
        """
        text = input_data.get('text', '')
        context_type = kwargs.get('context_type', 'conversation')

        result = await self.analyze_emotions(text, context_type)

        # Yield result as JSON string
        import json
        yield json.dumps(result)

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(EmotionTaggerExtension)
