"""Content Formatter Emotion Detection Module.

Centralized emotion detection using deep_agent with structured output.

Usage:
    # Simple function-based API (recommended)
    from src.extensions.content.content_formatter.emotion import detect_emotions, DialogueLine

    dialogue_lines = [DialogueLine("EMMA", "Hello there!"), ...]
    detected_lines = await detect_emotions(dialogue_lines)
    # Returns List[DialogueLine] with .emotion set

    # Class-based API (for configuration)
    from src.extensions.content.content_formatter.emotion import EmotionDetector

    detector = EmotionDetector(config)
    detected_lines = await detector.detect(dialogue_lines)
"""

from .schema import (
    EmotionVector,
    DialogueLine,
    EmotionSegment,
    EmotionResponse,
)

from .detector import (
    EmotionDetector,
    detect_emotions,
    get_emotion_detector,
)

__all__ = [
    # Schema classes
    "EmotionVector",
    "DialogueLine",
    "EmotionSegment",
    "EmotionResponse",
    # Detector classes and functions
    "EmotionDetector",
    "detect_emotions",
    "get_emotion_detector",
]
