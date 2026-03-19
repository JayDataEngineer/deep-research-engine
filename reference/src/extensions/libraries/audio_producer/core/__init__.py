"""Core components for Audio Producer Library.

These are reusable components that can be used by any workflow:
- Emotion detection (re-exported from content_formatter)
- merger: Audio file combining

NOTE: Emotion detection is now centralized in content_formatter.emotion module.
This module re-exports for backwards compatibility.
"""

# Re-export emotion detection from centralized module
from src.extensions.content.content_formatter.emotion import (
    DialogueLine,
    EmotionDetector,
    EmotionVector,
    detect_emotions,
    get_emotion_detector,
)

from .merger import AudioMerger

__all__ = [
    # Emotion detection (from content_formatter)
    "DialogueLine",
    "EmotionDetector",
    "EmotionVector",
    "detect_emotions",
    "get_emotion_detector",
    # Audio processing
    "AudioMerger",
]
