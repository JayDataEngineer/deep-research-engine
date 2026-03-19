"""Audio Producer Library - Smart audio workflow orchestration.

This library provides reusable workflows for audio production:
- Simple tagging: Text -> Tag -> Speak
- Podcast creation: Markdown -> Parse Speakers -> Tag -> Multi-Voice Speak -> Merge
- Audiobook reading: (Future)

Architecture:
- core/: Reusable components (merger, emotion detection via content_formatter)
- workflows/: Specific workflow strategies (simple_tagger, podcast_director)
- config_schemas.py: Pydantic models for configuration

Note: Emotion detection is now centralized in content_formatter.emotion module.
"""

from .config_schemas import (
    AudioJobConfig,
    EmotionConfig,
    IndexSamplingConfig,
    VoiceMap,
    load_config_from_yaml,
)
from .workflows import PodcastDirector, create_podcast_director

__all__ = [
    "AudioJobConfig",
    "EmotionConfig",
    "IndexSamplingConfig",
    "VoiceMap",
    "PodcastDirector",
    "create_podcast_director",
    "load_config_from_yaml",
]
