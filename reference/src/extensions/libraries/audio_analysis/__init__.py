"""
Audio Analysis Extensions.

Optional extensions for audio analysis including:
- speaker_diarization: Identify speakers and timestamps
- audio_comparison: Compare audio files for similarity/duplicates
"""

from src.extensions.analysis.speaker_diarization.extension import (
    register_extension as register_diarization,
)
from src.extensions.analysis.audio_comparison.extension import (
    register_extension as register_comparison,
)

__all__ = ["register_diarization", "register_comparison"]
