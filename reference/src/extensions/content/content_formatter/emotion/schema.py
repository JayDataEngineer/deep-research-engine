"""Emotion Detection Schema - Pydantic models for structured emotion detection.

This module defines the data structures for emotion detection:
- EmotionVector: 8-dimensional emotion vector matching IndexTTS2 format
- DialogueLine: A dialogue line with optional emotion vector
- EmotionSegment: Pydantic model for deep_agent structured output
- EmotionResponse: Container for list of segments
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class EmotionVector:
    """An 8-dimensional emotion vector matching IndexTTS2 format.

    Format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

    Backwards compatibility: If 9 dimensions are provided, the intensity value (9th) is ignored.
    """

    # IndexTTS2 format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    DIMENSIONS = 8
    DIMENSION_NAMES = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
    NEUTRAL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # calm=1.0

    def __init__(self, values: List[float]):
        # Handle 9-dim input (old format with intensity) - drop the last element
        if len(values) == 9:
            values = values[:8]

        # Detect old format [anger, disgust, fear, happiness, sadness, surprise, neutral, intensity]
        # vs new format [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        if len(values) == 8:
            # Detect if this is old format by checking if 'neutral' (index 6) is high
            # Old format has neutral at position 6, new format has 'surprised' at position 6
            if values[6] > 0.5:  # High value at index 6 = old format's 'neutral'
                # Convert old format to new format
                # old: [anger, disgust, fear, happiness, sadness, surprise, neutral]
                # new: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
                old = values
                new = [
                    old[3],  # happy = happiness
                    old[0],  # angry = anger
                    old[4],  # sad = sadness
                    old[2],  # afraid = fear
                    old[1],  # disgusted = disgust
                    0.0,     # melancholic (no direct mapping)
                    old[5],  # surprised = surprise
                    old[6],  # calm = neutral
                ]
                values = new

        if len(values) != self.DIMENSIONS:
            raise ValueError(f"Emotion vector must have {self.DIMENSIONS} dimensions, got {len(values)}")
        self.values = [max(0.0, min(1.0, float(v))) for v in values]

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __repr__(self):
        return f"EmotionVector({self.values})"

    def to_list(self) -> List[float]:
        return self.values

    def to_indextts_format(self) -> List[float]:
        """Return IndexTTS2 format."""
        return self.values

    def to_index_tts_format(self) -> List[float]:
        """Alias for to_indextts_format() for backwards compatibility."""
        return self.values

    @classmethod
    def neutral(cls) -> "EmotionVector":
        return cls(cls.NEUTRAL)

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionVector":
        """Create from dictionary with dimension names."""
        values = [data.get(name, 0.0) for name in cls.DIMENSION_NAMES]
        return cls(values)


class DialogueLine:
    """A dialogue line for emotion detection."""

    def __init__(
        self,
        speaker: str,
        text: str,
        direction: str = "",
        section_id: int = 0
    ):
        self.speaker = speaker
        self.text = text
        self.direction = direction
        self.emotion: Optional[EmotionVector] = None
        self.section_id: int = section_id  # For tracking SEGMENT_BREAK sections

    def __repr__(self):
        emo_str = f", emotion={self.emotion}" if self.emotion else ""
        return f"DialogueLine(speaker={self.speaker!r}, text={self.text[:30]!r}...{emo_str})"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker": self.speaker,
            "text": self.text,
            "direction": self.direction,
            "emotion": self.emotion.to_list() if self.emotion else None,
            "section_id": self.section_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DialogueLine":
        """Create from dictionary."""
        line = cls(
            data["speaker"],
            data["text"],
            data.get("direction", ""),
            data.get("section_id", 0)
        )
        if data.get("emotion"):
            line.emotion = EmotionVector(data["emotion"])
        return line


# ============================================================================
# Pydantic models for deep_agent structured output
# ============================================================================

class EmotionSegment(BaseModel):
    """A single segment with emotion tagging for deep_agent output."""

    speaker: str = Field(description="Speaker name (e.g., EMMA, OLIVIA, NARRATOR)")
    text: str = Field(description="The dialogue text for this segment")
    emotion: List[float] = Field(
        description="8-dimensional emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]"
    )

    @field_validator('emotion')
    @classmethod
    def validate_emotion(cls, v):
        """Validate emotion vector has exactly 8 dimensions, each 0.0-1.0."""
        if len(v) != 8:
            raise ValueError(f"Emotion vector must have 8 dimensions, got {len(v)}")
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Emotion dimension {i} must be a number, got {type(val)}")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Emotion dimension {i} must be between 0.0 and 1.0, got {val}")
        return v


class EmotionResponse(BaseModel):
    """Complete response from deep agent with all segments."""

    segments: List[EmotionSegment] = Field(
        description="List of dialogue segments with emotion vectors"
    )
