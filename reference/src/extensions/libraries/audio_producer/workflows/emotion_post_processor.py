"""Emotion Post-Processor for Podcast Generation.

This module implements mathematical transformations on emotion vectors before they're
sent to IndexTTS. These transformations fix common issues like:
- "Constant shouting" (LLM outputs 0.8+ for moderate emotions)
- Jarring emotional transitions
- Inconsistent character voices
- Extreme values that cause audio distortion

Example usage:
    from podcast_config import PRESET_NATURAL
    from emotion_post_processor import EmotionPostProcessor

    config = PRESET_NATURAL
    processor = EmotionPostProcessor(config.post_processing)

    # Process a single emotion vector
    processed = processor.process(
        emotions=[0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2, 0.9],
        speaker="EMMA",
        text="I'm very annoyed right now.",
        previous_emotions=[0.7, 0.0, 0.4, 0.0, 0.0, 0.0, 0.1, 0.8]
    )
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import math

from .podcast_config import (
    PostProcessorConfig,
    EmotionSmoothingConfig,
    EmotionConsistencyConfig,
    EmotionWeightingConfig,
    EmotionInterpolationConfig,
    EmotionClampingConfig,
    SceneAwareEmotionConfig,
    DynamicsCompressionConfig,
    CharacterIntensityReducerConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Emotion Vector Dimensions (VAD-based 8-dimensional system)
# =============================================================================

EMOTION_DIMENSIONS = [
    "valence",      # Positive/negative (-1 to 1)
    "arousal",      # Calm/energetic (0 to 1)
    "dominance",    # Submissive/dominant (0 to 1)
    "joy",          # Joy (0 to 1)
    "sadness",      # Sadness (0 to 1)
    "anger",        # Anger (0 to 1)
    "fear",         # Fear (0 to 1)
    "surprise",     # Surprise (0 to 1)
]


@dataclass
class ProcessingState:
    """State maintained across emotion vector processing."""
    previous_vector: Optional[List[float]] = None
    smoothing_window: deque = field(default_factory=lambda: deque(maxlen=10))  # For window-based smoothing
    character_history: Dict[str, List[List[float]]] = field(default_factory=dict)  # Per-character emotion history


# =============================================================================
# Individual Post-Processor Implementations
# =============================================================================

class DynamicsCompressionProcessor:
    """Apply power curve (gamma correction) to prevent "constant shouting".

    LLMs tend to output binary emotions: 0.0 for neutral, 0.8+ for anything else.
    Human speech hovers in the middle (0.3-0.7). This power curve suppresses
    mid-range values while preserving true peaks.

    Formula: y = x^gamma
    - gamma > 1: Suppresses mid-range, preserves peaks (prevents shouting)
    - gamma < 1: Boosts quiet emotions
    - gamma = 1: No change
    """

    def __init__(self, config: DynamicsCompressionConfig):
        self.gamma = config.power_curve
        self.max_cap = config.max_intensity
        self.respect_exclamation = config.respect_exclamation
        self.exclamation_threshold = config.exclamation_threshold

    def process(self, emotions: List[float], text: str = "", **kwargs) -> List[float]:
        """Apply power curve transformation.

        Args:
            emotions: Raw emotion vector [8 dimensions]
            text: Associated text (for exclamation detection)

        Returns:
            Transformed emotion vector
        """
        import numpy as np

        vector = np.array(emotions)
        is_exclamation = False

        # Check for exclamation marks if enabled
        if self.respect_exclamation and text:
            exclam_count = text.count("!")
            is_exclamation = exclam_count >= self.exclamation_threshold

        # Apply power curve (unless it's a true exclamation/scream)
        if not is_exclamation:
            # This pushes 0.7 down to ~0.49, but keeps 1.0 near 1.0
            vector = np.power(vector, self.gamma)

        # Hard clamp to prevent audio distortion
        vector = np.clip(vector, 0.0, self.max_cap)

        if is_exclamation:
            logger.debug(f"Exclamation detected, bypassing compression: {text[:30]}")

        return np.round(vector, 3).tolist()


class EmotionSmoothingProcessor:
    """Reduce sudden emotion changes between segments.

    Uses Exponential Moving Average (EMA) for smooth transitions:
    smoothed = alpha * current + (1 - alpha) * previous

    Or linear weighted average over a window.
    """

    def __init__(self, config: EmotionSmoothingConfig):
        self.alpha = config.alpha  # Smoothing factor (0-1)
        self.method = config.method
        self.window_size = config.window_size
        self.state = ProcessingState()

    def process(self, emotions: List[float], **kwargs) -> List[float]:
        """Apply smoothing transformation.

        Args:
            emotions: Raw emotion vector [8 dimensions]

        Returns:
            Smoothed emotion vector
        """
        import numpy as np

        current_vector = np.array(emotions)

        if self.method == "exponential_moving_average":
            # EMA: Blend with previous state
            if self.state.previous_vector is not None:
                prev = np.array(self.state.previous_vector)
                smoothed = (self.alpha * current_vector) + ((1 - self.alpha) * prev)
            else:
                smoothed = current_vector

            self.state.previous_vector = smoothed.tolist()

        elif self.method == "linear_weighted":
            # Linear weighted average over window
            self.state.smoothing_window.append(current_vector)

            if len(self.state.smoothing_window) < 2:
                smoothed = current_vector
            else:
                # Weighted average (more recent = higher weight)
                weights = list(range(1, len(self.state.smoothing_window) + 1))
                window_array = np.array(list(self.state.smoothing_window))
                smoothed = np.average(window_array, axis=0, weights=weights)

        else:  # kalman_filter (simplified version)
            # Simplified Kalman-like filter
            if self.state.previous_vector is not None:
                prev = np.array(self.state.previous_vector)
                # Estimate and update with measurement noise
                Q = 0.1  # Process noise
                R = 0.2  # Measurement noise
                smoothed = (1 - Q) * prev + Q * current_vector
                # Blend with previous estimate
                smoothed = (1 - R) * smoothed + R * current_vector
            else:
                smoothed = current_vector

            self.state.previous_vector = smoothed.tolist()

        return np.round(smoothed, 3).tolist()


class EmotionConsistencyProcessor:
    """Maintain character-specific emotional baselines.

    Each character has a baseline emotional profile. Deviations are allowed
    but limited to keep characters sounding distinct.
    """

    def __init__(self, config: EmotionConsistencyConfig):
        self.character_profiles = config.character_profiles
        self.max_deviation = config.max_deviation
        self.state = ProcessingState()

    def process(self, emotions: List[float], speaker: str = "UNKNOWN", **kwargs) -> List[float]:
        """Apply consistency constraints.

        Args:
            emotions: Raw emotion vector [8 dimensions]
            speaker: Character name

        Returns:
            Constrained emotion vector
        """
        import numpy as np

        if speaker not in self.character_profiles:
            # No profile for this character, return as-is
            return emotions

        profile = self.character_profiles[speaker]
        baseline = np.array(profile["baseline"])
        variance = profile.get("variance", self.max_deviation)

        current = np.array(emotions)

        # Calculate deviation from baseline
        deviation = np.abs(current - baseline)

        # Pull back towards baseline if over max deviation
        correction = np.where(deviation > self.max_deviation, 1.0, 0.0)
        corrected = current - (correction * (deviation - self.max_deviation) * np.sign(current - baseline))

        # Clamp to valid range
        corrected = np.clip(corrected, 0.0, 1.0)

        return np.round(corrected, 3).tolist()


class EmotionWeightingProcessor:
    """Adjust emotion intensity for specific effects.

    Apply multipliers to specific dimensions or overall intensity.
    """

    def __init__(self, config: EmotionWeightingConfig):
        self.intensity_multiplier = config.intensity_multiplier
        self.dimension_weights = config.dimension_weights

    def process(self, emotions: List[float], **kwargs) -> List[float]:
        """Apply intensity adjustments.

        Args:
            emotions: Raw emotion vector [8 dimensions]

        Returns:
            Weighted emotion vector
        """
        import numpy as np

        vector = np.array(emotions)

        # Apply overall intensity multiplier
        vector = vector * self.intensity_multiplier

        # Apply dimension-specific weights
        for i, dim in enumerate(EMOTION_DIMENSIONS):
            if dim in self.dimension_weights:
                vector[i] = vector[i] * self.dimension_weights[dim]

        # Clamp to valid range
        vector = np.clip(vector, 0.0, 1.0)

        return np.round(vector, 3).tolist()


class EmotionInterpolationProcessor:
    """Generate intermediate emotion steps for smooth transitions.

    This increases the number of segments for very smooth emotional transitions.
    Note: This would require TTS to support variable-length outputs or
    splitting dialogue text. For now, this is a placeholder.
    """

    def __init__(self, config: EmotionInterpolationConfig):
        self.steps = config.steps
        self.method = config.method
        self.state = ProcessingState()

    def process(self, emotions: List[float], **kwargs) -> List[float]:
        """Note: Interpolation requires modifying the segment structure.

        This is a simplified version that just returns the input.
        Full implementation would require modifying the dialogue line structure.
        """
        # Placeholder - full interpolation would require generating
        # additional dialogue segments which is beyond current scope
        return emotions


class EmotionClampingProcessor:
    """Limit emotion values to specific ranges.

    Prevents extreme emotions that may cause audio distortion or unwanted effects.
    """

    def __init__(self, config: EmotionClampingConfig):
        self.min_valence = config.min_valence
        self.max_valence = config.max_valence
        self.min_arousal = config.min_arousal
        self.max_arousal = config.max_arousal

    def process(self, emotions: List[float], **kwargs) -> List[float]:
        """Apply clamping constraints.

        Args:
            emotions: Raw emotion vector [8 dimensions]

        Returns:
            Clamped emotion vector
        """
        import numpy as np

        vector = np.array(emotions)

        # Clamp valence (index 0)
        vector[0] = np.clip(vector[0], self.min_valence, self.max_valence)

        # Clamp arousal (index 1)
        vector[1] = np.clip(vector[1], self.min_arousal, self.max_arousal)

        # Note: Other dimensions are 0-1 by default, no clamping needed

        return np.round(vector, 3).tolist()


class SceneAwareEmotionProcessor:
    """Adjust emotions based on scene type.

    Detects scene type from text patterns and applies appropriate adjustments.
    This is a simplified rule-based version.
    """

    def __init__(self, config: SceneAwareEmotionConfig):
        self.scene_rules = config.scene_rules
        self.auto_detect = config.auto_detect

    def process(self, emotions: List[float], text: str = "", **kwargs) -> List[float]:
        """Apply scene-aware adjustments.

        Args:
            emotions: Raw emotion vector [8 dimensions]
            text: Dialogue text for scene detection

        Returns:
            Adjusted emotion vector
        """
        if not self.auto_detect or not text:
            return emotions

        import numpy as np

        vector = np.array(emotions)

        # Simple scene detection based on text patterns
        text_lower = text.lower()

        # Action scene (aggressive words, exclamation marks)
        if any(word in text_lower for word in ["fight", "run", "hit", "shot", "kill", "attack"]):
            scene_type = "action"
        # Romance scene
        elif any(word in text_lower for word in ["love", "kiss", "heart", "feel", "embrace"]):
            scene_type = "romance"
        # Mystery scene
        elif any(word in text_lower for word in ["strange", "mystery", "weird", "suspect", "clue"]):
            scene_type = "mystery"
        else:
            scene_type = "exposition"

        # Apply scene-specific rules
        if scene_type in self.scene_rules:
            rules = self.scene_rules[scene_type]
            for dim, multiplier in rules.items():
                if dim in EMOTION_DIMENSIONS:
                    idx = EMOTION_DIMENSIONS.index(dim)
                    vector[idx] = vector[idx] * multiplier

        # Clamp to valid range
        vector = np.clip(vector, 0.0, 1.0)

        return np.round(vector, 3).tolist()


class CharacterIntensityReducerProcessor:
    """Reduce emotion intensity for specific characters.

    Some speakers tend to be too emotional or "shout" constantly.
    This applies a per-character intensity reduction.
    """

    def __init__(self, config: CharacterIntensityReducerConfig):
        self.character_multipliers = config.character_multipliers
        self.dimension_reduction = config.dimension_reduction

    def process(self, emotions: List[float], speaker: str = "UNKNOWN", **kwargs) -> List[float]:
        """Apply character-specific intensity reduction.

        Args:
            emotions: Raw emotion vector [8 dimensions]
            speaker: Character name

        Returns:
            Reduced emotion vector
        """
        import numpy as np

        vector = np.array(emotions)

        # Apply overall character multiplier if defined
        if speaker in self.character_multipliers:
            multiplier = self.character_multipliers[speaker]
            vector = vector * multiplier
            logger.debug(f"Applied {multiplier}x intensity reduction for {speaker}")

        # Apply dimension-specific reduction if defined
        if speaker in self.dimension_reduction:
            dims = self.dimension_reduction[speaker]
            for dim, reduction in dims.items():
                if dim in EMOTION_DIMENSIONS:
                    idx = EMOTION_DIMENSIONS.index(dim)
                    vector[idx] = vector[idx] * reduction

        # Clamp to valid range
        vector = np.clip(vector, 0.0, 1.0)

        return np.round(vector, 3).tolist()


# =============================================================================
# Main Post-Processor Pipeline
# =============================================================================

class EmotionPostProcessor:
    """Main post-processor that applies all transformations in sequence.

    Processes emotion vectors through a pipeline of post-processing steps.
    """

    def __init__(self, configs: List[PostProcessorConfig]):
        """Initialize the post-processor pipeline.

        Args:
            configs: List of post-processor configurations (applied in order)
        """
        self.processors = []
        self.configs = configs

        for config in configs:
            processor = self._create_processor(config)
            if processor:
                self.processors.append(processor)
                logger.info(f"Added post-processor: {config.name}")

        if not self.processors:
            logger.info("No post-processors configured")

    def _create_processor(self, config: PostProcessorConfig):
        """Create a processor instance from config."""
        if isinstance(config, DynamicsCompressionConfig):
            return DynamicsCompressionProcessor(config)
        elif isinstance(config, EmotionSmoothingConfig):
            return EmotionSmoothingProcessor(config)
        elif isinstance(config, EmotionConsistencyConfig):
            return EmotionConsistencyProcessor(config)
        elif isinstance(config, EmotionWeightingConfig):
            return EmotionWeightingProcessor(config)
        elif isinstance(config, EmotionInterpolationConfig):
            return EmotionInterpolationProcessor(config)
        elif isinstance(config, EmotionClampingConfig):
            return EmotionClampingProcessor(config)
        elif isinstance(config, SceneAwareEmotionConfig):
            return SceneAwareEmotionProcessor(config)
        elif isinstance(config, CharacterIntensityReducerConfig):
            return CharacterIntensityReducerProcessor(config)
        else:
            logger.warning(f"Unknown post-processor type: {type(config)}")
            return None

    def process(
        self,
        emotions: List[float],
        speaker: str = "UNKNOWN",
        text: str = "",
        previous_emotions: Optional[List[float]] = None
    ) -> List[float]:
        """Process emotion vector through the entire pipeline.

        Args:
            emotions: Raw emotion vector [8 dimensions]
            speaker: Character name (for consistency processing)
            text: Dialogue text (for scene detection, exclamation detection)
            previous_emotions: Previous emotion vector (for context-aware processing)

        Returns:
            Processed emotion vector
        """
        current = emotions.copy()

        # Apply each processor in sequence
        for i, processor in enumerate(self.processors):
            try:
                current = processor.process(
                    current,
                    speaker=speaker,
                    text=text,
                    previous_emotions=previous_emotions
                )
            except Exception as e:
                logger.error(f"Post-processor {i} ({type(processor).__name__}) failed: {e}")
                # Continue with unprocessed value

        return current

    def reset(self):
        """Reset all processor states (e.g., for new podcast)."""
        for processor in self.processors:
            if hasattr(processor, 'state'):
                processor.state = ProcessingState()
        logger.info("Post-processor states reset")


# =============================================================================
# Utility Functions
# =============================================================================

def create_processor_from_preset(preset_name: str) -> EmotionPostProcessor:
    """Create a post-processor from a preset name.

    Args:
        preset_name: Name of the preset (quality, speed, balanced, story, natural, dramatic)

    Returns:
        Configured EmotionPostProcessor instance
    """
    from .podcast_config import get_preset

    config = get_preset(preset_name)
    return EmotionPostProcessor(config.post_processing)


def demo_processing():
    """Demonstrate the effect of different post-processing configurations."""
    import numpy as np

    print("\n" + "=" * 60)
    print("Emotion Post-Processor Demo")
    print("=" * 60)

    # Sample "angry" emotions that LLM might output
    raw_emotions = [0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2, 0.9]

    print(f"\nRaw LLM Output (annoyed): {raw_emotions}")
    print("(High values sound like shouting in TTS)\n")

    # Test different processors
    configs = [
        ("Dynamics Compression (Natural)", [DynamicsCompressionConfig(power_curve=2.5)]),
        ("Smoothing Only", [EmotionSmoothingConfig(alpha=0.3)]),
        ("Combined (Natural Preset)", [
            DynamicsCompressionConfig(power_curve=2.5),
            EmotionSmoothingConfig(alpha=0.4)
        ]),
    ]

    for name, config_list in configs:
        processor = EmotionPostProcessor(config_list)
        result = processor.process(raw_emotions, speaker="EMMA", text="I'm very annoyed.")
        print(f"{name}:")
        print(f"  Output: {result}")
        print()

    # Test with exclamation
    print("With Exclamation Mark:")
    processor = EmotionPostProcessor([DynamicsCompressionConfig(power_curve=2.5, respect_exclamation=True)])
    result = processor.process(raw_emotions, speaker="EMMA", text="STOP IT!!!")
    print(f"  Input:  {raw_emotions}")
    print(f"  Output: {result}")
    print(f"  (Peak preserved for exclamation)\n")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    demo_processing()
