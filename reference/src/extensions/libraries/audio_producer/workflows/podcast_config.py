"""Podcast Generation Configuration System.

This module defines all configuration options for the three-stage podcast generation pipeline:
1. Chunking - How dialogue lines are grouped for LLM processing
2. Context - What conversation history is included for emotion detection
3. Post-Processing - Emotion vector manipulation before IndexTTS

Example usage:
    from podcast_config import PodcastConfig, PRESET_QUALITY, PRESET_BALANCED, PRESET_SPEED

    # Use a preset
    config = PRESET_BALANCED

    # Or customize
    config = PodcastConfig(
        chunking=ChunkingConfig(mode="rolling_window", window_size=7),
        context=ContextConfig(mode="rolling_window", window_size=7),
        post_processing=[
            EmotionSmoothingConfig(alpha=0.3, method="exponential_moving_average"),
            EmotionClampingConfig(min_valence=-0.5, max_valence=0.8),
        ]
    )
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from enum import Enum


# =============================================================================
# Stage 1: Chunking Configuration
# =============================================================================

class ChunkingMode(str, Enum):
    """How dialogue lines are grouped for LLM processing."""
    WHOLE = "whole"  # All lines in one LLM call
    ROLLING_WINDOW = "rolling_window"  # Fixed-size sliding window
    SINGLE = "single"  # One line at a time (current default)
    SEMANTIC_BREAK = "semantic_break"  # LLM identifies natural break points


@dataclass
class ChunkingConfig:
    """Configuration for Stage 1: Chunking LLM Inputs.

    Attributes:
        mode: How to group dialogue lines for LLM processing
        window_size: Number of lines per chunk (for rolling_window mode)
        max_lines_per_chunk: Maximum lines in a semantic break chunk
        break_criteria: Criteria for semantic break detection
        overlap: Number of overlapping lines between chunks (for context continuity)
    """
    mode: Literal["whole", "rolling_window", "single", "semantic_break"] = "single"
    window_size: int = 7  # For rolling_window mode
    max_lines_per_chunk: int = 20  # For semantic_break mode
    overlap: int = 0  # Overlapping lines between chunks (0 = no overlap)
    break_criteria: List[str] = field(default_factory=lambda: ["scene", "topic", "speaker"])

    def validate(self) -> None:
        """Validate configuration."""
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if self.max_lines_per_chunk < 1:
            raise ValueError("max_lines_per_chunk must be at least 1")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.window_size and self.mode == "rolling_window":
            raise ValueError("overlap must be less than window_size")


# =============================================================================
# Stage 2: Context Configuration
# =============================================================================

class ContextMode(str, Enum):
    """What conversation history is included for emotion detection."""
    FULL_CONTEXT = "full_context"  # Entire conversation history
    ROLLING_WINDOW = "rolling_window"  # Last N lines
    NO_CONTEXT = "no_context"  # Process each line independently
    SMART_CONTEXT = "smart_context"  # Dynamic context based on content


@dataclass
class ContextConfig:
    """Configuration for Stage 2: Context Handling.

    Attributes:
        mode: How to handle conversation context
        window_size: Number of previous lines to include (for rolling_window/smart_context)
        include_metadata: Include speaker names, scene info in context
        include_emotions: Include previous emotion tags in context
        base_window: Base window size for smart_context
        expand_on_scene_change: Expand context on scene changes (smart_context)
        include_same_speaker_history: Include more context for same speaker (smart_context)
        max_context_lines: Maximum context lines for smart_context
    """
    mode: Literal["full_context", "rolling_window", "no_context", "smart_context"] = "rolling_window"
    window_size: int = 7  # For rolling_window and smart_context modes
    include_metadata: bool = True
    include_emotions: bool = True  # Include previous emotion tags in LLM prompt
    # Smart context options
    base_window: int = 5
    expand_on_scene_change: bool = True
    include_same_speaker_history: bool = True
    max_context_lines: int = 15

    def validate(self) -> None:
        """Validate configuration."""
        if self.window_size < 0:
            raise ValueError("window_size must be non-negative")
        if self.max_context_lines < self.base_window:
            raise ValueError("max_context_lines must be >= base_window")


# =============================================================================
# Stage 3: Post-Processing Configuration
# =============================================================================

class PostProcessorType(str, Enum):
    """Types of post-processors for emotion vectors."""
    SMOOTHING = "emotion_smoothing"
    CONSISTENCY = "emotion_consistency"
    WEIGHTING = "emotion_weighting"
    INTERPOLATION = "emotion_interpolation"
    CLAMPING = "emotion_clamping"
    SCENE_AWARE = "scene_aware_emotion"
    DYNAMICS_COMPRESSION = "dynamics_compression"  # Gamma correction, etc.


@dataclass
class EmotionSmoothingConfig:
    """Reduce sudden emotion changes between segments.

    Uses exponential moving average or linear weighted smoothing.
    """
    name: Literal["emotion_smoothing"] = "emotion_smoothing"
    method: Literal["exponential_moving_average", "linear_weighted", "kalman_filter"] = "exponential_moving_average"
    alpha: float = 0.3  # Smoothing factor (0-1). Lower = smoother/slower transitions.
    window_size: int = 3  # Number of previous segments to average

    def validate(self) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")


@dataclass
class EmotionConsistencyConfig:
    """Maintain character-specific emotional baselines.

    Each character has a baseline emotion profile and allowed variance.
    """
    name: Literal["emotion_consistency"] = "emotion_consistency"
    character_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Example: {"EMMA": {"baseline": [0.1, 0.5, 0.6, 0.3, 0.0, 0.1, 0.1, 0.1], "variance": 0.3}}
    max_deviation: float = 0.4  # Maximum allowed deviation from baseline

    def validate(self) -> None:
        if self.max_deviation < 0:
            raise ValueError("max_deviation must be non-negative")
        for char, profile in self.character_profiles.items():
            if "baseline" not in profile:
                raise ValueError(f"Missing baseline for character {char}")
            if len(profile["baseline"]) != 8:
                raise ValueError(f"Baseline must have 8 dimensions for {char}")


@dataclass
class EmotionWeightingConfig:
    """Adjust emotion intensity for specific effects.

    Apply multipliers to specific emotion dimensions or overall intensity.
    """
    name: Literal["emotion_weighting"] = "emotion_weighting"
    intensity_multiplier: float = 1.0  # Overall intensity
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    # Example: {"valence": 1.0, "arousal": 1.2, "dominance": 0.9, ...}


@dataclass
class CharacterIntensityReducerConfig:
    """Reduce emotion intensity for specific characters.

    Some speakers (like EMMA) tend to be too emotional/shouting.
    This applies a per-character intensity reduction.
    """
    name: Literal["character_intensity_reducer"] = "character_intensity_reducer"
    character_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "EMMA": 0.6,  # Reduce Emma's emotions to 60% (she yells too much)
    })
    # More aggressive reduction for specific dimensions
    dimension_reduction: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "EMMA": {"arousal": 0.5, "dominance": 0.6},  # Even less arousal/dominance for Emma
    })

    def validate(self) -> None:
        for char, multiplier in self.character_multipliers.items():
            if multiplier < 0:
                raise ValueError(f"Character multiplier for {char} must be non-negative")
        for char, dims in self.dimension_reduction.items():
            for dim, val in dims.items():
                if val < 0:
                    raise ValueError(f"Dimension reduction for {char}.{dim} must be non-negative")


@dataclass
class EmotionInterpolationConfig:
    """Smooth transitions between emotion states.

    Add intermediate emotion steps between segments for gradual transitions.
    """
    name: Literal["emotion_interpolation"] = "emotion_interpolation"
    method: Literal["linear", "bezier", "cosine"] = "linear"
    steps: int = 3  # Number of interpolation steps

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")


@dataclass
class EmotionClampingConfig:
    """Limit emotion values to specific ranges.

    Prevents extreme emotions that may cause audio distortion.
    """
    name: Literal["emotion_clamping"] = "emotion_clamping"
    min_valence: float = -1.0
    max_valence: float = 1.0
    min_arousal: float = 0.0
    max_arousal: float = 1.0
    # You can add min/max for other dimensions too

    def validate(self) -> None:
        if self.min_valence > self.max_valence:
            raise ValueError("min_valence must be <= max_valence")
        if self.min_arousal > self.max_arousal:
            raise ValueError("min_arousal must be <= max_arousal")


@dataclass
class SceneAwareEmotionConfig:
    """Adjust emotions based on scene type.

    Detects scene type and applies appropriate emotion adjustments.
    """
    name: Literal["scene_aware_emotion"] = "scene_aware_emotion"
    scene_rules: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "action": {"arousal": 1.3, "dominance": 1.2},
        "romance": {"valence": 1.2, "arousal": 1.1},
        "mystery": {"valence": 0.7, "surprise": 1.2},
        "exposition": {"valence": 1.0, "arousal": 1.0},
    })
    auto_detect: bool = True

    def validate(self) -> None:
        pass


@dataclass
class DynamicsCompressionConfig:
    """Apply power curve (gamma correction) to prevent "constant shouting".

    LLMs tend to output extreme values (0.8-1.0) for moderate emotions.
    This uses a power curve to suppress mid-range values while preserving peaks.
    """
    name: Literal["dynamics_compression"] = "dynamics_compression"
    power_curve: float = 2.0  # Gamma value. >1 suppresses mid-range, <1 boosts quiet
    max_intensity: float = 0.95  # Hard cap to prevent audio distortion
    respect_exclamation: bool = True  # Allow peaks for text ending in "!"
    exclamation_threshold: int = 2  # Number of "!" to trigger peak

    def validate(self) -> None:
        if self.power_curve <= 0:
            raise ValueError("power_curve must be positive")
        if not 0 <= self.max_intensity <= 1:
            raise ValueError("max_intensity must be between 0 and 1")


# Union type for all post-processor configs
PostProcessorConfig = (
    EmotionSmoothingConfig |
    EmotionConsistencyConfig |
    EmotionWeightingConfig |
    EmotionInterpolationConfig |
    EmotionClampingConfig |
    SceneAwareEmotionConfig |
    DynamicsCompressionConfig |
    CharacterIntensityReducerConfig
)


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class PodcastConfig:
    """Complete configuration for podcast generation pipeline.

    This class combines all three stages of the pipeline:
    1. Chunking - How dialogue lines are grouped
    2. Context - What history is included
    3. Post-Processing - Emotion vector manipulation

    Attributes:
        chunking: Configuration for Stage 1
        context: Configuration for Stage 2
        post_processing: Ordered list of post-processing steps
        description: Human-readable description of this configuration
    """
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    post_processing: List[PostProcessorConfig] = field(default_factory=list)
    description: str = "Custom podcast configuration"

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.chunking.validate()
        self.context.validate()
        for processor in self.post_processing:
            processor.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "chunking": {
                "mode": self.chunking.mode,
                "window_size": self.chunking.window_size,
                "max_lines_per_chunk": self.chunking.max_lines_per_chunk,
                "overlap": self.chunking.overlap,
                "break_criteria": self.chunking.break_criteria,
            },
            "context": {
                "mode": self.context.mode,
                "window_size": self.context.window_size,
                "include_metadata": self.context.include_metadata,
                "include_emotions": self.context.include_emotions,
                "base_window": self.context.base_window,
                "expand_on_scene_change": self.context.expand_on_scene_change,
                "include_same_speaker_history": self.context.include_same_speaker_history,
                "max_context_lines": self.context.max_context_lines,
            },
            "post_processing": [
                {
                    "name": p.name,
                    **{k: v for k, v in p.__dict__.items() if k != "name"}
                }
                for p in self.post_processing
            ]
        }


# =============================================================================
# Preset Configurations
# =============================================================================

# Quality Mode - Best emotions, slowest
PRESET_QUALITY = PodcastConfig(
    description="Quality Mode - Best emotion detection, slowest processing",
    chunking=ChunkingConfig(mode="whole"),
    context=ContextConfig(mode="full_context", include_emotions=True),
    post_processing=[
        EmotionConsistencyConfig(),
        EmotionSmoothingConfig(alpha=0.3),
    ]
)

# Speed Mode - Fastest, flat emotions
PRESET_SPEED = PodcastConfig(
    description="Speed Mode - Fastest processing, flat emotions",
    chunking=ChunkingConfig(mode="single"),
    context=ContextConfig(mode="no_context"),
    post_processing=[]
)

# Balanced Mode - Default (current implementation)
PRESET_BALANCED = PodcastConfig(
    description="Balanced Mode - Good balance of speed and quality",
    chunking=ChunkingConfig(mode="rolling_window", window_size=7),
    context=ContextConfig(mode="rolling_window", window_size=7, include_emotions=True),
    post_processing=[
        EmotionSmoothingConfig(alpha=0.3),
    ]
)

# Story Mode - Narrative focus with scene awareness
PRESET_STORY = PodcastConfig(
    description="Story Mode - Optimized for narrative content with scene awareness",
    chunking=ChunkingConfig(mode="semantic_break", max_lines_per_chunk=20),
    context=ContextConfig(mode="smart_context", base_window=5, max_context_lines=15),
    post_processing=[
        SceneAwareEmotionConfig(),
        EmotionInterpolationConfig(steps=2),
        EmotionSmoothingConfig(alpha=0.4),
        EmotionConsistencyConfig(),
    ]
)

# Natural Mode - Prevents "constant shouting" with dynamics compression
PRESET_NATURAL = PodcastConfig(
    description="Natural Mode - Rolling window batch + dynamics compression to prevent robotic shouting",
    chunking=ChunkingConfig(mode="rolling_window", window_size=7),
    context=ContextConfig(mode="rolling_window", window_size=7),
    post_processing=[
        CharacterIntensityReducerConfig(),  # Reduce Emma's yelling
        DynamicsCompressionConfig(power_curve=2.5, max_intensity=0.95),
        EmotionSmoothingConfig(alpha=0.4, method="exponential_moving_average"),
    ]
)

# Dramatic Mode - High emotion, good for thrillers
PRESET_DRAMATIC = PodcastConfig(
    description="Dramatic Mode - High intensity for action/thriller content",
    chunking=ChunkingConfig(mode="rolling_window", window_size=5),
    context=ContextConfig(mode="rolling_window", window_size=5),
    post_processing=[
        EmotionWeightingConfig(intensity_multiplier=1.3, dimension_weights={"arousal": 1.3}),
        SceneAwareEmotionConfig(),
    ]
)


# =============================================================================
# Configuration Presets Registry
# =============================================================================

PRESET_REGISTRY: Dict[str, PodcastConfig] = {
    "quality": PRESET_QUALITY,
    "speed": PRESET_SPEED,
    "balanced": PRESET_BALANCED,
    "story": PRESET_STORY,
    "natural": PRESET_NATURAL,
    "dramatic": PRESET_DRAMATIC,
}


def get_preset(name: str) -> PodcastConfig:
    """Get a preset configuration by name.

    Args:
        name: Preset name (quality, speed, balanced, story, natural, dramatic)

    Returns:
        PodcastConfig instance

    Raises:
        ValueError: If preset name is not found
    """
    name_lower = name.lower()
    if name_lower not in PRESET_REGISTRY:
        available = ", ".join(PRESET_REGISTRY.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESET_REGISTRY[name_lower]


def list_presets() -> List[str]:
    """List all available preset names."""
    return list(PRESET_REGISTRY.keys())
