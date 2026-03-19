"""Configuration schemas for Audio Producer Library.

All audio production workflows use these Pydantic models for configuration.
This provides type safety, validation, and IDE autocomplete.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError


class IndexTTS2EmotionVector(BaseModel):
    """Validated IndexTTS2 emotion vector.

    Format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    All values must be between 0.0 and 1.0.
    """

    happy: float = Field(ge=0.0, le=1.0, description="Happiness/emotion")
    angry: float = Field(ge=0.0, le=1.0, description="Anger/intensity")
    sad: float = Field(ge=0.0, le=1.0, description="Sadness")
    afraid: float = Field(ge=0.0, le=1.0, description="Fear")
    disgusted: float = Field(ge=0.0, le=1.0, description="Disgust")
    melancholic: float = Field(ge=0.0, le=1.0, description="Melancholy")
    surprised: float = Field(ge=0.0, le=1.0, description="Surprise")
    calm: float = Field(ge=0.0, le=1.0, description="Calmness/neutral")

    @field_validator("*")
    @classmethod
    def check_ranges(cls, v):
        """Ensure all values are in valid range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Emotion values must be between 0.0 and 1.0")
        return v

    @field_validator("calm")
    @classmethod
    def validate_neutral_speech(cls, v, info):
        """For neutral speech, calm should be high (0.7-1.0) and other emotions low."""
        other_emotions = [
            info.data.get("happy", 0),
            info.data.get("angry", 0),
            info.data.get("sad", 0),
            info.data.get("afraid", 0),
            info.data.get("disgusted", 0),
            info.data.get("melancholic", 0),
            info.data.get("surprised", 0),
        ]
        max_other = max(other_emotions) if other_emotions else 0
        if v < 0.5 and max_other < 0.3:
            raise ValueError(
                f"For neutral/low-emotion speech, 'calm' should be >= 0.5 (got {v}). "
                f"This prevents monotone shouting audio."
            )
        return v

    def to_list(self) -> List[float]:
        """Convert to list format for IndexTTS."""
        return [self.happy, self.angry, self.sad, self.afraid, self.disgusted,
                self.melancholic, self.surprised, self.calm]

    @classmethod
    def from_list(cls, values: List[float]) -> "IndexTTS2EmotionVector":
        """Create from list (with backwards compatibility for old format)."""
        # Handle 8-dim input - detect old format
        if len(values) == 8:
            # Old format: [anger, disgust, fear, happiness, sadness, surprise, neutral, intensity]
            # Detect: if 'neutral' (index 6) > 0.5, it's old format
            if values[6] > 0.5:
                # Convert old format to new
                return cls(
                    happy=values[3],  # happiness
                    angry=values[0],  # anger
                    sad=values[4],    # sadness
                    afraid=values[2], # fear
                    disgusted=values[1], # disgust
                    melancholic=0.0,
                    surprised=values[5], # surprise
                    calm=values[6],   # neutral
                )
            else:
                # Already IndexTTS2 format
                return cls(
                    happy=values[0], angry=values[1], sad=values[2], afraid=values[3],
                    disgusted=values[4], melancholic=values[5], surprised=values[6],
                    calm=values[7]
                )
        else:
            raise ValueError(f"Invalid emotion vector length: {len(values)} (expected 8)")

    def to_indextts_format(self) -> List[float]:
        """Return format for IndexTTS."""
        return self.to_list()


class EmotionConfig(BaseModel):
    """Configuration for emotion detection and tagging."""

    enabled: bool = True
    quality_mode: Literal["fast", "quality", "overdrive", "sectioned"] = "fast"  # fast, quality (thinking), overdrive (two-phase with semantic chunks), sectioned (two-phase per-section)
    mode: Literal["llm", "rule_based", "neutral"] = "llm"
    processing_mode: Literal["single_call", "rolling_window", "sequential", "parallel", "deep_agent"] = "single_call"
    batch_size: int = Field(default=50, ge=1, le=200)
    parallel_concurrency: int = Field(default=8, ge=1, le=32)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    prompts_file: str = "emotion.jinja2"

    # Overdrive mode specific settings
    overdrive_semantic_batch_size: int = Field(default=10, ge=6, le=15, description="Lines per semantic batch in overdrive mode")
    overdrive_max_concurrent: int = Field(default=8, ge=1, le=16, description="Max concurrent LLM calls in overdrive phase 2")

    # Sectioned mode specific settings
    sectioned_max_concurrent: int = Field(default=8, ge=1, le=16, description="Max concurrent sections in sectioned phase 2")


class VoiceConfig(BaseModel):
    """Configuration for a single voice."""

    voice_id: str = Field(description="Preset IndexTTS voice ID")
    reference_audio: Optional[str] = Field(default=None, description="Path to custom voice sample for cloning")


class VoiceMap(BaseModel):
    """Mapping of speaker names to voice configurations."""

    voices: Dict[str, VoiceConfig] = Field(default_factory=dict)

    def get_voice_id(self, speaker: str, default: str = "index2_ref_neutral") -> str:
        """Get voice ID for a speaker."""
        return self.voices.get(speaker, VoiceConfig(voice_id=default)).voice_id

    def get_reference_audio(self, speaker: str) -> Optional[str]:
        """Get reference audio path for a speaker."""
        return self.voices.get(speaker, VoiceConfig(voice_id="")).reference_audio


class IndexSamplingConfig(BaseModel):
    """Advanced IndexTTS generation parameters."""

    # Precision control (fp32 = full quality, fp16 = faster)
    precision: Literal["fp32", "fp16"] = Field(default="fp32", description="Inference precision (fp32=full quality, fp16=faster)")

    # Emotion control
    emo_alpha: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Emotion intensity")
    emo_audio_prompt: Optional[str] = Field(default=None, description="Emotional reference audio path")
    use_emo_text: bool = Field(default=False, description="Use text-based emotion instead of vectors")
    emo_text: Optional[str] = Field(default=None, description="Emotion text description")

    # Sampling control
    do_sample: bool = Field(default=True, description="Enable sampling")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="Top-P sampling")
    top_k: int = Field(default=30, ge=1, le=100, description="Top-K sampling")
    num_beams: int = Field(default=3, ge=1, le=10, description="Number of beams")
    use_random: bool = Field(default=False, description="Add extra randomness")

    # Length and repetition control
    length_penalty: float = Field(default=0.0, ge=-5.0, le=5.0, description="Length penalty")
    repetition_penalty: float = Field(default=10.0, ge=0.0, le=50.0, description="Repetition penalty")
    max_mel_tokens: int = Field(default=1500, ge=100, le=5000, description="Max mel tokens")
    max_text_tokens_per_segment: int = Field(default=120, ge=10, le=500, description="Max text tokens per segment")

    # Timing control
    interval_silence: int = Field(default=200, ge=0, le=2000, description="Silence between segments (ms)")

    # Speed and pitch
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Playback speed")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch multiplier")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    enabled: bool = Field(default=False, description="Enable chunking")
    max_tokens: int = Field(default=120, ge=10, le=500, description="Max tokens per chunk")


class OutputConfig(BaseModel):
    """Configuration for audio output."""

    format: Literal["wav", "mp3", "ogg"] = "wav"
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    add_silence: bool = Field(default=True, description="Add silence between dialogue lines")
    silence_duration: float = Field(default=0.5, ge=0.0, le=5.0, description="Silence duration in seconds")


class CrossfadeConfig(BaseModel):
    """Configuration for crossfade mode."""

    same_speaker_ms: int = Field(default=80, ge=0, le=500, description="Cross-fade duration for same speaker")
    speaker_change_ms: int = Field(default=0, ge=0, le=500, description="Cross-fade for speaker change (usually 0)")


class AdaptiveMergeConfig(BaseModel):
    """Configuration for adaptive merge mode (Solution 1)."""

    emotion_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Emotion distance threshold for merging")
    max_sentences_per_merge: int = Field(default=5, ge=2, le=20, description="Max sentences to merge together")


class GhostTextConfig(BaseModel):
    """Configuration for ghost text mode (Solution 2)."""

    overlap_words: int = Field(default=5, ge=1, le=20, description="Number of trailing words to overlap")
    overlap_chars: int = Field(default=30, ge=10, le=100, description="Alternative: character overlap")


class BreathPauseConfig(BaseModel):
    """Configuration for breath/pause mode (Solution 3)."""

    comma_pause_ms: int = Field(default=50, ge=0, le=500, description="Pause after comma (ms)")
    period_pause_ms: int = Field(default=150, ge=0, le=1000, description="Pause after period (ms)")
    paragraph_pause_ms: int = Field(default=500, ge=0, le=2000, description="Pause after paragraph break (ms)")
    breath_sample_path: str = Field(default="", description="Path to breath sound sample (.wav)")


class SentenceTransitionConfig(BaseModel):
    """Configuration for sentence transition smoothing.

    Addresses the "disjointed TTS" problem by smoothing transitions between
    independently generated TTS segments.
    """

    mode: Literal["none", "crossfade", "adaptive_merge", "ghost_text", "breath_pause"] = Field(
        default="crossfade",
        description="Transition smoothing mode"
    )
    crossfade: CrossfadeConfig = Field(default_factory=CrossfadeConfig)
    adaptive_merge: AdaptiveMergeConfig = Field(default_factory=AdaptiveMergeConfig)
    ghost_text: GhostTextConfig = Field(default_factory=GhostTextConfig)
    breath_pause: BreathPauseConfig = Field(default_factory=BreathPauseConfig)


class AudioJobConfig(BaseModel):
    """Main configuration for any audio production job.

    This is the "Recipe" that tells the Audio Producer how to process audio.
    """

    # Workflow selection
    workflow_id: Literal["simple", "podcast", "audiobook"] = Field(
        default="podcast",
        description="Which workflow to use for this job"
    )

    # Sub-configurations
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    sampling: IndexSamplingConfig = Field(default_factory=IndexSamplingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    sentence_transition: SentenceTransitionConfig = Field(default_factory=SentenceTransitionConfig)

    # Voice mapping (podcast workflow)
    voices: VoiceMap = Field(default_factory=VoiceMap)

    # LLM selection
    llm_model: Optional[str] = Field(default=None, description="Override default LLM model")

    @field_validator("voices", mode="before")
    @classmethod
    def validate_voices(cls, v):
        """Accept both dict and VoiceMap objects."""
        if isinstance(v, dict):
            # Convert dict to VoiceMap
            voice_configs = {}
            for speaker, voice_config in v.items():
                if isinstance(voice_config, str):
                    voice_configs[speaker] = VoiceConfig(voice_id=voice_config)
                elif isinstance(voice_config, dict):
                    voice_configs[speaker] = VoiceConfig(**voice_config)
                else:
                    voice_configs[speaker] = voice_config
            return VoiceMap(voices=voice_configs)
        return v

    def get_generation_params(self) -> dict:
        """Get sampling parameters as a dict for passing to TTS service."""
        params = {
            "speed": self.sampling.speed,
            "pitch": self.sampling.pitch,
            "precision": self.sampling.precision,
        }

        # Add optional parameters only if set
        if self.sampling.emo_alpha is not None:
            params["emo_alpha"] = self.sampling.emo_alpha
        if self.sampling.emo_audio_prompt:
            params["emo_audio_prompt"] = self.sampling.emo_audio_prompt
        if self.sampling.use_emo_text:
            params["use_emo_text"] = self.sampling.use_emo_text
        if self.sampling.emo_text:
            params["emo_text"] = self.sampling.emo_text
        if self.sampling.do_sample is not None:
            params["do_sample"] = self.sampling.do_sample
        if self.sampling.temperature is not None:
            params["temperature"] = self.sampling.temperature
        if self.sampling.top_p is not None:
            params["top_p"] = self.sampling.top_p
        if self.sampling.top_k is not None:
            params["top_k"] = self.sampling.top_k
        if self.sampling.num_beams is not None:
            params["num_beams"] = self.sampling.num_beams
        if self.sampling.use_random is not None:
            params["use_random"] = self.sampling.use_random
        if self.sampling.length_penalty is not None:
            params["length_penalty"] = self.sampling.length_penalty
        if self.sampling.repetition_penalty is not None:
            params["repetition_penalty"] = self.sampling.repetition_penalty
        if self.sampling.max_mel_tokens is not None:
            params["max_mel_tokens"] = self.sampling.max_mel_tokens
        if self.sampling.max_text_tokens_per_segment is not None:
            params["max_text_tokens_per_segment"] = self.sampling.max_text_tokens_per_segment
        if self.sampling.interval_silence is not None:
            params["interval_silence"] = self.sampling.interval_silence

        return params


# Factory function to load config from YAML
def load_config_from_yaml(config_path: str) -> AudioJobConfig:
    """Load AudioJobConfig from a YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        AudioJobConfig instance
    """
    import yaml
    from pathlib import Path

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return AudioJobConfig(**data.get("audio_producer", data))
