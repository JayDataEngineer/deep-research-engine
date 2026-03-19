"""
Pipeline Runner - Composition of Pure Functions

This is the GLUE CODE that composes the pipeline steps.

The pipeline is:
  Script Text → [Emotion Detection] → Emotion Tags
                          ↓
  Emotion Tags → [TTS Generation] → Audio Chunks
                          ↓
  Audio Chunks → [Post Processing] → Final Audio

Each step is a PURE FUNCTION - easy to test, easy to reason about.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Complete configuration for the entire pipeline."""
    # Step 1: Emotion Detection
    emotion_mode: str = "single_call"          # single_call, rolling_window, parallel
    emotion_prompt: str = "single_line"
    emotion_parallel_workers: int = 3

    # Step 2: TTS Generation
    tts_url: str = "http://127.0.0.1:9000"
    tts_parallel_workers: int = 4
    tts_diffusion_steps: int = 25

    # Step 3: Post Processing
    add_silence: bool = True
    silence_duration: float = 0.5
    crossfade_enabled: bool = False
    crossfade_duration_ms: int = 100

    # Voice mapping (speaker → reference_audio)
    voices: Dict[str, str] = field(default_factory=dict)

    # Output
    output_format: str = "wav"
    sample_rate: int = 22050


@dataclass
class PipelineResult:
    """Result of running the podcast pipeline."""
    final_audio: bytes                  # Final audio bytes
    output_path: Optional[Path]         # Where it was saved (if saved)
    duration: float                     # Duration in seconds
    num_chunks: int                     # Number of audio chunks

    # Timing stats
    emotion_detection_time: float = 0.0
    tts_generation_time: float = 0.0
    post_processing_time: float = 0.0
    total_time: float = 0.0

    # Intermediate results (for debugging)
    emotion_tags: Optional[List] = None
    audio_chunks: Optional[List] = None

    def save(self, path: Path) -> None:
        """Save final audio to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.final_audio)
        self.output_path = path
        logger.info(f"Saved podcast to {path} ({self.duration:.2f}s)")


# ============================================================================
# PIPELINE RUNNER - Composes the pure functions
# ============================================================================

async def run_podcast_pipeline(
    script_text: str,
    config: PipelineConfig,
    llm_client: Any,
    http_client: Any
) -> PipelineResult:
    """Run the complete podcast pipeline.

    This is COMPOSITION of pure functions:
    - No side effects in the pipeline itself
    - All I/O is explicit (passed in as parameters)
    - Easy to test by mocking llm_client and http_client

    Args:
        script_text: The podcast script
        config: Pipeline configuration
        llm_client: LLM client (for emotion detection)
        http_client: HTTP client (for TTS calls)

    Returns:
        PipelineResult with final audio and stats

    Example:
        result = await run_podcast_pipeline(
            script_text=script,
            config=config,
            llm_client=my_llm,
            http_client=my_http
        )
        result.save(Path("output/podcast.wav"))
        print(f"Generated {result.duration:.2f}s audio in {result.total_time:.2f}s")
    """
    pipeline_start = time.time()

    # ========================================================================
    # STEP 1: Emotion Detection
    # ========================================================================
    from .steps.emotion_detection import (
        detect_emotions,
        EmotionDetectionConfig,
        ProcessingMode
    )

    emotion_config = EmotionDetectionConfig(
        mode=ProcessingMode(config.emotion_mode),
        prompt_name=config.emotion_prompt,
        parallel_workers=config.emotion_parallel_workers
    )

    emotion_start = time.time()
    emotion_tags = await detect_emotions(script_text, emotion_config, llm_client)
    emotion_time = time.time() - emotion_start

    logger.info(f"[PIPELINE] Emotion detection: {len(emotion_tags)} lines in {emotion_time:.2f}s")

    # ========================================================================
    # STEP 2: TTS Generation
    # ========================================================================
    from .steps.tts_generation import (
        generate_audio_parallel,
        TTSGenerationConfig
    )

    tts_config = TTSGenerationConfig(
        tts_url=config.tts_url,
        parallel_workers=config.tts_parallel_workers,
        diffusion_steps=config.tts_diffusion_steps,
        output_format=config.output_format,
        sample_rate=config.sample_rate
    )

    tts_start = time.time()
    audio_chunks = await generate_audio_parallel(
        emotion_tagged_lines=emotion_tags,
        config=tts_config,
        voice_map=config.voices,
        http_client=http_client
    )
    tts_time = time.time() - tts_start

    logger.info(f"[PIPELINE] TTS generation: {len(audio_chunks)} chunks in {tts_time:.2f}s")

    # ========================================================================
    # STEP 3: Post Processing
    # ========================================================================
    from .steps.post_processing import (
        process_audio_chunks,
        PostProcessingConfig,
        CrossfadeConfig
    )

    post_config = PostProcessingConfig(
        add_silence=config.add_silence,
        silence_duration=config.silence_duration,
        crossfade=CrossfadeConfig(
            enabled=config.crossfade_enabled,
            duration_ms=config.crossfade_duration_ms
        ) if config.crossfade_enabled else None,
        output_format=config.output_format,
        sample_rate=config.sample_rate
    )

    post_start = time.time()
    final_audio = process_audio_chunks(audio_chunks, post_config)
    post_time = time.time() - post_start

    logger.info(f"[PIPELINE] Post processing: {post_time:.2f}s")

    # ========================================================================
    # RETURN RESULT
    # ========================================================================
    total_time = time.time() - pipeline_start

    return PipelineResult(
        final_audio=final_audio.audio_data,
        duration=final_audio.duration,
        num_chunks=final_audio.num_chunks,
        emotion_detection_time=emotion_time,
        tts_generation_time=tts_time,
        post_processing_time=post_time,
        total_time=total_time,
        emotion_tags=emotion_tags,
        audio_chunks=audio_chunks
    )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def run_podcast_pipeline_simple(
    script_text: str,
    voices: Dict[str, str],
    llm_client: Any,
    output_path: Path
) -> PipelineResult:
    """Run pipeline with sensible defaults.

    Convenience function for common use case.
    """
    import httpx

    config = PipelineConfig(
        voices=voices,
        emotion_mode="single_call",
        add_silence=True,
        silence_duration=0.5
    )

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        result = await run_podcast_pipeline(
            script_text=script_text,
            config=config,
            llm_client=llm_client,
            http_client=http_client
        )

    result.save(output_path)
    return result
