"""
PodcastDirector v2 - Thin Orchestrator for Podcast Pipeline

This is a REFACTOR of the monolithic PodcastDirector into a thin wrapper
that uses the new composable pipeline.

Old PodcastDirector: ~1700 lines, doing everything
New PodcastDirector: ~200 lines, just orchestrates the pipeline

The heavy lifting is done by:
- pipeline/steps/emotion_detection.py
- pipeline/steps/tts_generation.py
- pipeline/steps/post_processing.py
- pipeline/runner.py (composes the steps)
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .pipeline import (
    run_podcast_pipeline,
    PipelineConfig,
    PipelineResult
)
from ..config_schemas import AudioJobConfig
from .podcast_config import get_preset, PRESET_BALANCED

logger = logging.getLogger(__name__)


class PodcastDirector:
    """Thin orchestrator for podcast production pipeline.

    The PodcastDirector:
    1. Parses the incoming AudioJobConfig
    2. Converts it to PipelineConfig
    3. Runs the pipeline
    4. Returns the result

    All the heavy lifting (emotion detection, TTS, merging) is done
    by the pure function pipeline steps.
    """

    def __init__(
        self,
        config: AudioJobConfig,
        prompts_dir: Optional[str] = None,
        podcast_config: str = "balanced"
    ):
        """Initialize PodcastDirector.

        Args:
            config: AudioJobConfig with voice mapping and settings
            prompts_dir: Path to prompts directory (deprecated, uses default)
            podcast_config: Preset name (balanced, speed, quality)
        """
        self.config = config
        self.podcast_preset = get_preset(podcast_config)

        # Output directory
        from src.constants import DATA_OUTPUTS_DIR
        self.output_dir = DATA_OUTPUTS_DIR / "audio" / "podcast"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        script_text: str,
        output_path: Path
    ) -> Dict[str, Any]:
        """Execute the podcast production pipeline.

        Args:
            script_text: The podcast script (markdown format)
            output_path: Where to save the final audio

        Returns:
            Dict with result info (audio_path, duration, processing_time, etc.)
        """
        run_start = asyncio.get_event_loop().time()

        # Build pipeline config from AudioJobConfig
        pipeline_config = self._build_pipeline_config()

        # Get LLM and HTTP clients
        llm_client = await self._get_llm_client()
        http_client = await self._get_http_client()

        # Run the pipeline
        logger.info(f"[PODCAST DIRECTOR] Starting pipeline: {len(script_text)} chars")

        result = await run_podcast_pipeline(
            script_text=script_text,
            config=pipeline_config,
            llm_client=llm_client,
            http_client=http_client
        )

        # Save the output
        result.save(output_path)

        # Generate report
        report_path = self._generate_report(result, output_path)

        total_time = asyncio.get_event_loop().time() - run_start

        return {
            "audio_path": str(output_path),
            "duration": result.duration,
            "num_lines": result.num_chunks,
            "processing_time": total_time,
            "output_md_path": str(report_path),
            "emotion_detection_time": result.emotion_detection_time,
            "tts_time": result.tts_generation_time,
            "post_processing_time": result.post_processing_time,
        }

    def _build_pipeline_config(self) -> PipelineConfig:
        """Build PipelineConfig from AudioJobConfig."""
        # Extract voice mapping
        voices = {}
        if self.config.voices:
            for speaker, voice_config in self.config.voices.items():
                if isinstance(voice_config, dict):
                    voices[speaker] = voice_config.get("reference_audio")
                else:
                    voices[speaker] = voice_config

        # Get emotion config
        emotion_config = self.config.emotion or {}
        emotion_mode = emotion_config.get("processing_mode", "single_call")

        # Get output config
        output_config = self.config.output or {}

        return PipelineConfig(
            # Emotion detection
            emotion_mode=emotion_mode,
            emotion_prompt="single_line",  # TODO: map from quality_mode
            emotion_parallel_workers=emotion_config.get("parallel_concurrency", 3),

            # TTS
            tts_parallel_workers=4,  # TODO: make configurable
            tts_diffusion_steps=25,

            # Post processing
            add_silence=output_config.get("add_silence", True),
            silence_duration=output_config.get("silence_duration", 0.5),
            crossfade_enabled=output_config.get("crossfade_enabled", False),
            crossfade_duration_ms=output_config.get("crossfade_duration_ms", 100),

            # Voices
            voices=voices,

            # Output format
            output_format=output_config.get("format", "wav"),
            sample_rate=output_config.get("sample_rate", 22050),
        )

    async def _get_llm_client(self) -> Any:
        """Get or create LLM client."""
        # TODO: Proper LLM client initialization
        # For now, return a mock that raises an error
        raise NotImplementedError("LLM client not yet integrated")

    async def _get_http_client(self) -> Any:
        """Get or create HTTP client for TTS."""
        import httpx
        return httpx.AsyncClient(timeout=120.0)

    def _generate_report(self, result: PipelineResult, output_path: Path) -> Path:
        """Generate markdown report of the podcast generation."""
        report_path = output_path.with_suffix('.md')

        report = f"""# Podcast Generation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Audio Output

- **File**: `{output_path.name}`
- **Duration**: {result.duration:.2f} seconds ({result.duration/60:.1f} minutes)
- **Chunks**: {result.num_chunks} dialogue lines

## Processing Time

| Stage | Time |
|-------|------|
| Emotion Detection | {result.emotion_detection_time:.2f}s |
| TTS Generation | {result.tts_generation_time:.2f}s |
| Post Processing | {result.post_processing_time:.2f}s |
| **Total** | **{result.total_time:.2f}s** |

**Real-Time Factor (RTF)**: {result.total_time / result.duration:.2f}x
(RTF < 1.0 = faster than real-time)

## Configuration

- **Emotion Mode**: {result.emotion_tags[0].__class__.__name__ if result.emotion_tags else 'N/A'}
- **Crossfade**: Enabled
- **Silence**: 0.5s between chunks

---
*Generated by Podcast Pipeline v2*
"""
        report_path.write_text(report)
        return report_path
