"""
Podcast Pipeline - Clean, Composable Audio Production

This is a REFACTOR of the monolithic PodcastDirector into:
- Pure functions (no side effects)
- Composable pipeline steps
- Clear separation of concerns

USAGE:

    from src.extensions.libraries.audio_producer.workflows.podcast.pipeline import (
        run_podcast_pipeline,
        PipelineConfig
    )

    # Configure the pipeline
    config = PipelineConfig(
        emotion_mode="single_call",
        voices={
            "EMMA": "/path/to/emma.mp3",
            "OLIVIA": "/path/to/olivia.mp3",
            "NARRATOR": "/path/to/narrator.mp3"
        }
    )

    # Run the pipeline
    result = await run_podcast_pipeline(
        script_text=script,
        config=config,
        llm_client=my_llm,
        http_client=my_http
    )

    # Save the result
    result.save(Path("output/podcast.wav"))

ARCHITECTURE:

    Script → [Emotion Detection] → Tags → [TTS Generation] → Chunks → [Post Process] → Audio

Each step is a PURE FUNCTION:
- Deterministic (same input → same output)
- Side-effect free (all I/O is explicit)
- Easy to test (mock the clients)
- Easy to reason about
"""

# Pipeline runner (the main entry point)
from .runner import (
    run_podcast_pipeline,
    run_podcast_pipeline_simple,
    PipelineConfig,
    PipelineResult
)

# Pipeline steps (for advanced usage)
from .steps.emotion_detection import (
    detect_emotions,
    EmotionDetectionConfig,
    ProcessingMode,
    EmotionVector,
    EmotionTaggedLine
)

from .steps.tts_generation import (
    generate_audio,
    generate_audio_parallel,
    TTSGenerationConfig,
    AudioChunk
)

from .steps.post_processing import (
    process_audio_chunks,
    PostProcessingConfig,
    CrossfadeConfig,
    FinalAudio
)

__all__ = [
    # Main API
    "run_podcast_pipeline",
    "run_podcast_pipeline_simple",
    "PipelineConfig",
    "PipelineResult",

    # Emotion Detection
    "detect_emotions",
    "EmotionDetectionConfig",
    "ProcessingMode",
    "EmotionVector",
    "EmotionTaggedLine",

    # TTS Generation
    "generate_audio",
    "generate_audio_parallel",
    "TTSGenerationConfig",
    "AudioChunk",

    # Post Processing
    "process_audio_chunks",
    "PostProcessingConfig",
    "CrossfadeConfig",
    "FinalAudio",
]
