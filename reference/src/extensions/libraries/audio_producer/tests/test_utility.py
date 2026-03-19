"""Shared utility functions for audio producer pipeline tests.

This module provides common utilities used by all mode tests to ensure
consistent testing across different quality modes.
"""

import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import time

# Add backend root to path for imports
# test_utility.py is in: backend/src/extensions/libraries/audio_producer/tests/
# Backend root is 6 levels up
backend_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_root))

from src.extensions.libraries.audio_producer.config_schemas import (
    AudioJobConfig,
    EmotionConfig,
    VoiceConfig,
    VoiceMap,
    OutputConfig
)
from src.extensions.libraries.audio_producer.workflows.podcast_director import PodcastDirector


# Test configuration paths
PROJECT_ROOT = Path("/home/ubuntu/Documents/programs/text/unified-llm-interface")
BACKEND_ROOT = PROJECT_ROOT / "backend"
VOICE_SAMPLES_DIR = PROJECT_ROOT / "data/inputs/audio_samples"
PROMPTS_DIR = BACKEND_ROOT / "src/extensions/libraries/audio_producer/prompts"

# Test input files
TEST_INPUTS = {
    "short": PROJECT_ROOT / "data/inputs/tests/TTS Podcast Episode The Great Divide.txt",
}

# Output directories - use project root data/outputs, not backend/data/outputs
OUTPUT_BASE = PROJECT_ROOT / "data/outputs/audio/pipeline_tests"


def create_voice_map() -> VoiceMap:
    """Create standard voice map with reference audio for cloning."""
    return VoiceMap(voices={
        "EMMA": VoiceConfig(
            voice_id="index2_ref_neutral",
            reference_audio=str(VOICE_SAMPLES_DIR / "Emma.mp3")
        ),
        "OLIVIA": VoiceConfig(
            voice_id="index2_ref_voice2",
            reference_audio=str(VOICE_SAMPLES_DIR / "Olivia.mp3")
        ),
        "NARRATOR": VoiceConfig(
            voice_id="index2_ref_neutral",
            reference_audio=str(VOICE_SAMPLES_DIR / "Narrator.mp3")
        )
    })


def create_base_config(quality_mode: str = "fast") -> AudioJobConfig:
    """Create base configuration for testing.

    Args:
        quality_mode: The quality mode to test (fast, quality, overdrive, sectioned)

    Returns:
        AudioJobConfig with standard settings
    """
    return AudioJobConfig(
        workflow_id="podcast",
        emotion=EmotionConfig(
            enabled=True,
            quality_mode=quality_mode,
            mode="llm",
            processing_mode="rolling_window",
            sectioned_max_concurrent=8
        ),
        voices=create_voice_map(),
        output=OutputConfig(
            format="mp3",
            sample_rate=24000,
            add_silence=True,
            silence_duration=0.5
        )
    )


def load_test_input(name: str = "short") -> str:
    """Load test input text.

    Args:
        name: Name of test input (default: "short")

    Returns:
        Test input text
    """
    input_path = TEST_INPUTS.get(name)
    if not input_path or not input_path.exists():
        raise FileNotFoundError(f"Test input not found: {input_path}")
    return input_path.read_text()


def get_output_dir(mode_name: str) -> Path:
    """Get output directory for a specific mode test.

    Args:
        mode_name: Name of the mode being tested

    Returns:
        Path to output directory
    """
    output_dir = OUTPUT_BASE / mode_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_pipeline_test(
    mode_name: str,
    quality_mode: str,
    input_text: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Run a pipeline test for a specific quality mode.

    Args:
        mode_name: Display name for the mode
        quality_mode: Quality mode config value
        input_text: Input text to process
        output_dir: Output directory path

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {mode_name}")
    print(f"{'='*70}")
    print(f"   Quality mode: {quality_mode}")
    print(f"   Input characters: {len(input_text)}")
    print(f"   Output directory: {output_dir}")

    # Create configuration
    config = create_base_config(quality_mode)

    # Create podcast director
    director = PodcastDirector(config, prompts_dir=str(PROMPTS_DIR))

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"{quality_mode}_{timestamp}.{config.output.format}"

    print(f"\n🎬 Starting pipeline generation...")

    # Run pipeline
    start_time = time.time()
    try:
        result = await director.execute(
            input_text=input_text,
            output_path=output_path
        )
        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"✅ {mode_name} COMPLETE")
        print(f"{'='*70}")
        print(f"\n📄 Output files:")
        print(f"   Audio: {result.get('audio_path', 'N/A')}")
        print(f"   Report: {result.get('output_md_path', 'N/A')}")

        print(f"\n⏱️  Performance:")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")

        if 'duration' in result:
            print(f"   Audio duration: {result['duration']:.2f}s ({result['duration']/60:.1f} minutes)")
            print(f"   Real-time factor: {total_time/result['duration']:.2f}x")

        if 'num_segments' in result:
            print(f"   Segments processed: {result['num_segments']}")

        return {
            "mode": mode_name,
            "quality_mode": quality_mode,
            "success": True,
            "total_time": total_time,
            "audio_duration": result.get('duration', 0),
            "realtime_factor": total_time / result.get('duration', 1),
            "num_segments": result.get('num_segments', 0),
            "audio_path": result.get('audio_path'),
            "report_path": result.get('output_md_path')
        }

    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

        return {
            "mode": mode_name,
            "quality_mode": quality_mode,
            "success": False,
            "error": str(e),
            "total_time": total_time
        }


def print_summary(results: list) -> None:
    """Print test summary across all modes.

    Args:
        results: List of test result dictionaries
    """
    print(f"\n{'='*70}")
    print(f"📊 PIPELINE TEST SUMMARY")
    print(f"{'='*70}")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")

    if successful:
        print(f"\n{'Mode':<15} {'Time (s)':<12} {'Audio (s)':<12} {'RTF':<8} {'Segments'}")
        print(f"{'-'*70}")

        for r in successful:
            mode = r['mode']
            time_s = f"{r['total_time']:.1f}"
            audio_s = f"{r['audio_duration']:.1f}" if r.get('audio_duration') else 'N/A'
            rtf = f"{r['realtime_factor']:.2f}x" if r.get('realtime_factor') else 'N/A'
            segs = str(r['num_segments']) if r.get('num_segments') else 'N/A'
            print(f"{mode:<15} {time_s:<12} {audio_s:<12} {rtf:<8} {segs}")

    if failed:
        print(f"\n❌ Failed tests:")
        for r in failed:
            print(f"   {r['mode']}: {r.get('error', 'Unknown error')}")
