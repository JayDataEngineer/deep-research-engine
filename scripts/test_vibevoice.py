#!/usr/bin/env python3
"""Test VibeVoice ASR integration.

Usage:
    uv run python scripts/test_vibevoice.py /path/to/audio.mp3
    uv run python scripts/test_vibevoice.py --help
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional

import click
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deep_research_engine.utils.multimodal import asr_transcribe
from deep_research_engine.workers.implementations import VibeVoiceASRWorker
from deep_research_engine.workers.base import WorkerConfig


logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


async def test_utility(audio_path: Path, include_diarization: bool = True):
    """Test the asr_transcribe utility function."""
    logger.info(f"Testing asr_transcribe() with: {audio_path.name}")
    logger.info(f"Diarization enabled: {include_diarization}")
    
    result = await asr_transcribe(audio_path, include_diarization=include_diarization)
    
    if isinstance(result, dict):
        logger.success(f"✅ Transcription successful")
        logger.info(f"📊 Speakers detected: {result.get('speakers', 0)}")
        logger.info(f"⏱️  Duration: {result.get('duration', 0):.1f}s")
        logger.info(f"📝 Segments: {len(result.get('segments', []))}")
        
        if include_diarization:
            logger.info("\n=== Speaker Segments ===")
            for i, seg in enumerate(result.get('segments', [])[:5], 1):
                speaker = seg.get('speaker', 'Unknown')
                start = seg.get('start', 0)
                text = seg.get('text', '')[:80]
                logger.info(f"{i}. {speaker} [{start:.1f}s]: {text}...")
            
            if len(result.get('segments', [])) > 5:
                logger.info(f"... and {len(result['segments']) - 5} more segments")
        
        return result
    else:
        logger.success(f"✅ Transcription successful")
        logger.info(f"📝 Length: {len(result)} chars")
        logger.info(f"\n=== Preview ===\n{result[:200]}...")
        return result


async def test_worker(audio_path: Path):
    """Test the VibeVoiceASRWorker."""
    logger.info(f"Testing VibeVoiceASRWorker with: {audio_path.name}")
    
    worker = VibeVoiceASRWorker()
    config = WorkerConfig(
        worker_type="vibevoice_asr",
        custom={
            "audio_path": str(audio_path),
            "min_speakers": 1,
            "max_speakers": 10,
        }
    )
    
    state = {"context": {"audio_path": str(audio_path)}}
    
    result = await worker.execute(state, config)
    
    if result.success:
        logger.success("✅ Worker execution successful")
        output = result.output
        
        if isinstance(output, dict):
            logger.info(f"📊 Speakers: {output.get('speakers', 0)}")
            logger.info(f"⏱️  Duration: {output.get('duration', 0):.1f}s")
            logger.info(f"📝 Segments: {len(output.get('segments', []))}")
            logger.info(f"🏷️  Diarization: {output.get('diarization', 'unknown')}")
            
            logger.info("\n=== First 3 Segments ===")
            for i, seg in enumerate(output.get('segments', [])[:3], 1):
                speaker = seg.get('speaker', 'Unknown')
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                text = seg.get('text', '')[:60]
                logger.info(f"{i}. {speaker} [{start:.1f}s - {end:.1f}s]: {text}...")
        else:
            logger.info(f"📝 Output: {output}")
        
        return result
    else:
        logger.error(f"❌ Worker failed: {result.error}")
        return result


@click.command()
@click.argument("audio_path", type=click.Path(exists=True, path_type=Path))
@click.option("--mode", "-m", type=click.Choice(["utility", "worker", "both"]), default="both",
              help="Test mode: utility, worker, or both")
@click.option("--no-diarization", is_flag=True, help="Disable diarization for utility test")
@click.option("--output-json", type=click.Path(path_type=Path), help="Save full result to JSON file")
def main(audio_path: Path, mode: str, no_diarization: bool, output_json: Optional[Path]):
    """Test VibeVoice ASR integration with your audio files."""
    logger.info("🎙️  VibeVoice ASR Integration Test")
    logger.info("=" * 50)
    
    audio_path = audio_path.resolve()
    logger.info(f"Audio file: {audio_path.name}")
    logger.info(f"Size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("")
    
    results = {}
    
    if mode in ["utility", "both"]:
        logger.info("=" * 50)
        logger.info("MODE: Utility Function")
        logger.info("=" * 50)
        utility_result = asyncio.run(test_utility(audio_path, include_diarization=not no_diarization))
        results["utility"] = utility_result
        logger.info("")
    
    if mode in ["worker", "both"]:
        logger.info("=" * 50)
        logger.info("MODE: Worker")
        logger.info("=" * 50)
        worker_result = asyncio.run(test_worker(audio_path))
        results["worker"] = worker_result
        logger.info("")
    
    if output_json:
        # Convert results to JSON-serializable format
        json_results = {}
        for key, result in results.items():
            if hasattr(result, "__dict__"):
                json_results[key] = result.__dict__
            elif isinstance(result, dict):
                json_results[key] = result
            else:
                json_results[key] = {"output": str(result)}
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, default=str)
        logger.info(f"💾 Results saved to: {output_json}")
    
    logger.info("=" * 50)
    logger.info("✅ Test complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
