#!/usr/bin/env python3
"""
Test All TTS Extensions

This script tests all available TTS extensions by generating a simple test audio file.
Audio outputs are saved to: data/outputs/tts_tests/

Usage:
    python -m src.extensions.engines.tts_engine.test_all_tts
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
script_dir = Path(__file__).parent
src_root = script_dir.parent.parent.parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from src.extensions.core import ExtensionManager, ExtensionContext
from src.config import settings
from src.shared.factory import get_llm

# Setup logging
logger = get_logger(__name__)


async def test_tts_extension(manager: ExtensionManager, ext_name: str, test_text: str, output_dir: Path) -> dict:
    """Test a single TTS extension.

    Args:
        manager: ExtensionManager instance
        ext_name: Name of the TTS extension to test
        test_text: Text to synthesize
        output_dir: Directory to save audio outputs

    Returns:
        Dict with test results
    """
    result = {
        "name": ext_name,
        "success": False,
        "error": None,
        "audio_path": None,
        "duration": None,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {ext_name}")
    logger.info(f"{'='*60}")

    try:
        ext = manager.get_extension(ext_name)

        if ext is None:
            result["error"] = "Extension not found"
            logger.error(f"Extension '{ext_name}' not found")
            return result

        # Check if it implements TTS interface
        from src.extensions.core import TTSExtensionInterface
        if not isinstance(ext, TTSExtensionInterface):
            result["error"] = "Extension does not implement TTSExtensionInterface"
            logger.error(f"{ext_name} does not implement TTSExtensionInterface")
            return result

        # Get available voices
        try:
            voices = ext.get_voices()
            logger.info(f"Available voices ({len(voices)}):")
            for voice in voices[:5]:  # Show first 5
                if hasattr(voice, 'id'):
                    logger.info(f"  - {voice.id}: {getattr(voice, 'name', 'N/A')}")
                elif isinstance(voice, dict):
                    logger.info(f"  - {voice.get('id', 'N/A')}: {voice.get('name', 'N/A')}")
            if len(voices) > 5:
                logger.info(f"  ... and {len(voices) - 5} more")
        except Exception as e:
            logger.warning(f"Could not get voices: {e}")
            voices = []

        # Use first available voice or a default
        voice_id = None
        reference_audio = None

        if voices:
            voice_id = voices[0].id if hasattr(voices[0], 'id') else voices[0].get('id')
            logger.info(f"Using voice: {voice_id}")

        # Check for reference audio support
        # Some TTS extensions need reference audio (e.g., GPT-SoVITS)
        ext_config = ext.get_config() if hasattr(ext, 'get_config') else {}

        # Try to generate audio
        logger.info(f"Synthesizing: '{test_text[:50]}...'")

        try:
            # Try with emotion vector for IndexTTS
            if ext_name == "index_tts":
                audio_bytes, duration = ext.generate_audio(
                    text=test_text,
                    voice_id=voice_id or "default",
                    emo_vector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],  # neutral
                )
            else:
                audio_bytes, duration = ext.generate_audio(
                    text=test_text,
                    voice_id=voice_id or "default",
                )

            # Save audio
            output_path = output_dir / f"{ext_name}_test.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

            result["success"] = True
            result["audio_path"] = str(output_path)
            result["duration"] = duration

            logger.info(f"✅ SUCCESS")
            logger.info(f"   Output: {output_path}")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   Size: {len(audio_bytes)} bytes")

        except TypeError as e:
            # Try different parameter signatures
            logger.warning(f"First attempt failed: {e}")
            try:
                audio_bytes, duration = ext.generate_audio(test_text)
                output_path = output_dir / f"{ext_name}_test.wav"
                output_path.write_bytes(audio_bytes)

                result["success"] = True
                result["audio_path"] = str(output_path)
                result["duration"] = duration

                logger.info(f"✅ SUCCESS (fallback)")
                logger.info(f"   Output: {output_path}")
                logger.info(f"   Duration: {duration:.2f}s")
            except Exception as e2:
                result["error"] = str(e2)
                logger.error(f"❌ FAILED (both attempts): {e2}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ FAILED: {e}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ FAILED (setup): {e}")

    return result


async def main():
    """Main test runner."""
    logger.info("="*60)
    logger.info("TTS Extensions Test Suite")
    logger.info("="*60)

    # Test text
    test_text = "Hello, this is a test of the TTS system. We are checking if all text-to-speech extensions work correctly."

    # Output directory (using constants)
    from src.constants import DATA_OUTPUTS_DIR
    output_base = DATA_OUTPUTS_DIR / "tts_tests"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Test text: '{test_text}'")

    # Initialize extension manager
    logger.info("\nInitializing ExtensionManager...")
    extensions_dir = src_root / "src" / "extensions"

    manager = ExtensionManager()
    context = ExtensionContext(
        llm_factory=get_llm,
        settings=settings,
        manager=manager,
    )
    manager.set_context(context)

    await manager.load_all(extensions_dir)

    # Get TTS extensions
    tts_extensions = []
    for name, ext in manager.extensions.items():
        from src.extensions.core import TTSExtensionInterface
        if isinstance(ext, TTSExtensionInterface):
            tts_extensions.append(name)

    logger.info(f"\nFound {len(tts_extensions)} TTS extensions: {tts_extensions}")

    # Test each extension
    results = []
    for ext_name in tts_extensions:
        result = await test_tts_extension(manager, ext_name, test_text, output_dir)
        results.append(result)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    logger.info(f"\nResults: {successful}/{len(results)} passed\n")

    for result in results:
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} {result['name']}")
        if result["success"]:
            logger.info(f"   Output: {result['audio_path']}")
            logger.info(f"   Duration: {result['duration']:.2f}s")
        else:
            logger.info(f"   Error: {result['error']}")

    # Print successful audio paths
    if successful > 0:
        logger.info(f"\n📁 All audio outputs saved to: {output_dir}")
        logger.info("\nSuccessful outputs:")
        for result in results:
            if result["success"]:
                logger.info(f"  - {result['audio_path']}")

    return failed == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)
