"""TTS CLI Plugin

Provides commands for text-to-speech functionality.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from src.constants import DATA_OUTPUTS_DIR


def cmd_tts(args):
    """Convert text to speech."""
    from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager

    try:
        # Get text from argument or stdin
        text = args.text
        if not text and not args.file:
            print("Please provide text or a file to convert to speech")
            return {"error": "No input provided"}

        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()

        # Start IndexTTS if not running
        import asyncio

        async def generate():
            # Start IndexTTS
            result = await extension_process_manager.start_extension("index_tts")
            if result.get("status") != "ready":
                return {"error": f"Failed to start IndexTTS: {result.get('message', 'Unknown error')}"}

            # Generate speech
            voice_id = args.voice or "index2_ref_neutral"
            audio_bytes, duration = await extension_process_manager.generate_audio(
                text=text,
                voice_id=voice_id,
            )

            # Save to file
            output_path = args.output or f"/tmp/tts_output_{hash(text) % 1000000}.wav"
            Path(output_path).write_bytes(audio_bytes)

            # Stop extension
            await extension_process_manager.stop_current()

            return {"output_file": output_path, "duration": duration}

        result = asyncio.run(generate())

        if result.get("output_file"):
            print(f"TTS completed. Output saved to: {result['output_file']}")
            print(f"Duration: {result['duration']:.2f}s")
            return result
        else:
            print(f"TTS failed: {result.get('error')}")
            return result
    except Exception as e:
        import traceback
        print(f"Error in TTS: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def cmd_podcast(args):
    """Convert markdown to podcast audio."""
    import asyncio
    from pathlib import Path

    try:
        if not args.input_file:
            print("Please provide an input file for podcast conversion")
            return {"error": "No input file provided"}

        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Input file not found: {args.input_file}")
            return {"error": f"File not found: {args.input_file}"}

        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()

        # Import and use the audio_producer extension
        from src.extensions.content.audio_producer.extension import AudioProducerExtension
        import yaml

        # Load the actual config from YAML
        config_path = Path("src/extensions/features/audio_producer/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        async def run_podcast():
            # Create extension instance with proper config
            extension = AudioProducerExtension(
                config=config,
                root_path=Path("src/extensions/features/audio_producer"),
            )

            # Handle output path - create timestamped run directory if not specified
            output_path = None
            if args.output_file:
                output_path = Path(args.output_file)
            else:
                # Create timestamped run directory
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_output_dir = DATA_OUTPUTS_DIR / "audio/podcast"
                run_dir = base_output_dir / f"podcast_{timestamp}"
                run_dir.mkdir(parents=True, exist_ok=True)
                output_path = run_dir / "podcast_full.wav"
                print(f"Created run directory: {run_dir}")

            # Process the podcast
            result = await extension.process_audio(input_text, output_path)

            # Cleanup
            await extension.shutdown()

            return result

        # Run the async function
        result = asyncio.run(run_podcast())

        if result and result.get("audio_path"):
            print(f"Podcast conversion completed!")
            print(f"  Audio: {result['audio_path']}")
            if result.get("output_md_path"):
                print(f"  Report: {result['output_md_path']}")
            print(f"  Duration: {result.get('duration', 0):.2f}s")
            print(f"  Lines processed: {result.get('num_lines', 0)}")
            print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
            return result
        else:
            print("Podcast conversion failed")
            return {"error": "Podcast conversion failed"}

    except Exception as e:
        import traceback
        print(f"Error in podcast conversion: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def cmd_audio(args):
    """Audio production functionality."""
    print("Audio production with emotion tagging is now handled by the audio_producer extension.")
    print("Use: manage.py audio_producer produce <input_file> --output <output_file>")
    return {"error": "Use audio_producer extension instead"}


def register(parser):
    """Register TTS commands with the parser."""
    subparsers = parser.add_subparsers(dest="tts_subcommand", help="TTS commands")
    
    # TTS command
    tts_parser = subparsers.add_parser("tts", help="Convert text to speech")
    tts_parser.add_argument("text", nargs="?", help="Text to convert to speech")
    tts_parser.add_argument("--file", "-f", help="File to read text from")
    tts_parser.add_argument("--voice", "-v", help="Voice to use for TTS")
    tts_parser.add_argument("--output", "-o", help="Output file path")
    tts_parser.set_defaults(func=cmd_tts)
    
    # Podcast command
    podcast_parser = subparsers.add_parser("podcast", help="Convert markdown to podcast audio")
    podcast_parser.add_argument("input_file", help="Input markdown file")
    podcast_parser.add_argument("--output-file", "-o", help="Output file path")
    podcast_parser.set_defaults(func=cmd_podcast)
    
    # Audio command
    audio_parser = subparsers.add_parser("audio", help="Audio production with emotion tagging")
    audio_parser.add_argument("input_file", help="Input file")
    audio_parser.add_argument("--output-file", "-o", help="Output file path")
    audio_parser.add_argument("--emotion-tags", "-e", nargs="*", help="Emotion tags for audio production")
    audio_parser.set_defaults(func=cmd_audio)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="TTS CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()