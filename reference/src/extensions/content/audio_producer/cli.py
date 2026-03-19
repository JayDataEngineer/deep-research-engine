"""Audio Producer CLI Plugin

Provides commands for audio production functionality.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_produce(args):
    """Produce audio from input."""
    try:
        from src.extensions.libraries.audio_producer.manager import AudioProducerManager
        manager = AudioProducerManager()
        
        if not args.input_file:
            print("Please provide an input file for audio production")
            return {"error": "No input file provided"}
        
        # Produce audio
        result = manager.produce_audio(
            input_file=args.input_file,
            output_file=args.output_file,
            voice=args.voice,
            emotion_tags=args.emotion_tags
        )
        
        if result:
            print(f"Audio production completed. Output saved to: {result}")
            return {"output_file": result}
        else:
            print("Audio production failed")
            return {"error": "Audio production failed"}
    except Exception as e:
        print(f"Error in audio production: {e}")
        return {"error": str(e)}


def cmd_merge(args):
    """Merge audio files."""
    try:
        from src.extensions.libraries.audio_producer.manager import AudioProducerManager
        manager = AudioProducerManager()
        
        if not args.files or len(args.files) < 2:
            print("Please provide at least 2 audio files to merge")
            return {"error": "Not enough files to merge"}
        
        # Merge audio files
        result = manager.merge_audio_files(args.files, args.output_file)
        
        if result:
            print(f"Audio merge completed. Output saved to: {result}")
            return {"output_file": result}
        else:
            print("Audio merge failed")
            return {"error": "Audio merge failed"}
    except Exception as e:
        print(f"Error merging audio files: {e}")
        return {"error": str(e)}


def register(parser):
    """Register audio producer commands with the parser."""
    subparsers = parser.add_subparsers(dest="audio_subcommand", help="Audio producer commands")
    
    # Produce command
    produce_parser = subparsers.add_parser("produce", help="Produce audio from input")
    produce_parser.add_argument("input_file", help="Input file for audio production")
    produce_parser.add_argument("--output-file", "-o", help="Output file path")
    produce_parser.add_argument("--voice", "-v", help="Voice to use for audio production")
    produce_parser.add_argument("--emotion-tags", "-e", nargs="*", help="Emotion tags for audio production")
    produce_parser.set_defaults(func=cmd_produce)
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge audio files")
    merge_parser.add_argument("files", nargs="+", help="Audio files to merge")
    merge_parser.add_argument("--output-file", "-o", help="Output file path")
    merge_parser.set_defaults(func=cmd_merge)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Audio Producer CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()