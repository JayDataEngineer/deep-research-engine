"""Translation CLI Plugin

Provides commands for text translation.

Usage:
    uv run backend/manage.py translate translate <text> --to <language>
    uv run backend/manage.py translate languages
"""

import argparse
import asyncio
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the TranslateManager instance."""
    from src.extensions.features.text.translate.manager import TranslateManager
    return TranslateManager()


def cmd_translate(args):
    """Translate text."""
    manager = get_manager()

    # Get text from args or file
    if args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    else:
        text = args.text

    if not text:
        print("\n✗ Error: No text provided")
        return {"error": "No text"}

    async def _translate():
        print(f"\nTranslating to: {args.target_language}")
        print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        result = await manager.translate(
            text=text,
            target_language=args.target_language,
            source_language=args.from_lang or "auto"
        )

        print(f"\n  Translation: {result[:200]}{'...' if len(result) > 200 else ''}")
        return {"translation": result}

    return asyncio.run(_translate())


def cmd_languages(args):
    """List supported languages."""
    manager = get_manager()
    languages = manager.get_supported_languages()

    print("\nSupported Languages:")
    print("-" * 40)
    for i, lang in enumerate(languages, 1):
        print(f"  {lang}")
    print(f"\n  Total: {len(languages)} languages")

    return {"languages": languages}


def cmd_detect(args):
    """Detect language of text."""
    manager = get_manager()

    # Get text from args or file
    if args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    else:
        text = args.text

    detected = manager.detect_language(text)

    print(f"\nDetected Language: {detected}")
    print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")

    return {"detected_language": detected}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="translate_subcommand", help="Translation commands")

    # translate command
    trans_parser = subparsers.add_parser("translate", help="Translate text")
    trans_parser.add_argument("text", nargs="?", help="Text to translate")
    trans_parser.add_argument("--to", "-t", required=True, help="Target language")
    trans_parser.add_argument("--from", "-f", dest="from_lang", help="Source language (default: auto)")
    trans_parser.add_argument("--file", help="Read text from file")
    trans_parser.set_defaults(func=cmd_translate)

    # languages command
    lang_parser = subparsers.add_parser("languages", help="List supported languages")
    lang_parser.set_defaults(func=cmd_languages)

    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect language of text")
    detect_parser.add_argument("text", nargs="?", help="Text to analyze")
    detect_parser.add_argument("--file", help="Read text from file")
    detect_parser.set_defaults(func=cmd_detect)
