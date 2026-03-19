"""CLI interface for content formatter extension.

Usage:
    manage.py content_formatter format <content> --format <name> [--session-id <id>]
    manage.py content_formatter list
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def register(parser):
    """Register CLI commands for content formatter.

    Args:
        parser: ArgumentParser subparser
    """
    subparser = parser.add_subparsers(dest="command", help="Content formatter commands")

    # Format command
    format_cmd = subparser.add_parser(
        "format",
        help="Format content into a specific format"
    )
    format_cmd.add_argument(
        "content",
        help="Content to format (use '-' for stdin)"
    )
    format_cmd.add_argument(
        "--format", "-f",
        required=True,
        help="Format to use (tweets, podcast, substack, etc.)"
    )
    format_cmd.add_argument(
        "--session-id",
        help="Session ID for Context Engine access (optional)"
    )
    format_cmd.add_argument(
        "--character",
        help="Character name for character-based formats (optional)"
    )
    format_cmd.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Number of retries (default: 10)"
    )
    format_cmd.set_defaults(func=handle_format)

    # List formats command
    list_cmd = subparser.add_parser(
        "list",
        help="List available formats"
    )
    list_cmd.set_defaults(func=handle_list)


async def handle_format(args):
    """Handle format command.

    Args:
        args: Parsed arguments
    """
    from .extension import deep_formatter

    # Get content
    if args.content == "-":
        content = sys.stdin.read()
    else:
        content = args.content

    print(f"Formatting content ({len(content)} chars) into '{args.format}' format...")

    try:
        result = await deep_formatter(
            config_selection=args.format,
            content=content,
            session_id=args.session_id,
            retries=args.retries,
            character=args.character
        )

        print("\n" + "=" * 60)
        print("FORMATTED RESULT:")
        print("=" * 60)
        print(result.model_dump_json(indent=2))

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_list(args):
    """Handle list command.

    Args:
        args: Parsed arguments
    """
    from src.extensions.core.manager.loader import get_extension_manager

    try:
        manager = get_extension_manager()
        if not manager:
            print("Error: Extension manager not available")
            sys.exit(1)

        formatter_ext = manager.get_extension("content_formatter")
        if not formatter_ext or not formatter_ext.enabled:
            print("Error: Content formatter extension not available or not enabled")
            sys.exit(1)

        formatter = formatter_ext.instance
        formats = formatter.list_formats()

        print("\n" + "=" * 60)
        print("AVAILABLE FORMATS:")
        print("=" * 60)
        for fmt in formats:
            print(f"  - {fmt}")

        if not formats:
            print("  (No formats found)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# For synchronous invocation
def handle_format_sync(args):
    """Synchronous wrapper for format command."""
    asyncio.run(handle_format(args))
