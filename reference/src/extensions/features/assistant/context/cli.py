"""Context CLI Plugin

Provides commands for managing context/conversation.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_info(args):
    """Show context information."""
    from src.extensions.features.assistant.context.manager import ContextManager
    
    try:
        manager = ContextManager()
        info = manager.get_context_info()
        
        print(f"\nContext Information:")
        print(f"  Current Length: {info.get('current_length', 0)}")
        print(f"  Max Length: {info.get('max_length', 0)}")
        print(f"  Segments: {info.get('segment_count', 0)}")
        print(f"  Status: {info.get('status', 'unknown')}")
        
        return {"context_info": info}
    except Exception as e:
        print(f"Error getting context info: {e}")
        return {"error": str(e)}


def cmd_segments(args):
    """Show context segments."""
    from src.extensions.features.assistant.context.manager import ContextManager
    
    try:
        manager = ContextManager()
        segments = manager.get_segments()
        
        print(f"\nContext Segments ({len(segments)} found):")
        for i, segment in enumerate(segments):
            print(f"  [{i}] {segment.get('summary', 'No summary')}")
        
        return {"segments": segments}
    except Exception as e:
        print(f"Error getting context segments: {e}")
        return {"error": str(e)}


def cmd_trim(args):
    """Trim context."""
    from src.extensions.features.assistant.context.manager import ContextManager
    
    try:
        manager = ContextManager()
        success = manager.trim_context(args.target_length)
        
        if success:
            print(f"Trimmed context to {args.target_length} tokens")
            return {"success": True}
        else:
            print(f"Failed to trim context")
            return {"error": "Trim failed"}
    except Exception as e:
        print(f"Error trimming context: {e}")
        return {"error": str(e)}


def register(parser):
    """Register context commands with the parser."""
    subparsers = parser.add_subparsers(dest="context_subcommand", help="Context commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show context information")
    info_parser.set_defaults(func=cmd_info)
    
    # Segments command
    segments_parser = subparsers.add_parser("segments", help="Show context segments")
    segments_parser.set_defaults(func=cmd_segments)
    
    # Trim command
    trim_parser = subparsers.add_parser("trim", help="Trim context to target length")
    trim_parser.add_argument("target_length", type=int, help="Target length in tokens")
    trim_parser.set_defaults(func=cmd_trim)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Context CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()