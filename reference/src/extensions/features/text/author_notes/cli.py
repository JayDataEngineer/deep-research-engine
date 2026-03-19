"""Author Notes CLI Plugin

Provides commands for managing author's notes.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all author notes."""
    from src.extensions.features.text.author_notes.manager import AuthorNotesManager
    
    try:
        manager = AuthorNotesManager()
        notes = manager.list_notes()
        
        print(f"\nAuthor Notes ({len(notes)} found):")
        for note in notes:
            print(f"  - {note}")
        
        return {"notes": notes}
    except Exception as e:
        print(f"Error listing author notes: {e}")
        return {"error": str(e)}


def cmd_add(args):
    """Add a new author note."""
    from src.extensions.features.text.author_notes.manager import AuthorNotesManager
    
    try:
        manager = AuthorNotesManager()
        success = manager.add_note(args.name, args.content)
        
        if success:
            print(f"Added author note '{args.name}': {args.content}")
            return {"success": True}
        else:
            print(f"Failed to add author note '{args.name}'")
            return {"error": "Add failed"}
    except Exception as e:
        print(f"Error adding author note: {e}")
        return {"error": str(e)}


def cmd_remove(args):
    """Remove an author note."""
    from src.extensions.features.text.author_notes.manager import AuthorNotesManager
    
    try:
        manager = AuthorNotesManager()
        success = manager.remove_note(args.name)
        
        if success:
            print(f"Removed author note '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to remove author note '{args.name}'")
            return {"error": "Remove failed"}
    except Exception as e:
        print(f"Error removing author note: {e}")
        return {"error": str(e)}


def register(parser):
    """Register author note commands with the parser."""
    subparsers = parser.add_subparsers(dest="author_note_subcommand", help="Author note commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all author notes")
    list_parser.set_defaults(func=cmd_list)
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new author note")
    add_parser.add_argument("name", help="Name of the author note")
    add_parser.add_argument("content", help="Content of the author note")
    add_parser.set_defaults(func=cmd_add)
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove an author note")
    remove_parser.add_argument("name", help="Name of the author note to remove")
    remove_parser.set_defaults(func=cmd_remove)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Author Notes CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()