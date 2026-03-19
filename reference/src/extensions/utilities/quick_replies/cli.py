"""Quick Replies CLI Plugin

Provides commands for managing quick response buttons.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all quick replies."""
    from src.extensions.features.quick_replies.manager import QuickReplyManager
    
    try:
        manager = QuickReplyManager()
        replies = manager.list_replies()
        
        print(f"\nQuick Replies ({len(replies)} found):")
        for name, reply in replies.items():
            print(f"  - {name}: {reply}")
        
        return {"replies": replies}
    except Exception as e:
        print(f"Error listing quick replies: {e}")
        return {"error": str(e)}


def cmd_add(args):
    """Add a new quick reply."""
    from src.extensions.features.quick_replies.manager import QuickReplyManager
    
    try:
        manager = QuickReplyManager()
        success = manager.add_reply(args.name, args.text)
        
        if success:
            print(f"Added quick reply '{args.name}': {args.text}")
            return {"success": True}
        else:
            print(f"Failed to add quick reply '{args.name}'")
            return {"error": "Add failed"}
    except Exception as e:
        print(f"Error adding quick reply: {e}")
        return {"error": str(e)}


def cmd_remove(args):
    """Remove a quick reply."""
    from src.extensions.features.quick_replies.manager import QuickReplyManager
    
    try:
        manager = QuickReplyManager()
        success = manager.remove_reply(args.name)
        
        if success:
            print(f"Removed quick reply '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to remove quick reply '{args.name}'")
            return {"error": "Remove failed"}
    except Exception as e:
        print(f"Error removing quick reply: {e}")
        return {"error": str(e)}


def register(parser):
    """Register quick reply commands with the parser."""
    subparsers = parser.add_subparsers(dest="quick_reply_subcommand", help="Quick reply commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all quick replies")
    list_parser.set_defaults(func=cmd_list)
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new quick reply")
    add_parser.add_argument("name", help="Name of the quick reply")
    add_parser.add_argument("text", help="Text of the quick reply")
    add_parser.set_defaults(func=cmd_add)
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a quick reply")
    remove_parser.add_argument("name", help="Name of the quick reply to remove")
    remove_parser.set_defaults(func=cmd_remove)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Quick Replies CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()