"""Inlines CLI Plugin

Provides commands for managing inline scripts/JavaScript.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all inline scripts."""
    from src.extensions.features.text.inlines.manager import InlineScriptManager
    
    try:
        manager = InlineScriptManager()
        scripts = manager.list_scripts()
        
        print(f"\nInline Scripts ({len(scripts)} found):")
        for script in scripts:
            print(f"  - {script}")
        
        return {"scripts": scripts}
    except Exception as e:
        print(f"Error listing inline scripts: {e}")
        return {"error": str(e)}


def cmd_add(args):
    """Add a new inline script."""
    from src.extensions.features.text.inlines.manager import InlineScriptManager
    
    try:
        manager = InlineScriptManager()
        success = manager.add_script(args.name, args.code)
        
        if success:
            print(f"Added inline script '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to add inline script '{args.name}'")
            return {"error": "Add failed"}
    except Exception as e:
        print(f"Error adding inline script: {e}")
        return {"error": str(e)}


def cmd_remove(args):
    """Remove an inline script."""
    from src.extensions.features.text.inlines.manager import InlineScriptManager
    
    try:
        manager = InlineScriptManager()
        success = manager.remove_script(args.name)
        
        if success:
            print(f"Removed inline script '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to remove inline script '{args.name}'")
            return {"error": "Remove failed"}
    except Exception as e:
        print(f"Error removing inline script: {e}")
        return {"error": str(e)}


def register(parser):
    """Register inline script commands with the parser."""
    subparsers = parser.add_subparsers(dest="inline_subcommand", help="Inline script commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all inline scripts")
    list_parser.set_defaults(func=cmd_list)
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new inline script")
    add_parser.add_argument("name", help="Name of the inline script")
    add_parser.add_argument("code", help="Code of the inline script")
    add_parser.set_defaults(func=cmd_add)
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove an inline script")
    remove_parser.add_argument("name", help="Name of the inline script to remove")
    remove_parser.set_defaults(func=cmd_remove)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Inlines CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()