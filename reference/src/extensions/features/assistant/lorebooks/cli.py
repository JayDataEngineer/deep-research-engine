"""Lorebooks CLI Plugin

Provides commands for managing lorebooks/world info.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all lorebooks."""
    from src.extensions.features.assistant.lorebooks.manager import LorebookManager
    
    try:
        manager = LorebookManager()
        lorebooks = manager.list_lorebooks()
        
        print(f"\nLorebooks ({len(lorebooks)} found):")
        for lorebook in lorebooks:
            print(f"  - {lorebook}")
        
        return {"lorebooks": lorebooks}
    except Exception as e:
        print(f"Error listing lorebooks: {e}")
        return {"error": str(e)}


def cmd_create(args):
    """Create a new lorebook."""
    from src.extensions.features.assistant.lorebooks.manager import LorebookManager
    
    try:
        manager = LorebookManager()
        
        # Create lorebook with provided parameters
        success = manager.create_lorebook(
            name=args.name,
            description=args.description or "",
            entries=args.entries or []
        )
        
        if success:
            print(f"Created lorebook '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to create lorebook '{args.name}'")
            return {"error": "Creation failed"}
    except Exception as e:
        print(f"Error creating lorebook: {e}")
        return {"error": str(e)}


def cmd_delete(args):
    """Delete a lorebook."""
    from src.extensions.features.assistant.lorebooks.manager import LorebookManager
    
    try:
        manager = LorebookManager()
        success = manager.delete_lorebook(args.name)
        
        if success:
            print(f"Deleted lorebook '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to delete lorebook '{args.name}'")
            return {"error": "Deletion failed"}
    except Exception as e:
        print(f"Error deleting lorebook: {e}")
        return {"error": str(e)}


def register(parser):
    """Register lorebook commands with the parser."""
    subparsers = parser.add_subparsers(dest="lorebook_subcommand", help="Lorebook commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all lorebooks")
    list_parser.set_defaults(func=cmd_list)
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new lorebook")
    create_parser.add_argument("name", help="Name of the lorebook")
    create_parser.add_argument("--description", "-d", help="Description of the lorebook")
    create_parser.add_argument("--entries", "-e", nargs="*", help="Initial entries for the lorebook")
    create_parser.set_defaults(func=cmd_create)
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a lorebook")
    delete_parser.add_argument("name", help="Name of the lorebook to delete")
    delete_parser.set_defaults(func=cmd_delete)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Lorebooks CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()