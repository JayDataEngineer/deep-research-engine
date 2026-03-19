"""Backgrounds CLI Plugin

Provides commands for managing background images.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all backgrounds."""
    from src.extensions.features.backgrounds.manager import BackgroundManager
    
    try:
        manager = BackgroundManager()
        backgrounds = manager.list_backgrounds()
        
        print(f"\nBackgrounds ({len(backgrounds)} found):")
        for bg in backgrounds:
            print(f"  - {bg}")
        
        return {"backgrounds": backgrounds}
    except Exception as e:
        print(f"Error listing backgrounds: {e}")
        return {"error": str(e)}


def cmd_set(args):
    """Set a background."""
    from src.extensions.features.backgrounds.manager import BackgroundManager
    
    try:
        manager = BackgroundManager()
        success = manager.set_background(args.name)
        
        if success:
            print(f"Set background to '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to set background to '{args.name}'")
            return {"error": "Set failed"}
    except Exception as e:
        print(f"Error setting background: {e}")
        return {"error": str(e)}


def cmd_random(args):
    """Set a random background."""
    from src.extensions.features.backgrounds.manager import BackgroundManager
    
    try:
        manager = BackgroundManager()
        bg_name = manager.set_random_background()
        
        if bg_name:
            print(f"Set random background to '{bg_name}'")
            return {"background": bg_name}
        else:
            print(f"Failed to set random background")
            return {"error": "Random selection failed"}
    except Exception as e:
        print(f"Error setting random background: {e}")
        return {"error": str(e)}


def register(parser):
    """Register background commands with the parser."""
    subparsers = parser.add_subparsers(dest="background_subcommand", help="Background commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all backgrounds")
    list_parser.set_defaults(func=cmd_list)
    
    # Set command
    set_parser = subparsers.add_parser("set", help="Set a background")
    set_parser.add_argument("name", help="Name of the background to set")
    set_parser.set_defaults(func=cmd_set)
    
    # Random command
    random_parser = subparsers.add_parser("random", help="Set a random background")
    random_parser.set_defaults(func=cmd_random)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Backgrounds CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()