"""Presets CLI Plugin

Provides commands for managing generation parameter presets.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all presets."""
    from src.extensions.features.text.presets.manager import PresetManager

    try:
        manager = PresetManager()
        presets = manager.list_items()

        print(f"\nPresets ({len(presets)} found):")
        for preset in presets:
            print(f"  - {preset.name}: {preset.description}")

        return {"presets": [p.to_dict() for p in presets]}
    except Exception as e:
        print(f"Error listing presets: {e}")
        return {"error": str(e)}


def cmd_show(args):
    """Show a specific preset."""
    from src.extensions.features.text.presets.manager import PresetManager

    try:
        manager = PresetManager()
        preset = manager.get_preset(args.name)

        if not preset:
            print(f"Preset '{args.name}' not found")
            return {"error": "Not found"}

        print(f"\nPreset: {preset.name}")
        print(f"  Description: {preset.description}")
        print(f"  Parameters:")
        for key, value in preset.parameters.items():
            print(f"    {key}: {value}")

        return {"preset": preset.to_dict()}
    except Exception as e:
        print(f"Error showing preset: {e}")
        return {"error": str(e)}


def cmd_create(args):
    """Create a new preset."""
    from src.extensions.features.text.presets.manager import PresetManager

    try:
        manager = PresetManager()

        # Create preset with provided parameters
        parameters = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
        }

        # Add any extra parameters
        if args.extra:
            for pair in args.extra.split(','):
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    parameters[key.strip()] = val.strip()

        success = manager.create_preset(args.name, parameters, args.description or "")

        if success:
            print(f"Created preset '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to create preset '{args.name}'")
            return {"error": "Creation failed"}
    except Exception as e:
        print(f"Error creating preset: {e}")
        return {"error": str(e)}


def cmd_delete(args):
    """Delete a preset."""
    from src.extensions.features.text.presets.manager import PresetManager

    try:
        manager = PresetManager()
        success = manager.delete_item(args.name)

        if success:
            print(f"Deleted preset '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to delete preset '{args.name}'")
            return {"error": "Deletion failed"}
    except Exception as e:
        print(f"Error deleting preset: {e}")
        return {"error": str(e)}


def register(parser):
    """Register preset commands with the parser."""
    subparsers = parser.add_subparsers(dest="preset_subcommand", help="Preset commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all presets")
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show a specific preset")
    show_parser.add_argument("name", help="Name of the preset")
    show_parser.set_defaults(func=cmd_show)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new preset")
    create_parser.add_argument("name", help="Name of the preset")
    create_parser.add_argument("--description", "-d", help="Description of the preset")
    create_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter")
    create_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p parameter")
    create_parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens parameter")
    create_parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    create_parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    create_parser.add_argument("--extra", "-e", help="Extra parameters (key=val,key2=val2)")
    create_parser.set_defaults(func=cmd_create)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a preset")
    delete_parser.add_argument("name", help="Name of the preset to delete")
    delete_parser.set_defaults(func=cmd_delete)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Presets CLI")
    register(parser)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()