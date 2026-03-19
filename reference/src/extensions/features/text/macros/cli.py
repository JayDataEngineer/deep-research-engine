"""Macros CLI Plugin

Provides commands for managing text macros/shortcuts.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all macros."""
    from src.extensions.features.text.macros.manager import MacroManager

    try:
        manager = MacroManager()
        macros = manager.list_items()

        print(f"\nMacros ({len(macros)} found):")
        for macro in macros:
            print(f"  - {macro.name}: {macro.value}")

        return {"macros": [m.to_dict() for m in macros]}
    except Exception as e:
        print(f"Error listing macros: {e}")
        return {"error": str(e)}


def cmd_show(args):
    """Show a specific macro."""
    from src.extensions.features.text.macros.manager import MacroManager

    try:
        manager = MacroManager()
        macro = manager.get_macro(args.name)

        if not macro:
            print(f"Macro '{args.name}' not found")
            return {"error": "Not found"}

        print(f"\nMacro: {macro.name}")
        print(f"  Value: {macro.value}")
        print(f"  Description: {macro.description}")

        return {"macro": macro.to_dict()}
    except Exception as e:
        print(f"Error showing macro: {e}")
        return {"error": str(e)}


def cmd_add(args):
    """Add a new macro."""
    from src.extensions.features.text.macros.manager import MacroManager

    try:
        manager = MacroManager()
        macro = manager.get_macro(args.name)

        if macro:
            print(f"Macro '{args.name}' already exists")
            return {"error": "Already exists"}

        from src.utils.storage import StorageManager
        from src.extensions.features.text.macros.manager import Macro

        macro = Macro(
            id=StorageManager.generate_id_from_name(args.name),
            name=args.name,
            value=args.value,
            description=args.description or "",
        )

        if manager.add_item(macro):
            print(f"Added macro '{args.name}': {args.value}")
            return {"success": True, "macro": macro.to_dict()}
        else:
            print(f"Failed to add macro '{args.name}'")
            return {"error": "Add failed"}
    except Exception as e:
        print(f"Error adding macro: {e}")
        return {"error": str(e)}


def cmd_remove(args):
    """Remove a macro."""
    from src.extensions.features.text.macros.manager import MacroManager

    try:
        manager = MacroManager()
        success = manager.delete_item(args.name)

        if success:
            print(f"Removed macro '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to remove macro '{args.name}'")
            return {"error": "Remove failed"}
    except Exception as e:
        print(f"Error removing macro: {e}")
        return {"error": str(e)}


def cmd_expand(args):
    """Expand a macro with context."""
    from src.extensions.features.text.macros.manager import MacroManager

    try:
        manager = MacroManager()
        context = {}
        if args.context:
            for pair in args.context.split(','):
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    context[key.strip()] = val.strip()

        result = manager.expand_macro(args.name, context if context else None)

        if result:
            print(f"\nExpanded: {result}")
            return {"expanded": result}
        else:
            print(f"Macro '{args.name}' not found")
            return {"error": "Not found"}
    except Exception as e:
        print(f"Error expanding macro: {e}")
        return {"error": str(e)}


def register(parser):
    """Register macro commands with the parser."""
    subparsers = parser.add_subparsers(dest="macro_subcommand", help="Macro commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all macros")
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show a specific macro")
    show_parser.add_argument("name", help="Name of the macro")
    show_parser.set_defaults(func=cmd_show)

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new macro")
    add_parser.add_argument("name", help="Name of the macro")
    add_parser.add_argument("value", help="Value of the macro")
    add_parser.add_argument("--description", "-d", help="Description")
    add_parser.set_defaults(func=cmd_add)

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a macro")
    remove_parser.add_argument("name", help="Name of the macro to remove")
    remove_parser.set_defaults(func=cmd_remove)

    # Expand command
    expand_parser = subparsers.add_parser("expand", help="Expand a macro")
    expand_parser.add_argument("name", help="Name of the macro")
    expand_parser.add_argument("--context", "-c", help="Context variables (key=val,key2=val2)")
    expand_parser.set_defaults(func=cmd_expand)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Macros CLI")
    register(parser)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()