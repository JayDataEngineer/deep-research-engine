"""Variables CLI Plugin

Provides commands for managing variables.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all variables."""
    from src.extensions.features.text.variables.manager import VariableManager

    try:
        manager = VariableManager()
        variables = manager.list_items()

        print(f"\nVariables ({len(variables)} found):")
        for var in variables:
            print(f"  - {var.name}: {var.value} ({var.scope})")

        return {"variables": [v.to_dict() for v in variables]}
    except Exception as e:
        print(f"Error listing variables: {e}")
        return {"error": str(e)}


def cmd_show(args):
    """Show a specific variable."""
    from src.extensions.features.text.variables.manager import VariableManager

    try:
        manager = VariableManager()
        var = manager.get_variable(args.name)

        if not var:
            print(f"Variable '{args.name}' not found")
            return {"error": "Not found"}

        print(f"\nVariable: {var.name}")
        print(f"  Value: {var.value}")
        print(f"  Description: {var.description}")
        print(f"  Scope: {var.scope}")

        return {"variable": var.to_dict()}
    except Exception as e:
        print(f"Error showing variable: {e}")
        return {"error": str(e)}


def cmd_set(args):
    """Set a variable."""
    from src.extensions.features.text.variables.manager import VariableManager

    try:
        manager = VariableManager()
        success = manager.set_variable(args.name, args.value, args.description or "", args.scope)

        if success:
            print(f"Set variable '{args.name}' to '{args.value}'")
            return {"success": True}
        else:
            print(f"Failed to set variable '{args.name}'")
            return {"error": "Set failed"}
    except Exception as e:
        print(f"Error setting variable: {e}")
        return {"error": str(e)}


def cmd_delete(args):
    """Delete a variable."""
    from src.extensions.features.text.variables.manager import VariableManager

    try:
        manager = VariableManager()
        success = manager.delete_variable(args.name)

        if success:
            print(f"Deleted variable '{args.name}'")
            return {"success": True}
        else:
            print(f"Failed to delete variable '{args.name}'")
            return {"error": "Delete failed"}
    except Exception as e:
        print(f"Error deleting variable: {e}")
        return {"error": str(e)}


def cmd_replace(args):
    """Replace variables in text."""
    from src.extensions.features.text.variables.manager import VariableManager

    try:
        manager = VariableManager()

        # Get text from argument or stdin
        text = args.text
        if not text and not args.file:
            print("Please provide text or a file to replace variables in")
            return {"error": "No input provided"}

        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()

        # Build context
        context = None
        if args.context:
            context = {}
            for pair in args.context.split(','):
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    context[key.strip()] = val.strip()

        processed_text = manager.replace_variables(text, context)

        print(f"Processed text:\n{processed_text}")

        return {"processed_text": processed_text}
    except Exception as e:
        print(f"Error replacing variables: {e}")
        return {"error": str(e)}


def register(parser):
    """Register variable commands with the parser."""
    subparsers = parser.add_subparsers(dest="variable_subcommand", help="Variable commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all variables")
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show a specific variable")
    show_parser.add_argument("name", help="Name of the variable")
    show_parser.set_defaults(func=cmd_show)

    # Set command
    set_parser = subparsers.add_parser("set", help="Set a variable")
    set_parser.add_argument("name", help="Name of the variable")
    set_parser.add_argument("value", help="Value of the variable")
    set_parser.add_argument("--description", "-d", help="Description")
    set_parser.add_argument("--scope", "-s", default="global", help="Variable scope")
    set_parser.set_defaults(func=cmd_set)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a variable")
    delete_parser.add_argument("name", help="Name of the variable")
    delete_parser.set_defaults(func=cmd_delete)

    # Replace command
    replace_parser = subparsers.add_parser("replace", help="Replace variables in text")
    replace_parser.add_argument("text", nargs="?", help="Text to replace variables in")
    replace_parser.add_argument("--file", "-f", help="File to read text from")
    replace_parser.add_argument("--context", "-c", help="Context variables (key=val,key2=val2)")
    replace_parser.set_defaults(func=cmd_replace)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Variables CLI")
    register(parser)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()