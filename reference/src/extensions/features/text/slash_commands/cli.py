"""Slash Commands CLI Plugin

Provides commands for managing comprehensive slash commands.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all slash commands."""
    from src.extensions.features.text.slash_commands.manager import SlashCommandManager
    
    try:
        manager = SlashCommandManager()
        commands = manager.list_commands()
        
        print(f"\nSlash Commands ({len(commands)} found):")
        for cmd in commands:
            print(f"  - /{cmd}")
        
        return {"commands": commands}
    except Exception as e:
        print(f"Error listing slash commands: {e}")
        return {"error": str(e)}


def cmd_execute(args):
    """Execute a slash command."""
    from src.extensions.features.text.slash_commands.manager import SlashCommandManager
    
    try:
        manager = SlashCommandManager()
        
        # Parse command and arguments
        cmd_parts = args.command.split()
        cmd_name = cmd_parts[0].lstrip('/')
        cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []
        
        result = manager.execute_command(cmd_name, cmd_args)
        
        print(f"Executed command /{cmd_name} with args {cmd_args}")
        print(f"Result: {result}")
        
        return {"result": result}
    except Exception as e:
        print(f"Error executing slash command: {e}")
        return {"error": str(e)}


def register(parser):
    """Register slash command commands with the parser."""
    subparsers = parser.add_subparsers(dest="slash_command_subcommand", help="Slash command commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all slash commands")
    list_parser.set_defaults(func=cmd_list)
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute a slash command")
    execute_parser.add_argument("command", help="Command to execute (with arguments)")
    execute_parser.set_defaults(func=cmd_execute)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Slash Commands CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()