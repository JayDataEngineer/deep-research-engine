"""CLI Parser Manager

Handles argument parsing, command routing, and help display.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class ParserManager:
    """Manages CLI argument parsing and command routing."""

    def __init__(self, description: str = "Unified LLM Interface - Command Center"):
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.subparsers = self.parser.add_subparsers(
            dest="command",
            help="Available commands"
        )
        self.command_handlers: Dict[str, Callable] = {}
        self.command_aliases: Dict[str, str] = {}  # alias -> canonical name

    def register_command(
        self,
        name: str,
        handler: Callable,
        aliases: Optional[List[str]] = None,
        help_text: Optional[str] = None
    ):
        """
        Register a command handler.

        Args:
            name: Command name
            handler: Function to call when command is invoked
            aliases: Optional list of command aliases
            help_text: Optional help text for the command
        """
        self.command_handlers[name] = handler

        # Register aliases
        if aliases:
            for alias in aliases:
                self.command_aliases[alias] = name

        # Add subparser for this command
        if help_text:
            self.subparsers.add_parser(name, help=help_text, add_help=False)
            for alias in aliases or []:
                self.subparsers.add_parser(alias, help=f"Alias for {name}", add_help=False)

    def register_subcommand(
        self,
        parent: str,
        name: str,
        handler: Callable,
        aliases: Optional[List[str]] = None
    ):
        """
        Register a subcommand under a parent command.

        Args:
            parent: Parent command name
            name: Subcommand name
            handler: Function to call
            aliases: Optional list of aliases
        """
        full_name = f"{parent}_{name}"
        self.command_handlers[full_name] = handler

        # Register aliases for subcommand
        if aliases:
            for alias in aliases:
                self.command_aliases[f"{parent}_{alias}"] = full_name

    def parse_args(self, args: List[str] = None):
        """Parse command line arguments."""
        parsed = self.parser.parse_args(args)
        return parsed

    def resolve_command(self, command: str) -> Optional[str]:
        """
        Resolve a command name to its canonical form (handles aliases).

        Args:
            command: Command name or alias

        Returns:
            Canonical command name, or None if not found
        """
        # Check direct match
        if command in self.command_handlers:
            return command

        # Check aliases
        if command in self.command_aliases:
            return self.command_aliases[command]

        return None

    async def execute(self, args: List[str] = None):
        """
        Parse and execute a command.

        Args:
            args: Command line arguments (defaults to sys.argv[1:])
        """
        parsed = self.parse_args(args)

        if not parsed.command:
            self.parser.print_help()
            return

        # Resolve command (handle aliases)
        command = self.resolve_command(parsed.command)
        if not command:
            print(f"Unknown command: {parsed.command}")
            self.print_available_commands()
            sys.exit(1)

        # Get handler
        handler = self.command_handlers.get(command)
        if not handler:
            print(f"No handler for command: {command}")
            sys.exit(1)

        # Execute handler
        try:
            result = handler(parsed)
            if asyncio.iscoroutine(result):
                await result
        except KeyboardInterrupt:
            print("\nInterrupted")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Error executing command: {command}")
            print(f"Error: {e}")
            sys.exit(1)

    def print_available_commands(self):
        """Print all available commands."""
        print("\nAvailable commands:")
        for name in sorted(self.command_handlers.keys()):
            print(f"  {name}")


class ExtensionCLIManager:
    """Manager for extension-specific CLI commands."""

    def __init__(self, extension_name: str, parser: argparse.ArgumentParser):
        self.extension_name = extension_name
        self.parser = parser
        self.subparsers = parser.add_subparsers(
            dest=f"{extension_name}_subcommand",
            help=f"{extension_name} commands"
        )
        self.handlers: Dict[str, Callable] = {}

    def add_command(
        self,
        name: str,
        handler: Callable,
        aliases: Optional[List[str]] = None,
        help_text: Optional[str] = None
    ):
        """
        Add a command to this extension's CLI.

        Args:
            name: Command name
            handler: Handler function
            aliases: Optional aliases
            help_text: Help text
        """
        self.handlers[name] = handler

        # Create subparser for this command
        cmd_parser = self.subparsers.add_parser(
            name,
            aliases=aliases or [],
            help=help_text,
            add_help=False
        )

        # Store parser for adding arguments later
        return cmd_parser

    def get_handler(self, command: str) -> Optional[Callable]:
        """Get handler for a command."""
        return self.handlers.get(command)
