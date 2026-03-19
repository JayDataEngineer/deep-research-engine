"""Slash Commands Manager

Manages comprehensive slash commands functionality.
Refactored to use BaseManager and StorageManager.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager

logger = logging.getLogger(__name__)


@dataclass
class SlashCommand(BaseItem):
    """A slash command definition."""

    id: str
    name: str
    description: str
    handler: Optional[str] = None  # Function name to call
    aliases: List[str] = None
    enabled: bool = True

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

    def to_dict(self) -> Dict[str, Any]:
        # Can't serialize handler function, so store name
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "handler": self.handler,
            "aliases": self.aliases,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlashCommand":
        return cls(**data)


class SlashCommandManager(BaseManager):
    """Manages slash commands functionality."""

    # Built-in commands
    BUILTIN_COMMANDS = {
        "help": SlashCommand(
            id="help",
            name="help",
            description="Show help information",
            aliases=["h", "?"],
        ),
        "clear": SlashCommand(
            id="clear",
            name="clear",
            description="Clear the current conversation",
            aliases=["c", "reset"],
        ),
        "history": SlashCommand(
            id="history",
            name="history",
            description="Show conversation history",
            aliases=["hist"],
        ),
    }

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the slash command manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="slash_commands", storage_file="commands.json")

        # Register built-in commands
        for cmd in self.BUILTIN_COMMANDS.values():
            if cmd.id not in self._items:
                self._items[cmd.id] = cmd

        if not self._items:
            self._save()

    def _get_item_class(self):
        """Get the SlashCommand item class."""
        return SlashCommand

    # ========================================================================
    # Command-specific methods
    # ========================================================================

    def get_command(self, name: str) -> Optional[SlashCommand]:
        """Get a command by name or alias.

        Args:
            name: Command name or alias

        Returns:
            SlashCommand or None
        """
        # Try direct lookup
        cmd = self._items.get(name)
        if cmd:
            return cmd

        # Try alias lookup
        for cmd in self._items.values():
            if name in cmd.aliases and cmd.enabled:
                return cmd

        return None

    def register_command(self, name: str, description: str, handler: Optional[str] = None, aliases: List[str] = None) -> bool:
        """Register a new command.

        Args:
            name: Command name
            description: Command description
            handler: Handler function name
            aliases: Optional aliases

        Returns:
            True if registered
        """
        cmd = SlashCommand(
            id=name,
            name=name,
            description=description,
            handler=handler,
            aliases=aliases or [],
        )
        return self.add_item(cmd)

    def unregister_command(self, name: str) -> bool:
        """Unregister a command (won't remove built-ins).

        Args:
            name: Command name

        Returns:
            True if unregistered
        """
        if name in self.BUILTIN_COMMANDS:
            return False

        return self.delete_item(name)

    def execute_command(self, cmd_name: str, args: List[str] = None) -> Optional[str]:
        """Execute a command (stub - actual execution handled elsewhere).

        Args:
            cmd_name: Command name
            args: Command arguments

        Returns:
            Result message or None
        """
        cmd = self.get_command(cmd_name)
        if not cmd:
            return f"Unknown command: /{cmd_name}"

        if not cmd.enabled:
            return f"Command disabled: /{cmd_name}"

        # Return the handler name - actual execution is done by the caller
        return cmd.handler if cmd.handler else f"Executed: /{cmd_name}"

    def list_commands(self, include_disabled: bool = False) -> List[SlashCommand]:
        """List all commands.

        Args:
            include_disabled: Include disabled commands

        Returns:
            List of commands
        """
        if include_disabled:
            return list(self._items.values())
        return [c for c in self._items.values() if c.enabled]

    def toggle_command(self, name: str) -> Optional[bool]:
        """Toggle a command's enabled state.

        Args:
            name: Command name

        Returns:
            New enabled state or None
        """
        cmd = self.get_command(name)
        if not cmd:
            return None

        # Find the actual command by id (not by alias)
        for c in self._items.values():
            if c.id == cmd.id:
                c.enabled = not c.enabled
                self.update_item(c)
                return c.enabled

        return None

    def search_commands(self, query: str) -> List[SlashCommand]:
        """Search commands by name or description.

        Args:
            query: Search query

        Returns:
            Matching commands
        """
        query = query.lower()
        return [
            c for c in self._items.values()
            if query in c.name.lower() or query in c.description.lower()
        ]