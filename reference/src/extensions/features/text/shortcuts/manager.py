"""Keyboard Shortcuts Manager

Manages keyboard shortcut configurations.
Backend storage for keybinding settings (frontend would apply these).
Refactored to use BaseManager and ManagerWithBuiltinItems.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.extensions.libraries.base.manager_base import BaseItem, ManagerWithBuiltinItems

logger = logging.getLogger(__name__)


@dataclass
class Shortcut(BaseItem):
    """A keyboard shortcut configuration."""
    id: str
    keys: List[str]  # e.g., ["ctrl", "k"] or ["ctrl", "shift", "s"]
    command: str  # Command to execute
    description: str = ""
    context: str = "global"  # global, chat, editor, etc.
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "keys": self.keys,
            "command": self.command,
            "description": self.description,
            "context": self.context,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Shortcut":
        return cls(**data)

    def get_key_combo(self) -> str:
        """Get the key combination string."""
        return "+".join(k.upper() for k in self.keys)


class ShortcutsManager(ManagerWithBuiltinItems):
    """Manages keyboard shortcut configurations using BaseManager infrastructure."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the shortcuts manager.

        Args:
            base_dir: Base directory for storage
        """
        super().__init__(base_dir=base_dir, feature_name="shortcuts", storage_file="shortcuts.json")

    def _get_item_class(self):
        """Get the Shortcut item class."""
        return Shortcut

    def _load_builtin_items(self):
        """Load built-in shortcut definitions."""
        builtin_shortcuts = [
            Shortcut(
                id="chat.send",
                keys=["enter"],
                command="chat.send",
                description="Send message",
                context="chat",
            ),
            Shortcut(
                id="chat.new_line",
                keys=["shift", "enter"],
                command="chat.new_line",
                description="New line in chat input",
                context="chat",
            ),
            Shortcut(
                id="chat.focus",
                keys=["ctrl", "k"],
                command="chat.focus",
                description="Focus chat input",
                context="global",
            ),
            Shortcut(
                id="chat.clear",
                keys=["ctrl", "l"],
                command="chat.clear",
                description="Clear chat input",
                context="chat",
            ),
            Shortcut(
                id="nav.history_back",
                keys=["alt", "left"],
                command="nav.history_back",
                description="Go back in history",
                context="global",
            ),
            Shortcut(
                id="nav.history_forward",
                keys=["alt", "right"],
                command="nav.history_forward",
                description="Go forward in history",
                context="global",
            ),
            Shortcut(
                id="edit.find",
                keys=["ctrl", "f"],
                command="edit.find",
                description="Find in editor",
                context="editor",
            ),
            Shortcut(
                id="edit.save",
                keys=["ctrl", "s"],
                command="edit.save",
                description="Save",
                context="editor",
            ),
            Shortcut(
                id="app.settings",
                keys=["ctrl", ","],
                command="app.settings",
                description="Open settings",
                context="global",
            ),
            Shortcut(
                id="app.help",
                keys=["f1"],
                command="app.help",
                description="Show help",
                context="global",
            ),
        ]
        self._builtin_items = {s.id: s for s in builtin_shortcuts}

    def _save(self):
        """Save only custom (non-built-in) shortcuts to storage."""
        custom_items = [
            item.to_dict()
            for item_id, item in self._items.items()
            if not self.is_builtin(item_id)
        ]
        StorageManager.save_json_file(self.storage_file, custom_items)

    # ========================================================================
    # Shortcut-specific convenience methods
    # ========================================================================

    def list_shortcuts(self, context: Optional[str] = None) -> List[Shortcut]:
        """List all shortcuts.

        Args:
            context: Filter by context (optional)

        Returns:
            List of shortcuts
        """
        shortcuts = list(self._items.values())

        if context:
            shortcuts = [s for s in shortcuts if s.context == context or s.context == "global"]

        return sorted(shortcuts, key=lambda s: (s.context, s.command))

    def get_shortcut(self, shortcut_id: str) -> Optional[Shortcut]:
        """Get a shortcut by ID.

        Args:
            shortcut_id: Shortcut ID

        Returns:
            Shortcut or None
        """
        return self.get_item(shortcut_id)

    def create_shortcut(
        self,
        id: str,
        keys: List[str],
        command: str,
        description: str = "",
        context: str = "global",
    ) -> Shortcut:
        """Create a new shortcut.

        Args:
            id: Shortcut ID
            keys: Key combination
            command: Command to execute
            description: Description
            context: Shortcut context

        Returns:
            Created Shortcut
        """
        if self.item_exists(id):
            raise ValueError(f"Shortcut already exists: {id}")

        # Check for key conflicts
        key_combo = "+".join(k.lower() for k in keys)
        for s in self._items.values():
            if s.get_key_combo().lower() == key_combo and s.context == context:
                logger.warning(f"Key conflict: {key_combo} already bound to {s.id}")

        shortcut = Shortcut(
            id=id,
            keys=keys,
            command=command,
            description=description,
            context=context,
        )

        self.add_item(shortcut)
        return shortcut

    def update_shortcut(
        self,
        shortcut_id: str,
        keys: Optional[List[str]] = None,
        command: Optional[str] = None,
        description: Optional[str] = None,
        context: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[Shortcut]:
        """Update a shortcut.

        Args:
            shortcut_id: Shortcut ID
            keys: New keys (optional)
            command: New command (optional)
            description: New description (optional)
            context: New context (optional)
            enabled: New enabled state (optional)

        Returns:
            Updated Shortcut or None
        """
        shortcut = self.get_item(shortcut_id)
        if not shortcut:
            return None

        if keys is not None:
            shortcut.keys = keys
        if command is not None:
            shortcut.command = command
        if description is not None:
            shortcut.description = description
        if context is not None:
            shortcut.context = context
        if enabled is not None:
            shortcut.enabled = enabled

        self.update_item(shortcut)
        return shortcut

    def delete_shortcut(self, shortcut_id: str) -> bool:
        """Delete a custom shortcut.

        Args:
            shortcut_id: Shortcut ID

        Returns:
            True if deleted
        """
        return self.delete_item(shortcut_id)

    def enable_shortcut(self, shortcut_id: str) -> bool:
        """Enable a shortcut.

        Args:
            shortcut_id: Shortcut ID

        Returns:
            True if enabled
        """
        shortcut = self.get_item(shortcut_id)
        if shortcut:
            shortcut.enabled = True
            self.update_item(shortcut)
            return True
        return False

    def disable_shortcut(self, shortcut_id: str) -> bool:
        """Disable a shortcut.

        Args:
            shortcut_id: Shortcut ID

        Returns:
            True if disabled
        """
        shortcut = self.get_item(shortcut_id)
        if shortcut:
            shortcut.enabled = False
            self.update_item(shortcut)
            return True
        return False

    # ========================================================================
    # Lookup methods
    # ========================================================================

    def find_by_keys(self, keys: List[str], context: str = "global") -> Optional[Shortcut]:
        """Find a shortcut by key combination.

        Args:
            keys: Key combination
            context: Context to search

        Returns:
            Matching Shortcut or None
        """
        key_combo = "+".join(k.lower() for k in keys)

        # First try exact context match
        for shortcut in self._items.values():
            if (
                shortcut.get_key_combo().lower() == key_combo
                and shortcut.context == context
                and shortcut.enabled
            ):
                return shortcut

        # Then try global shortcuts
        for shortcut in self._items.values():
            if (
                shortcut.get_key_combo().lower() == key_combo
                and shortcut.context == "global"
                and shortcut.enabled
            ):
                return shortcut

        return None

    def get_conflicts(
        self, keys: List[str], context: str, exclude_id: Optional[str] = None
    ) -> List[Shortcut]:
        """Find shortcuts that conflict with given keys.

        Args:
            keys: Key combination
            context: Context
            exclude_id: ID to exclude from check

        Returns:
            List of conflicting shortcuts
        """
        key_combo = "+".join(k.lower() for k in keys)
        conflicts = []

        for shortcut in self._items.values():
            if (
                shortcut.id != exclude_id
                and shortcut.get_key_combo().lower() == key_combo
                and shortcut.context == context
                and shortcut.enabled
            ):
                conflicts.append(shortcut)

        return conflicts

    # ========================================================================
    # Export/Import
    # ========================================================================

    def export_shortcuts(self, output_path: str):
        """Export shortcuts to a file.

        Args:
            output_path: Output file path
        """
        data = {
            "shortcuts": [s.to_dict() for s in self._items.values()],
            "exported_at": str(Path(__file__).stat().st_mtime),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Shortcuts exported to {output_path}")

    def import_shortcuts(self, filepath: str, merge: bool = True):
        """Import shortcuts from a file.

        Args:
            filepath: Path to shortcuts JSON file
            merge: If True, merge with existing; if False, replace all
        """
        with open(filepath) as f:
            data = json.load(f)

        if not merge:
            # Clear only custom items, keep built-ins
            custom_ids = [id for id in self._items.keys() if not self.is_builtin(id)]
            for id in custom_ids:
                del self._items[id]

        for shortcut_data in data.get("shortcuts", []):
            shortcut = Shortcut.from_dict(shortcut_data)
            self._items[shortcut.id] = shortcut

        self._save()
        logger.info(f"Shortcuts imported from {filepath}")

    def reset_to_defaults(self):
        """Reset all shortcuts to built-in defaults."""
        # Clear only custom items, keep built-ins
        custom_ids = [id for id in self._items.keys() if not self.is_builtin(id)]
        for id in custom_ids:
            del self._items[id]
        self._save()


# Import StorageManager at the end to avoid circular import
from src.utils.storage import StorageManager
