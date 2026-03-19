"""Inlines Manager

Manages inline scripts/JavaScript functionality.
Refactored to use BaseManager and StorageManager.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class InlineScript(BaseItem):
    """An inline JavaScript script."""

    id: str
    name: str
    code: str
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InlineScript":
        return cls(**data)


class InlineScriptManager(BaseManager):
    """Manages inline scripts/JavaScript functionality."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the inline script manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="inline_scripts", storage_file="scripts.json")

    def _get_item_class(self):
        """Get the InlineScript item class."""
        return InlineScript

    # ========================================================================
    # Script-specific methods
    # ========================================================================

    def get_script(self, name: str) -> Optional[InlineScript]:
        """Get a script by name.

        Args:
            name: Script name

        Returns:
            InlineScript or None
        """
        return self._items.get(name)

    def get_enabled_scripts(self) -> List[InlineScript]:
        """Get all enabled scripts.

        Returns:
            List of enabled scripts
        """
        return [s for s in self._items.values() if s.enabled]

    def add_script(self, name: str, code: str, description: str = "") -> bool:
        """Add a new inline script.

        Args:
            name: Name of the script
            code: JavaScript code
            description: Optional description

        Returns:
            True if added
        """
        script = InlineScript(
            id=StorageManager.generate_id_from_name(name),
            name=name,
            code=code,
            description=description,
        )
        return self.add_item(script)

    def update_script_code(self, name: str, code: str) -> bool:
        """Update a script's code.

        Args:
            name: Script name
            code: New code

        Returns:
            True if updated
        """
        script = self._items.get(name)
        if not script:
            return False

        script.code = code
        return self.update_item(script)

    def toggle_script(self, name: str) -> Optional[bool]:
        """Toggle a script's enabled state.

        Args:
            name: Script name

        Returns:
            New enabled state or None
        """
        script = self._items.get(name)
        if not script:
            return None

        script.enabled = not script.enabled
        self.update_item(script)
        return script.enabled

    def remove_script(self, name: str) -> bool:
        """Remove a script.

        Args:
            name: Script name

        Returns:
            True if removed
        """
        return self.delete_item(name)

    def search_scripts(self, query: str) -> List[InlineScript]:
        """Search scripts by name, description, or code.

        Args:
            query: Search query

        Returns:
            Matching scripts
        """
        query = query.lower()
        return [
            s for s in self._items.values()
            if query in s.name.lower() or query in s.description.lower() or query in s.code.lower()
        ]