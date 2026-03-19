"""Presets Manager

Manages generation parameter presets functionality.
Refactored to use BaseManager and StorageManager.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class Preset(BaseItem):
    """A generation parameter preset."""

    id: str
    name: str
    parameters: Dict[str, Any]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "parameters": self.parameters,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        return cls(**data)


class PresetManager(BaseManager):
    """Manages generation parameter presets functionality."""

    # Common default presets
    DEFAULT_PRESETS = {
        "balanced": {
            "id": "balanced",
            "name": "Balanced",
            "description": "Balanced generation preset",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
        "creative": {
            "id": "creative",
            "name": "Creative",
            "description": "More creative generation",
            "parameters": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 50,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.2,
            },
        },
        "precise": {
            "id": "precise",
            "name": "Precise",
            "description": "More focused generation",
            "parameters": {
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
    }

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the preset manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="presets", storage_file="presets.json")

        # Load defaults if empty
        if not self._items:
            self._load_defaults()

    def _load_defaults(self):
        """Load default presets."""
        for preset_data in self.DEFAULT_PRESETS.values():
            preset = Preset.from_dict(preset_data)
            self._items[preset.id] = preset
        self._save()

    def _get_item_class(self):
        """Get the Preset item class."""
        return Preset

    # ========================================================================
    # Preset-specific methods
    # ========================================================================

    def get_preset(self, name: str) -> Optional[Preset]:
        """Get a preset by name.

        Args:
            name: Preset name

        Returns:
            Preset or None
        """
        return self._items.get(name)

    def get_preset_parameters(self, name: str) -> Optional[Dict[str, Any]]:
        """Get preset parameters by name.

        Args:
            name: Preset name

        Returns:
            Parameters dict or None
        """
        preset = self._items.get(name)
        return preset.parameters if preset else None

    def create_preset(self, name: str, parameters: Dict[str, Any], description: str = "") -> bool:
        """Create a new preset.

        Args:
            name: Preset name
            parameters: Generation parameters
            description: Optional description

        Returns:
            True if created
        """
        preset = Preset(
            id=StorageManager.generate_id_from_name(name),
            name=name,
            parameters=parameters,
            description=description,
        )
        return self.add_item(preset)

    def update_preset(self, name: str, parameters: Optional[Dict[str, Any]] = None, description: Optional[str] = None) -> bool:
        """Update a preset.

        Args:
            name: Preset name
            parameters: New parameters (optional)
            description: New description (optional)

        Returns:
            True if updated
        """
        preset = self._items.get(name)
        if not preset:
            return False

        if parameters is not None:
            preset.parameters = parameters
        if description is not None:
            preset.description = description

        return self.update_item(preset)

    def delete_preset(self, name: str) -> bool:
        """Delete a preset.

        Args:
            name: Preset name

        Returns:
            True if deleted
        """
        # Don't allow deleting built-in presets
        if name in self.DEFAULT_PRESETS:
            return False

        return self.delete_item(name)

    def search_presets(self, query: str) -> List[Preset]:
        """Search presets by name or description.

        Args:
            query: Search query

        Returns:
            Matching presets
        """
        query = query.lower()
        return [
            p for p in self._items.values()
            if query in p.name.lower() or query in p.description.lower()
        ]