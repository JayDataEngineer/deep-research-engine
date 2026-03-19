"""Macros Manager

Manages text macros/shortcuts functionality.
Refactored to use BaseManager and StorageManager.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class Macro(BaseItem):
    """A text macro/shortcut."""

    id: str
    name: str
    value: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Macro":
        return cls(**data)


class MacroManager(BaseManager):
    """Manages text macros/shortcuts functionality."""

    # Default macros
    DEFAULT_MACROS = {
        "time": {"id": "time", "name": "time", "value": "The current time is {{TIME}}", "description": "Insert current time"},
        "date": {"id": "date", "name": "date", "value": "Today's date is {{DATE}}", "description": "Insert current date"},
    }

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the macro manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="macros", storage_file="macros.json")

        # Load defaults if empty
        if not self._items:
            self._load_defaults()

    def _load_defaults(self):
        """Load default macros."""
        for macro_data in self.DEFAULT_MACROS.values():
            macro = Macro.from_dict(macro_data)
            self._items[macro.id] = macro
        self._save()

    def _get_item_class(self):
        """Get the Macro item class."""
        return Macro

    # ========================================================================
    # Macro-specific methods
    # ========================================================================

    def get_macro(self, name: str) -> Optional[Macro]:
        """Get a macro by name.

        Args:
            name: Macro name

        Returns:
            Macro or None
        """
        return self._items.get(name)

    def get_macro_value(self, name: str) -> Optional[str]:
        """Get a macro's value by name.

        Args:
            name: Macro name

        Returns:
            Macro value or None
        """
        macro = self._items.get(name)
        return macro.value if macro else None

    def expand_macro(self, name: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Expand a macro with context variables using Jinja2.

        Args:
            name: Macro name
            context: Optional context dict for variable substitution

        Returns:
            Expanded value or None
        """
        from jinja2 import Environment, BaseLoader
        from datetime import datetime

        value = self.get_macro_value(name)
        if not value:
            return None

        # Build context from all macro values (for macro composition)
        template_context = {m.name: m.value for m in self._items.values()}

        # Add runtime context
        if context:
            template_context.update(context)

        # Add common values
        template_context.setdefault('TIME', datetime.now().strftime("%H:%M:%S"))
        template_context.setdefault('DATE', datetime.now().strftime("%Y-%m-%d"))
        template_context.setdefault('NOW', datetime.now().isoformat())

        try:
            env = Environment(loader=BaseLoader())
            template = env.from_string(value)
            return template.render(**template_context)
        except Exception as e:
            logger.warning(f"Failed to expand macro '{name}': {e}")
            # Fall back to simple replacement
            for key, val in (context or {}).items():
                value = value.replace(f"{{{{{key}}}}}", str(val))
            return value

    def search_macros(self, query: str) -> List[Macro]:
        """Search macros by name, description, or value.

        Args:
            query: Search query

        Returns:
            Matching macros
        """
        query = query.lower()
        return [
            m for m in self._items.values()
            if query in m.name.lower() or query in m.description.lower() or query in m.value.lower()
        ]