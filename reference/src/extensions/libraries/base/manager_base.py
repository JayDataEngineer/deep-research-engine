"""Base Manager Infrastructure

Provides common storage and management patterns for feature implementations.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
from dataclasses import dataclass, asdict

from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseItem')


class BaseItem(ABC):
    """Base class for items that can be serialized to/from dict.

    All feature items should inherit from this for consistent serialization.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary.

        Returns:
            Dictionary representation
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseItem':
        """Create item from dictionary.

        Args:
            data: Dictionary data

        Returns:
            Item instance
        """
        pass

    @classmethod
    def get_id_field(cls) -> str:
        """Get the field name used as the ID.

        Returns:
            ID field name (default: "id")
        """
        return "id"

    def get_id(self) -> str:
        """Get the item's ID.

        Returns:
            Item ID
        """
        return getattr(self, self.get_id_field(), "")


@dataclass
class SimpleItem(BaseItem):
    """Simple item with just an ID and name.

    Can be used as a base for other items or directly for simple cases.
    """

    id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleItem':
        return cls(**data)


class BaseManager(ABC):
    """Base class for feature managers.

    Provides common patterns for:
    - Directory and file management
    - Loading/saving JSON data
    - ID generation
    - Error handling
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        feature_name: str = "",
        storage_file: str = None,
    ):
        """Initialize the base manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
            feature_name: Name of the feature
            storage_file: Name of the main storage file
        """
        self.feature_name = feature_name
        self.base_dir = self._get_base_dir(base_dir, feature_name)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.storage_file = self.base_dir / (storage_file or f"{feature_name}.json")

        self._items: Dict[str, Any] = {}
        self._load()

    def _get_base_dir(self, base_dir: Optional[Path], feature_name: str) -> Path:
        """Get the base directory for storage.

        Args:
            base_dir: Provided base directory or None
            feature_name: Feature name

        Returns:
            Base directory path
        """
        if base_dir is None:
            from src.constants import PROJECT_ROOT
            return PROJECT_ROOT / "data" / "inputs" / feature_name
        return Path(base_dir)

    # Data loading/saving

    def _load(self):
        """Load data from storage.

        Override _get_item_class() to specify the item class.
        """
        item_class = self._get_item_class()
        if item_class:
            items = StorageManager.load_json_file(self.storage_file, item_class)
            self._items = {item.get_id(): item for item in items}
        else:
            data = StorageManager.load_json_file(self.storage_file)
            self._items = data if isinstance(data, dict) else {}

        logger.info(f"{self.__class__.__name__} initialized: {len(self._items)} items")

    def _save(self):
        """Save data to storage."""
        items_list = list(self._items.values())
        StorageManager.save_json_file(self.storage_file, items_list)

    def _get_item_class(self) -> Optional[Type[BaseItem]]:
        """Get the item class for deserialization.

        Override this in subclasses to specify the item class.

        Returns:
            Item class or None
        """
        return None

    # CRUD operations

    def list_items(self) -> List[Any]:
        """List all items.

        Returns:
            List of items
        """
        return list(self._items.values())

    def get_item(self, item_id: str) -> Optional[Any]:
        """Get an item by ID.

        Args:
            item_id: Item identifier

        Returns:
            Item or None
        """
        return self._items.get(item_id)

    def add_item(self, item: BaseItem) -> bool:
        """Add an item.

        Args:
            item: Item to add

        Returns:
            True if added
        """
        item_id = item.get_id()
        if item_id in self._items:
            return False

        self._items[item_id] = item
        self._save()
        return True

    def update_item(self, item: BaseItem) -> bool:
        """Update an item.

        Args:
            item: Item to update

        Returns:
            True if updated
        """
        item_id = item.get_id()
        if item_id not in self._items:
            return False

        self._items[item_id] = item
        self._save()
        return True

    def delete_item(self, item_id: str) -> bool:
        """Delete an item.

        Args:
            item_id: Item identifier

        Returns:
            True if deleted
        """
        if item_id not in self._items:
            return False

        del self._items[item_id]
        self._save()
        return True

    # Utility methods

    def create_id(self, name: str) -> str:
        """Generate a safe ID from a name.

        Args:
            name: Name to convert

        Returns:
            Safe ID string
        """
        return StorageManager.generate_id_from_name(name)

    def item_exists(self, item_id: str) -> bool:
        """Check if an item exists.

        Args:
            item_id: Item identifier

        Returns:
            True if exists
        """
        return item_id in self._items

    def get_count(self) -> int:
        """Get the number of items.

        Returns:
            Item count
        """
        return len(self._items)

    def clear_all(self) -> int:
        """Clear all items.

        Returns:
            Number of items cleared
        """
        count = len(self._items)
        self._items.clear()
        self._save()
        return count


class ManagerWithBuiltinItems(BaseManager):
    """Base manager for features that have built-in items plus custom items.

    Examples: Themes, Shortcuts, etc. that have defaults but can be extended.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._builtin_items: Dict[str, Any] = {}
        self._load_builtin_items()
        # Built-ins are loaded first, then custom items can override
        for item_id, item in self._builtin_items.items():
            if item_id not in self._items:
                self._items[item_id] = item

    @abstractmethod
    def _load_builtin_items(self):
        """Load built-in items.

        Subclasses must implement this to populate self._builtin_items.
        """
        pass

    def is_builtin(self, item_id: str) -> bool:
        """Check if an item is built-in.

        Args:
            item_id: Item identifier

        Returns:
            True if built-in
        """
        return item_id in self._builtin_items

    def delete_item(self, item_id: str) -> bool:
        """Delete an item (only custom items, not built-ins).

        Args:
            item_id: Item identifier

        Returns:
            True if deleted
        """
        if self.is_builtin(item_id):
            return False

        return super().delete_item(item_id)

    def list_builtin_items(self) -> List[Any]:
        """List built-in items.

        Returns:
            List of built-in items
        """
        return list(self._builtin_items.values())

    def list_custom_items(self) -> List[Any]:
        """List custom (non-built-in) items.

        Returns:
            List of custom items
        """
        return [
            item for item_id, item in self._items.items()
            if not self.is_builtin(item_id)
        ]
