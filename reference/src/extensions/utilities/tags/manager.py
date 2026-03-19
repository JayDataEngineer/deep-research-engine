"""Tags Manager

Manages tags for characters, messages, and content.
Similar to SillyTavern's tagging system.

Refactored to use base infrastructure.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass

from src.extensions.libraries.base.manager_base import BaseItem
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class Tag(BaseItem):
    """A tag that can be applied to characters, messages, or content."""
    id: str
    name: str
    color: str = "#3b82f6"  # Default blue
    description: str = ""
    created_at: str = ""
    count: int = 0  # Number of items tagged

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "description": self.description,
            "created_at": self.created_at,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tag":
        return cls(**data)


class TagsManager:
    """Manages tags and tag assignments."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the tags manager.

        Args:
            base_dir: Base directory for tag storage
        """
        self.base_dir = StorageManager.get_project_subdir("tags") if base_dir is None else Path(base_dir)
        self.base_dir = StorageManager.ensure_directory(self.base_dir)

        self.tags_file = self.base_dir / "tags.json"
        self.assignments_file = self.base_dir / "assignments.json"

        self._tags: Dict[str, Tag] = {}
        self._assignments: Dict[str, Set[str]] = {}  # item_id -> set of tag_ids

        self._load()

    def _load(self):
        """Load tags and assignments from storage."""
        # Load tags
        tags_data = StorageManager.load_json_file(self.tags_file, Tag)
        self._tags = {tag.id: tag for tag in tags_data}

        # Load assignments
        if self.assignments_file.exists():
            try:
                with open(self.assignments_file) as f:
                    data = __import__('json').load(f)
                for item_id, tag_ids in data.items():
                    self._assignments[item_id] = set(tag_ids)
            except Exception as e:
                logger.warning(f"Failed to load assignments: {e}")

        logger.info(f"Tags Manager initialized: {len(self._tags)} tags, {len(self._assignments)} assignments")

    def _save(self):
        """Save tags and assignments to storage."""
        # Save tags
        StorageManager.save_json_file(self.tags_file, list(self._tags.values()))

        # Save assignments
        StorageManager.save_json_file(
            self.assignments_file,
            {k: list(v) for k, v in self._assignments.items()}
        )

    # Tag Management

    def create_tag(
        self,
        name: str,
        color: str = "#3b82f6",
        description: str = "",
    ) -> Tag:
        """Create a new tag.

        Args:
            name: Tag name
            color: Hex color code
            description: Tag description

        Returns:
            Created Tag
        """
        # Generate ID from name (slug-like)
        tag_id = StorageManager.generate_id_from_name(name)
        if tag_id in self._tags:
            raise ValueError(f"Tag already exists: {name}")

        tag = Tag(
            id=tag_id,
            name=name,
            color=color,
            description=description,
        )
        self._tags[tag_id] = tag
        self._save()
        return tag

    def list_tags(self) -> List[Tag]:
        """List all tags."""
        return sorted(self._tags.values(), key=lambda t: t.name)

    def get_tag(self, tag_id: str) -> Optional[Tag]:
        """Get a tag by ID."""
        return self._tags.get(tag_id)

    def update_tag(
        self,
        tag_id: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[Tag]:
        """Update a tag.

        Args:
            tag_id: Tag ID
            name: New name (optional)
            color: New color (optional)
            description: New description (optional)

        Returns:
            Updated Tag or None
        """
        tag = self._tags.get(tag_id)
        if not tag:
            return None

        if name is not None:
            tag.name = name
        if color is not None:
            tag.color = color
        if description is not None:
            tag.description = description

        self._save()
        return tag

    def delete_tag(self, tag_id: str) -> bool:
        """Delete a tag and remove all assignments.

        Args:
            tag_id: Tag ID

        Returns:
            True if deleted
        """
        if tag_id not in self._tags:
            return False

        del self._tags[tag_id]

        # Remove all assignments
        for item_id in self._assignments:
            self._assignments[item_id].discard(tag_id)

        self._save()
        return True

    # Tag Assignment

    def tag_item(self, item_id: str, tag_id: str) -> bool:
        """Apply a tag to an item.

        Args:
            item_id: ID of the item (character, message, etc.)
            tag_id: Tag ID to apply

        Returns:
            True if tagged
        """
        if tag_id not in self._tags:
            return False

        if item_id not in self._assignments:
            self._assignments[item_id] = set()

        if tag_id in self._assignments[item_id]:
            return False  # Already tagged

        self._assignments[item_id].add(tag_id)
        self._tags[tag_id].count += 1
        self._save()
        return True

    def untag_item(self, item_id: str, tag_id: str) -> bool:
        """Remove a tag from an item.

        Args:
            item_id: ID of the item
            tag_id: Tag ID to remove

        Returns:
            True if untagged
        """
        if item_id not in self._assignments:
            return False

        if tag_id not in self._assignments[item_id]:
            return False

        self._assignments[item_id].remove(tag_id)
        self._tags[tag_id].count = max(0, self._tags[tag_id].count - 1)
        self._save()
        return True

    def get_item_tags(self, item_id: str) -> List[Tag]:
        """Get all tags applied to an item.

        Args:
            item_id: ID of the item

        Returns:
            List of Tags
        """
        if item_id not in self._assignments:
            return []

        return [
            self._tags[tag_id]
            for tag_id in self._assignments[item_id]
            if tag_id in self._tags
        ]

    def find_items_by_tag(self, tag_id: str) -> List[str]:
        """Find all items with a specific tag.

        Args:
            tag_id: Tag ID to search

        Returns:
            List of item IDs
        """
        return [
            item_id
            for item_id, tags in self._assignments.items()
            if tag_id in tags
        ]

    def search_tags(self, query: str) -> List[Tag]:
        """Search tags by name or description.

        Args:
            query: Search query

        Returns:
            Matching tags
        """
        query = query.lower()
        return [
            tag
            for tag in self._tags.values()
            if query in tag.name.lower() or query in tag.description.lower()
        ]

    # Bulk Operations

    def bulk_tag(self, item_ids: List[str], tag_ids: List[str]) -> Dict[str, int]:
        """Apply multiple tags to multiple items.

        Args:
            item_ids: List of item IDs
            tag_ids: List of tag IDs

        Returns:
            Dict with counts: {"tagged": N, "skipped": M}
        """
        tagged = 0
        skipped = 0

        for item_id in item_ids:
            for tag_id in tag_ids:
                if self.tag_item(item_id, tag_id):
                    tagged += 1
                else:
                    skipped += 1

        return {"tagged": tagged, "skipped": skipped}

    def bulk_untag(self, item_ids: List[str], tag_ids: List[str]) -> Dict[str, int]:
        """Remove multiple tags from multiple items.

        Args:
            item_ids: List of item IDs
            tag_ids: List of tag IDs

        Returns:
            Dict with counts
        """
        untagged = 0
        skipped = 0

        for item_id in item_ids:
            for tag_id in tag_ids:
                if self.untag_item(item_id, tag_id):
                    untagged += 1
                else:
                    skipped += 1

        return {"untagged": untagged, "skipped": skipped}

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get tag statistics.

        Returns:
            Stats dict
        """
        return {
            "total_tags": len(self._tags),
            "total_assignments": sum(len(tags) for tags in self._assignments.values()),
            "most_used_tags": sorted(
                self._tags.values(),
                key=lambda t: t.count,
                reverse=True
            )[:10],
            "untagged_items": len(
                [id for id in self._assignments if not self._assignments[id]]
            ),
        }
