"""Author Notes Manager

Manages author's notes functionality.
Refactored to use BaseManager and StorageManager.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class AuthorNote(BaseItem):
    """An author's note."""

    id: str
    name: str
    content: str
    character_filter: str = ""  # Optional: only show for this character

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "character_filter": self.character_filter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthorNote":
        return cls(**data)


class AuthorNotesManager(BaseManager):
    """Manages author's notes functionality."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the author notes manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="author_notes", storage_file="notes.json")

    def _get_item_class(self):
        """Get the AuthorNote item class."""
        return AuthorNote

    # ========================================================================
    # Note-specific methods
    # ========================================================================

    def get_note(self, name: str) -> Optional[AuthorNote]:
        """Get a note by name.

        Args:
            name: Note name

        Returns:
            AuthorNote or None
        """
        return self._items.get(name)

    def add_note(self, name: str, content: str, character_filter: str = "") -> bool:
        """Add a new author note.

        Args:
            name: Note name
            content: Note content
            character_filter: Optional character filter

        Returns:
            True if added
        """
        note = AuthorNote(
            id=StorageManager.generate_id_from_name(name),
            name=name,
            content=content,
            character_filter=character_filter,
        )
        return self.add_item(note)

    def update_note(self, name: str, content: Optional[str] = None, character_filter: Optional[str] = None) -> bool:
        """Update a note.

        Args:
            name: Note name
            content: New content (optional)
            character_filter: New character filter (optional)

        Returns:
            True if updated
        """
        note = self._items.get(name)
        if not note:
            return False

        if content is not None:
            note.content = content
        if character_filter is not None:
            note.character_filter = character_filter

        return self.update_item(note)

    def remove_note(self, name: str) -> bool:
        """Remove a note.

        Args:
            name: Note name

        Returns:
            True if removed
        """
        return self.delete_item(name)

    def get_notes_for_character(self, character_name: str) -> List[AuthorNote]:
        """Get notes for a specific character.

        Args:
            character_name: Character name

        Returns:
            List of notes for this character
        """
        return [
            note for note in self._items.values()
            if not note.character_filter or note.character_filter == character_name
        ]

    def search_notes(self, query: str) -> List[AuthorNote]:
        """Search notes by name or content.

        Args:
            query: Search query

        Returns:
            Matching notes
        """
        query = query.lower()
        return [
            n for n in self._items.values()
            if query in n.name.lower() or query in n.content.lower()
        ]