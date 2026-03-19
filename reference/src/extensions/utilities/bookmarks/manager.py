"""Bookmarks & Snapshots Manager

Manages chat state bookmarks for saving and restoring chat positions.
Similar to SillyTavern's bookmarks feature.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from src.extensions.features.assistant.chat_history.storage import get_chat_storage

logger = logging.getLogger(__name__)


class Bookmark:
    """A chat state bookmark."""

    def __init__(
        self,
        bookmark_id: str,
        chat_name: str,
        message_index: int,
        name: str = "",
        notes: str = "",
        created_at: Optional[str] = None,
        character_name: str = "",
        snapshot_data: Optional[Dict[str, Any]] = None
    ):
        self.id = bookmark_id
        self.chat_name = chat_name
        self.message_index = message_index
        self.name = name or f"Bookmark at message {message_index}"
        self.notes = notes
        self.created_at = created_at or datetime.now().isoformat()
        self.character_name = character_name
        self.snapshot_data = snapshot_data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "chat_name": self.chat_name,
            "message_index": self.message_index,
            "name": self.name,
            "notes": self.notes,
            "created_at": self.created_at,
            "character_name": self.character_name,
            "snapshot_data": self.snapshot_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bookmark":
        """Create from dictionary."""
        return cls(
            bookmark_id=data["id"],
            chat_name=data["chat_name"],
            message_index=data["message_index"],
            name=data.get("name", ""),
            notes=data.get("notes", ""),
            created_at=data.get("created_at"),
            character_name=data.get("character_name", ""),
            snapshot_data=data.get("snapshot_data"),
        )


class BookmarkManager:
    """Manages chat bookmarks and snapshots."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            storage_path = get_inputs_root_dir() / "bookmarks"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.bookmarks_file = self.storage_path / "bookmarks.json"
        self.snapshots_dir = self.storage_path / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._bookmarks: List[Bookmark] = []
        self._load_bookmarks()

        logger.info(f"Bookmark Manager initialized at: {self.storage_path}")

    def _load_bookmarks(self) -> None:
        """Load bookmarks from disk."""
        if self.bookmarks_file.exists():
            try:
                with open(self.bookmarks_file, 'r') as f:
                    data = json.load(f)
                    self._bookmarks = [Bookmark.from_dict(b) for b in data]
                logger.debug(f"Loaded {len(self._bookmarks)} bookmarks")
            except Exception as e:
                logger.error(f"Failed to load bookmarks: {e}")
                self._bookmarks = []
        else:
            self._bookmarks = []

    def _save_bookmarks(self) -> None:
        """Save bookmarks to disk."""
        try:
            data = [b.to_dict() for b in self._bookmarks]
            with open(self.bookmarks_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bookmarks: {e}")

    def create_bookmark(
        self,
        chat_name: str,
        message_index: int,
        name: str = "",
        notes: str = "",
        create_snapshot: bool = False
    ) -> Bookmark:
        """Create a new bookmark.

        Args:
            chat_name: Name of the chat
            message_index: Index of the message to bookmark
            name: Optional name for the bookmark
            notes: Optional notes
            create_snapshot: Whether to create a full snapshot

        Returns:
            The created Bookmark
        """
        # Get character name from chat
        storage = get_chat_storage()
        chat_data = storage.load_chat(chat_name)
        character_name = ""
        snapshot_data = None

        if chat_data:
            messages, metadata = chat_data
            character_name = metadata.character_name or chat_name

            # Optionally create snapshot
            if create_snapshot:
                snapshot_data = {
                    "messages": [m.to_dict() for m in messages[:message_index + 1]],
                    "metadata": metadata.to_dict(),
                }
                # Save snapshot to file
                snapshot_path = self.snapshots_dir / f"{chat_name}_{message_index}.json"
                with open(snapshot_path, 'w') as f:
                    json.dump(snapshot_data, f, indent=2)

        bookmark = Bookmark(
            bookmark_id=str(uuid.uuid4()),
            chat_name=chat_name,
            message_index=message_index,
            name=name,
            notes=notes,
            character_name=character_name,
            snapshot_data=snapshot_data,
        )

        self._bookmarks.append(bookmark)
        self._save_bookmarks()
        logger.info(f"Created bookmark: {bookmark.name} for {chat_name} at message {message_index}")
        return bookmark

    def list_bookmarks(self, chat_name: Optional[str] = None) -> List[Bookmark]:
        """List all bookmarks or bookmarks for a specific chat.

        Args:
            chat_name: Optional chat name filter

        Returns:
            List of Bookmarks
        """
        if chat_name:
            return [b for b in self._bookmarks if b.chat_name == chat_name]
        return self._bookmarks.copy()

    def get_bookmark(self, bookmark_id: str) -> Optional[Bookmark]:
        """Get a specific bookmark by ID.

        Args:
            bookmark_id: The bookmark ID

        Returns:
            The Bookmark or None
        """
        for bookmark in self._bookmarks:
            if bookmark.id == bookmark_id:
                return bookmark
        return None

    def delete_bookmark(self, bookmark_id: str) -> bool:
        """Delete a bookmark.

        Args:
            bookmark_id: The bookmark ID

        Returns:
            True if deleted, False if not found
        """
        for i, bookmark in enumerate(self._bookmarks):
            if bookmark.id == bookmark_id:
                # Delete snapshot if exists
                if bookmark.snapshot_data:
                    snapshot_path = self.snapshots_dir / f"{bookmark.chat_name}_{bookmark.message_index}.json"
                    if snapshot_path.exists():
                        snapshot_path.unlink()

                del self._bookmarks[i]
                self._save_bookmarks()
                logger.info(f"Deleted bookmark: {bookmark_id}")
                return True
        return False

    def restore_bookmark(self, bookmark_id: str) -> Optional[Dict[str, Any]]:
        """Restore a bookmark (returns chat state data).

        Args:
            bookmark_id: The bookmark ID

        Returns:
            Dictionary with chat state info or None
        """
        bookmark = self.get_bookmark(bookmark_id)
        if not bookmark:
            return None

        storage = get_chat_storage()
        chat_data = storage.load_chat(bookmark.chat_name)

        if not chat_data:
            return None

        messages, metadata = chat_data

        # Truncate to bookmark point
        truncated_messages = messages[:bookmark.message_index + 1]

        return {
            "bookmark": bookmark.to_dict(),
            "messages": [m.to_dict() for m in truncated_messages],
            "metadata": metadata.to_dict(),
            "original_message_count": len(messages),
            "restored_message_count": len(truncated_messages),
        }

    def export_bookmark(self, bookmark_id: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Export a bookmark to a file.

        Args:
            bookmark_id: The bookmark ID
            output_path: Optional output path

        Returns:
            Path to exported file or None
        """
        restore_data = self.restore_bookmark(bookmark_id)
        if not restore_data:
            return None

        bookmark = Bookmark.from_dict(restore_data["bookmark"])

        if output_path is None:
            safe_name = bookmark.chat_name.replace("/", "_").replace("\\", "_")
            output_path = Path.cwd() / f"{safe_name}_bookmark_{bookmark.message_index}.jsonl"

        output_path = Path(output_path)

        # Write as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for msg_dict in restore_data["messages"]:
                f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")

        logger.info(f"Exported bookmark to: {output_path}")
        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """Get bookmark statistics."""
        return {
            "total_bookmarks": len(self._bookmarks),
            "chats_with_bookmarks": len(set(b.chat_name for b in self._bookmarks)),
            "snapshots_count": sum(1 for b in self._bookmarks if b.snapshot_data),
            "storage_path": str(self.storage_path),
        }
