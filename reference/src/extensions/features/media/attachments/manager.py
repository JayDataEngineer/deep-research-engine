"""Attachments Manager

Manages attachments functionality.
Refactored to use BaseManager and StorageManager.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager

logger = logging.getLogger(__name__)


@dataclass
class Attachment(BaseItem):
    """An attachment (image, file, etc.)."""

    id: str
    name: str
    url: str = ""
    file_path: str = ""
    type: str = "unknown"  # image, file, link, etc.
    size: int = 0
    content_type: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "file_path": self.file_path,
            "type": self.type,
            "size": self.size,
            "content_type": self.content_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attachment":
        return cls(**data)


class AttachmentManager(BaseManager):
    """Manages attachments functionality."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the attachment manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="attachments", storage_file="attachments.json")

    def _get_item_class(self):
        """Get the Attachment item class."""
        return Attachment

    # ========================================================================
    # Attachment-specific methods
    # ========================================================================

    def get_attachment(self, att_id: str) -> Optional[Attachment]:
        """Get an attachment by ID.

        Args:
            att_id: Attachment ID

        Returns:
            Attachment or None
        """
        return self._items.get(att_id)

    def add_attachment(
        self,
        name: str,
        url: str = "",
        file_path: str = "",
        att_type: str = "unknown",
        size: int = 0,
        content_type: str = ""
    ) -> bool:
        """Add a new attachment.

        Args:
            name: Attachment name
            url: Optional URL
            file_path: Optional local file path
            att_type: Attachment type
            size: File size in bytes
            content_type: MIME type

        Returns:
            True if added
        """
        from src.utils.storage import StorageManager

        att = Attachment(
            id=StorageManager.generate_id_from_name(name),
            name=name,
            url=url,
            file_path=file_path,
            type=att_type,
            size=size,
            content_type=content_type,
        )
        return self.add_item(att)

    def list_attachments(self, att_type: Optional[str] = None) -> List[Attachment]:
        """List all attachments, optionally filtered by type.

        Args:
            att_type: Optional type filter

        Returns:
            List of attachments
        """
        if att_type:
            return [a for a in self._items.values() if a.type == att_type]
        return list(self._items.values())

    def list_attachment_ids(self) -> List[str]:
        """List all attachment IDs.

        Returns:
            List of IDs
        """
        return list(self._items.keys())

    def delete_attachment(self, att_id: str) -> bool:
        """Delete an attachment.

        Args:
            att_id: Attachment ID

        Returns:
            True if deleted
        """
        # Delete file if it exists
        att = self.get_attachment(att_id)
        if att and att.file_path:
            try:
                Path(att.file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete file {att.file_path}: {e}")

        return self.delete_item(att_id)

    def view_attachment(self, att_id: str) -> Optional[str]:
        """Get attachment content/path for viewing.

        Args:
            att_id: Attachment ID

        Returns:
            File path or URL, or None
        """
        att = self.get_attachment(att_id)
        if not att:
            return None

        if att.file_path:
            return att.file_path
        return att.url

    def scrape_attachments(self, url: str) -> List[str]:
        """Scrape attachments from a URL (placeholder).

        Args:
            url: URL to scrape

        Returns:
            List of new attachment IDs
        """
        # This is a placeholder implementation
        # In a real implementation, this would:
        # 1. Fetch the URL
        # 2. Parse HTML for images, files, etc.
        # 3. Download and save attachments
        # 4. Return the attachment IDs

        logger.info(f"Scraping attachments from {url}...")

        # Create a mock attachment
        att_id = f"scrape_{hash(url) % 10000}"
        att = Attachment(
            id=att_id,
            name=f"From {url}",
            url=url,
            type="link",
        )
        self.add_item(att)

        return [att_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get attachment statistics.

        Returns:
            Stats dict
        """
        total_size = sum(a.size for a in self._items.values())
        by_type = {}
        for att in self._items.values():
            by_type[att.type] = by_type.get(att.type, 0) + 1

        return {
            "total_attachments": len(self._items),
            "total_size": total_size,
            "by_type": by_type,
        }