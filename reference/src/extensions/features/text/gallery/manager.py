"""Gallery Manager

Manages image gallery for chat attachments and generated images.
Similar to SillyTavern's gallery feature.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class GalleryItem:
    """An item in the gallery (image)."""
    id: str
    filename: str
    path: str
    chat_id: str = ""
    source: str = "upload"  # upload, generated, attachment
    caption: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v or v == []}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GalleryItem":
        return cls(**data)


class GalleryManager:
    """Manages image gallery."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            storage_path = get_inputs_root_dir() / "gallery"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.storage_path / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"

        self._items: Dict[str, GalleryItem] = {}
        self._load_index()

        logger.info(f"Gallery Manager initialized at: {self.storage_path}")

    def _load_index(self) -> None:
        """Load gallery index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    for item_data in data:
                        item = GalleryItem.from_dict(item_data)
                        self._items[item.id] = item
                logger.debug(f"Loaded {len(self._items)} gallery items")
            except Exception as e:
                logger.error(f"Failed to load gallery index: {e}")
                self._items = {}

    def _save_index(self) -> None:
        """Save gallery index to disk."""
        try:
            data = [item.to_dict() for item in self._items.values()]
            with open(self.index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save gallery index: {e}")

    def add_image(
        self,
        image_path: str,
        chat_id: str = "",
        source: str = "upload",
        caption: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GalleryItem:
        """Add an image to the gallery.

        Args:
            image_path: Path to the image file
            chat_id: Optional associated chat ID
            source: Source of the image (upload, generated, attachment)
            caption: Optional caption
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            The created GalleryItem
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate unique ID
        item_id = str(uuid.uuid4())[:8]
        file_ext = image_path.suffix
        stored_filename = f"{item_id}_{image_path.stem}{file_ext}"
        stored_path = self.images_dir / stored_filename

        # Copy image to gallery
        shutil.copy2(image_path, stored_path)

        # Create gallery item
        item = GalleryItem(
            id=item_id,
            filename=stored_filename,
            path=str(stored_path),
            chat_id=chat_id,
            source=source,
            caption=caption,
            tags=tags or [],
            metadata=metadata or {}
        )

        self._items[item_id] = item
        self._save_index()
        logger.info(f"Added image to gallery: {item.filename}")
        return item

    def list_items(
        self,
        chat_id: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[GalleryItem]:
        """List gallery items.

        Args:
            chat_id: Filter by chat ID
            source: Filter by source
            tags: Filter by tags (any match)
            limit: Max items to return

        Returns:
            List of GalleryItems
        """
        items = list(self._items.values())

        if chat_id:
            items = [i for i in items if i.chat_id == chat_id]
        if source:
            items = [i for i in items if i.source == source]
        if tags:
            items = [i for i in items if any(tag in i.tags for tag in tags)]

        # Sort by created_at (newest first)
        items.sort(key=lambda x: x.created_at, reverse=True)

        return items[:limit]

    def get_item(self, item_id: str) -> Optional[GalleryItem]:
        """Get a gallery item by ID.

        Args:
            item_id: The item ID

        Returns:
            The GalleryItem or None
        """
        return self._items.get(item_id)

    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the gallery.

        Args:
            item_id: The item ID

        Returns:
            True if removed, False if not found
        """
        item = self.get_item(item_id)
        if not item:
            return False

        # Delete image file
        image_path = Path(item.path)
        if image_path.exists():
            image_path.unlink()

        # Remove from index
        del self._items[item_id]
        self._save_index()
        logger.info(f"Removed gallery item: {item_id}")
        return True

    def update_caption(self, item_id: str, caption: str) -> Optional[GalleryItem]:
        """Update the caption for a gallery item.

        Args:
            item_id: The item ID
            caption: New caption

        Returns:
            The updated GalleryItem or None
        """
        item = self.get_item(item_id)
        if not item:
            return None

        item.caption = caption
        self._save_index()
        logger.info(f"Updated caption for: {item_id}")
        return item

    def add_tags(self, item_id: str, tags: List[str]) -> Optional[GalleryItem]:
        """Add tags to a gallery item.

        Args:
            item_id: The item ID
            tags: Tags to add

        Returns:
            The updated GalleryItem or None
        """
        item = self.get_item(item_id)
        if not item:
            return None

        for tag in tags:
            if tag not in item.tags:
                item.tags.append(tag)

        self._save_index()
        logger.info(f"Added tags to: {item_id}")
        return item

    def export_gallery(
        self,
        output_path: Optional[Path] = None,
        include_images: bool = True
    ) -> Optional[Path]:
        """Export the gallery to a zip file.

        Args:
            output_path: Optional output path
            include_images: Whether to include image files

        Returns:
            Path to exported file or None
        """
        if output_path is None:
            output_path = Path.cwd() / f"gallery_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        output_path = Path(output_path)

        try:
            import zipfile

            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add index
                index_data = [item.to_dict() for item in self._items.values()]
                zipf.writestr("index.json", json.dumps(index_data, indent=2))

                # Add images if requested
                if include_images:
                    for item in self._items.values():
                        image_path = Path(item.path)
                        if image_path.exists():
                            zipf.write(image_path, f"images/{item.filename}")

            logger.info(f"Exported gallery to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export gallery: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get gallery statistics."""
        items = list(self._items.values())

        # Count by source
        by_source = {}
        for item in items:
            by_source[item.source] = by_source.get(item.source, 0) + 1

        # Count by chat
        chats_with_images = len(set(i.chat_id for i in items if i.chat_id))

        return {
            "total_images": len(items),
            "by_source": by_source,
            "chats_with_images": chats_with_images,
            "storage_path": str(self.storage_path),
        }
