"""Character Card CRUD Operations.

Create, Read, Update, Delete operations for character cards and books.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from PIL import Image
    from PIL import PngImagePlugin
    from io import BytesIO
    import base64
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from .models import CharacterCard, CharacterBook
from .parser import parse_character_card, parse_character_book

logger = logging.getLogger(__name__)


class CharacterStorage:
    """Storage backend for character cards and books."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize character storage.

        Args:
            base_dir: Base directory for storage. Defaults to inputs/characters
        """
        if base_dir is None:
            # Try to get from helpers
            try:
                from ....api.shared.helpers import get_inputs_root_dir
                base_dir = get_inputs_root_dir() / "characters"
            except (ImportError, AttributeError):
                # Fallback: go up to project root (backend/ -> project root)
                base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "data" / "inputs" / "characters"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.cards_dir = self.base_dir / "cards"
        self.books_dir = self.base_dir / "books"

        for dir_path in [self.cards_dir, self.books_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Character storage initialized at: {self.base_dir}")

    # ========================================================================
    # Character Card CRUD
    # ========================================================================

    def save_card(
        self,
        card: CharacterCard,
        format: str = "json",
        image_data: Optional[bytes] = None
    ) -> Path:
        """Save a character card.

        Args:
            card: Character card to save
            format: File format ("json", "yaml", "png")
            image_data: Optional image bytes for PNG format

        Returns:
            Path to saved file
        """
        safe_name = self._sanitize_filename(card.name)

        if format == "png" and image_data and _HAS_PIL:
            filepath = self.cards_dir / f"{safe_name}.png"
            self._save_png_card(card, image_data, filepath)
        else:
            filepath = self.cards_dir / f"{safe_name}.{format}"
            self._save_text_card(card, filepath, format)

        logger.info(f"Saved character card: {filepath}")
        return filepath

    def load_card(self, name: str, format: str = None) -> Optional[CharacterCard]:
        """Load a character card.

        Args:
            name: Character name or filename
            format: File format or None to auto-detect

        Returns:
            CharacterCard or None
        """
        safe_name = self._sanitize_filename(name)

        if format is None:
            # Auto-detect
            for filepath in self.cards_dir.glob(f"{safe_name}.*"):
                if filepath.suffix in [".json", ".yaml", ".yml"]:
                    return parse_character_card(filepath)
                elif filepath.suffix == ".png" and _HAS_PIL:
                    result = parse_character_card(filepath)
                    if result:
                        return result
            return None
        else:
            filepath = self.cards_dir / f"{safe_name}.{format}"
            if filepath.exists():
                return parse_character_card(filepath, format)
            return None

    def list_cards(self) -> List[str]:
        """List all character card names."""
        names = set()
        for filepath in self.cards_dir.iterdir():
            if filepath.is_file() and filepath.suffix in [".json", ".yaml", ".yml", ".png"]:
                names.add(filepath.stem)
        return sorted(names)

    def delete_card(self, name: str) -> bool:
        """Delete a character card.

        Args:
            name: Character name

        Returns:
            True if deleted, False if not found
        """
        safe_name = self._sanitize_filename(name)
        deleted = False

        for ext in ["json", "yaml", "yml", "png"]:
            filepath = self.cards_dir / f"{safe_name}.{ext}"
            if filepath.exists():
                filepath.unlink()
                deleted = True
                logger.info(f"Deleted character card: {filepath}")

        return deleted

    def update_card(
        self,
        name: str,
        updates: dict,
        format: str = "json"
    ) -> Optional[CharacterCard]:
        """Update a character card.

        Args:
            name: Current character name
            updates: Dict of fields to update
            format: File format for saving

        Returns:
            Updated CharacterCard or None
        """
        card = self.load_card(name, format)
        if not card:
            return None

        # Delete old file if name changed
        old_name = card.name
        for field, value in updates.items():
            if hasattr(card, field):
                setattr(card, field, value)

        # Save with new name
        self.save_card(card, format)

        # Delete old file if renamed
        if "name" in updates and updates["name"] != old_name:
            self.delete_card(old_name)

        return card

    # ========================================================================
    # Character Book CRUD
    # ========================================================================

    def save_book(self, book: CharacterBook, format: str = "json") -> Path:
        """Save a character book.

        Args:
            book: Character book to save
            format: File format ("json", "yaml")

        Returns:
            Path to saved file
        """
        safe_name = self._sanitize_filename(book.name)
        filepath = self.books_dir / f"{safe_name}.{format}"

        data = {
            "name": book.name,
            "description": book.description,
            "scan_depth": book.scan_depth,
            "token_budget": book.token_budget,
            "extensions": book.extensions,
            "entries": [e.model_dump() for e in book.entries],
        }

        if format == "yaml" and _HAS_YAML:
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved character book: {filepath}")
        return filepath

    def load_book(self, name: str, format: str = None) -> Optional[CharacterBook]:
        """Load a character book.

        Args:
            name: Book name
            format: File format or None to auto-detect

        Returns:
            CharacterBook or None
        """
        safe_name = self._sanitize_filename(name)

        if format is None:
            # Auto-detect
            for ext in ["json", "yaml", "yml"]:
                filepath = self.books_dir / f"{safe_name}.{ext}"
                if filepath.exists():
                    return parse_character_book(filepath, ext)
            return None
        else:
            filepath = self.books_dir / f"{safe_name}.{format}"
            if filepath.exists():
                return parse_character_book(filepath, format)
            return None

    def list_books(self) -> List[str]:
        """List all character book names."""
        names = set()
        for filepath in self.books_dir.iterdir():
            if filepath.is_file() and filepath.suffix in [".json", ".yaml", ".yml"]:
                names.add(filepath.stem)
        return sorted(names)

    def delete_book(self, name: str) -> bool:
        """Delete a character book.

        Args:
            name: Book name

        Returns:
            True if deleted, False if not found
        """
        safe_name = self._sanitize_filename(name)
        deleted = False

        for ext in ["json", "yaml", "yml"]:
            filepath = self.books_dir / f"{safe_name}.{ext}"
            if filepath.exists():
                filepath.unlink()
                deleted = True
                logger.info(f"Deleted character book: {filepath}")

        return deleted

    # ========================================================================
    # PNG Helper Methods
    # ========================================================================

    def _save_png_card(self, card: CharacterCard, image_data: bytes, filepath: Path):
        """Save character card embedded in PNG."""
        # Prepare character data
        card_dict = card.model_dump()
        card_json = json.dumps(card_dict, ensure_ascii=False)
        card_bytes = card_json.encode("utf-8")
        b64_data = base64.b64encode(card_bytes).decode("ascii")

        # Split into chunks (PNG chunks limited to 4KB)
        chunks = [b64_data[i:i+4000] for i in range(0, len(b64_data), 4000)]

        # Load image
        img = Image.open(BytesIO(image_data))

        # Add metadata
        meta = PngImagePlugin.PngInfo()
        for i, chunk in enumerate(chunks):
            meta.add_text(f"chara{i}", chunk)

        # Save
        img.save(filepath, pnginfo=meta)

    def _save_text_card(self, card: CharacterCard, filepath: Path, format: str):
        """Save character card as JSON or YAML."""
        data = card.model_dump()

        if format == "yaml" and _HAS_YAML:
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    # ========================================================================
    # Utilities
    # ========================================================================

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a name for safe filename usage."""
        unsafe = '<>:"/\\|?*'
        for char in unsafe:
            name = name.replace(char, "_")
        return name.strip()

    def get_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dict with counts and directories
        """
        def count_files(directory: Path) -> int:
            return sum(1 for _ in directory.iterdir() if _.is_file())

        return {
            "cards": {
                "count": count_files(self.cards_dir),
                "directory": str(self.cards_dir),
            },
            "books": {
                "count": count_files(self.books_dir),
                "directory": str(self.books_dir),
            },
        }


# Singleton instance
_character_storage: Optional[CharacterStorage] = None


def get_character_storage(base_dir: Optional[Path] = None) -> CharacterStorage:
    """Get the singleton character storage.

    Args:
        base_dir: Optional custom base directory

    Returns:
        CharacterStorage instance
    """
    global _character_storage
    if _character_storage is None or base_dir is not None:
        _character_storage = CharacterStorage(base_dir=base_dir)
    return _character_storage
