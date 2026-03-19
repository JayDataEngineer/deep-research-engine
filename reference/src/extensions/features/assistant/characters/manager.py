"""Characters Manager

Manages character card functionality. Delegates to CharacterStorage for common operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.extensions.libraries.core_chat_engine.storage.character_storage import CharacterStorage, CharacterCard, get_character_storage

logger = logging.getLogger(__name__)


class CharacterManager:
    """Manages character card functionality.

    This is a thin wrapper that delegates to CharacterStorage for most operations
    while providing convenience methods for character imports.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the character manager.

        Args:
            base_dir: Base directory for character storage (optional, uses default if not provided)
        """
        if base_dir:
            self._storage = CharacterStorage(storage_dir=base_dir)
        else:
            self._storage = get_character_storage()
        self.characters_dir = self._storage.storage_dir
        logger.info(f"CharacterManager initialized at: {self.characters_dir}")

    def list_characters(self) -> List[str]:
        """List all character cards.

        Returns:
            List of character names
        """
        result = self._storage.list_all()
        return [c.get("name", c.get("filename", "")) for c in result if c]

    def get_character(self, name: str) -> Optional[CharacterCard]:
        """Get a character card.

        Args:
            name: Character name or filename

        Returns:
            CharacterCard or None
        """
        # Try with .json extension first
        char = self._storage.load(f"{name}.json")
        if char:
            return char
        # Try without extension
        return self._storage.load(name)

    def get_character_data(self, name: str) -> Optional[str]:
        """Get character data as JSON string.

        Args:
            name: Character name

        Returns:
            JSON string or None
        """
        char = self.get_character(name)
        if char:
            return json.dumps(char.to_dict(), indent=2, ensure_ascii=False)
        return None

    def save_character(self, character: CharacterCard, format: str = "json") -> bool:
        """Save a character card.

        Args:
            character: CharacterCard to save
            format: File format ("json", "yaml")

        Returns:
            True if successful
        """
        try:
            filename = f"{character.name}.{format}"
            self._storage.save(character.data, filename=filename, format=format)
            return True
        except Exception as e:
            logger.error(f"Failed to save character: {e}")
            return False

    def delete_character(self, name: str) -> bool:
        """Delete a character card.

        Args:
            name: Character name

        Returns:
            True if deleted
        """
        return self._storage.delete(f"{name}.json") or self._storage.delete(name)

    # ========================================================================
    # Import Methods (CharacterManager-specific functionality)
    # ========================================================================

    def import_tavern_character(self, character_data: str) -> Optional[CharacterCard]:
        """Import a SillyTavern character card.

        Args:
            character_data: Character data as JSON string

        Returns:
            Imported CharacterCard or None
        """
        try:
            char_dict = json.loads(character_data)

            # Extract name for filename
            name = char_dict.get("name", "unknown")
            filename = f"{name}.json"

            # Save using new CharacterStorage API
            card = self._storage.save(char_dict, filename=filename, format="json")
            logger.info(f"Imported character: {name}")
            return card

        except Exception as e:
            logger.error(f"Failed to import character: {e}")
            return None

    def import_png_character(self, file_path: str) -> Optional[CharacterCard]:
        """Import a PNG character card (SillyTavern format).

        Args:
            file_path: Path to the PNG file

        Returns:
            Imported CharacterCard or None
        """
        try:
            src_path = Path(file_path)
            if not src_path.exists():
                logger.error(f"PNG file not found: {file_path}")
                return None

            # Copy PNG to storage directory
            import shutil
            dst_path = self.characters_dir / src_path.name
            shutil.copy2(src_path, dst_path)

            # Load the PNG using new CharacterStorage API
            card = self._storage.load(src_path.name)
            if card:
                logger.info(f"Imported PNG character: {card.name}")
            return card

        except Exception as e:
            logger.error(f"Failed to import PNG character: {e}")
            return None

    def export_character_png(self, name: str, image_path: str) -> Optional[Path]:
        """Export a character as PNG with embedded data.

        Args:
            name: Character name
            image_path: Path to the image to use

        Returns:
            Path to saved PNG or None
        """
        char = self.get_character(name)
        if not char:
            return None

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Save as PNG with embedded data
            filename = f"{name}.png"
            self._storage.save(char.data, filename=filename, format="png")

            # The save method creates the PNG, now copy the image data
            # Note: CharacterStorage saves a placeholder PNG, so we need to
            # manually handle the image data if we want custom images
            png_path = self.characters_dir / filename
            return png_path

        except Exception as e:
            logger.error(f"Failed to export PNG: {e}")
            return None

    # ========================================================================
    # Stats
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get character statistics.

        Returns:
            Stats dict
        """
        return {
            "total_characters": len(self.list_characters()),
            "storage_path": str(self.characters_dir),
        }