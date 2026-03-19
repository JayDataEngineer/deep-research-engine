"""Character Card Storage Abstraction.

Provides a storage interface for character cards, abstracting away
direct file system operations. This allows extensions to store
and retrieve characters without tight coupling to filesystem.

Features:
- CRUD operations for character cards
- Support for JSON, PNG, and YAML formats
- In-memory session caching
- Thread-safe operations
"""

import base64
import json
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image, PngImagePlugin

from src.config import settings

logger = logging.getLogger(__name__)


class CharacterCard:
    """Character card data model."""

    def __init__(
        self,
        name: str,
        data: Dict[str, Any],
        format: str = "json",
        filename: Optional[str] = None,
    ):
        self.name = name
        self.data = data
        self.format = format
        self.filename = filename

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data": self.data,
            "format": self.format,
            "filename": self.filename,
        }


class CharacterStorage:
    """Storage abstraction for character cards.

    Provides a clean interface for saving/loading character cards
    without direct filesystem access in route handlers.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize character storage.

        Args:
            storage_dir: Base directory for character storage.
                        Defaults to settings.upload_dir / "characters"
        """
        if storage_dir is None:
            storage_dir = Path(settings.upload_dir) / "characters"
        self.storage_dir = Path(storage_dir)
        self._ensure_directory()

        # In-memory cache for session storage
        self._session_cache: Dict[str, CharacterCard] = {}

    def _ensure_directory(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # CRUD Operations
    # =============================================================================

    def save(
        self,
        character: Dict[str, Any],
        filename: str,
        format: str = "json",
        session_id: Optional[str] = None,
    ) -> CharacterCard:
        """Save a character card to storage.

        Args:
            character: Character data dictionary
            filename: Name for the saved file
            format: File format (json, png, yaml)
            session_id: Optional session ID for caching

        Returns:
            CharacterCard instance

        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        """
        self._ensure_directory()

        name = character.get("name", "Unknown")
        file_path = self.storage_dir / filename

        if format == "json":
            self._save_json(character, file_path)
        elif format == "png":
            self._save_png(character, file_path)
        elif format in ("yaml", "yml"):
            self._save_yaml(character, file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        card = CharacterCard(name=name, data=character, format=format, filename=filename)

        # Cache to session if provided
        if session_id:
            self.cache_session(session_id, card)

        logger.info(f"Saved character '{name}' to {file_path}")
        return card

    def load(self, filename: str) -> Optional[CharacterCard]:
        """Load a character card from storage.

        Args:
            filename: Name of the file to load

        Returns:
            CharacterCard instance or None if not found

        Raises:
            IOError: If file cannot be read
        """
        file_path = self.storage_dir / filename

        if not file_path.exists():
            return None

        ext = file_path.suffix.lower()

        try:
            if ext == ".json":
                data = self._load_json(file_path)
                format = "json"
            elif ext == ".png":
                data = self._load_png(file_path)
                format = "png"
            elif ext in (".yaml", ".yml"):
                data = self._load_yaml(file_path)
                format = "yaml"
            else:
                logger.warning(f"Unsupported file format: {ext}")
                return None

            name = data.get("name", file_path.stem)
            return CharacterCard(name=name, data=data, format=format, filename=filename)

        except Exception as e:
            logger.error(f"Failed to load character from {file_path}: {e}")
            return None

    def delete(self, filename: str) -> bool:
        """Delete a character card from storage.

        Args:
            filename: Name of the file to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_dir / filename

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            logger.info(f"Deleted character: {filename}")

            # Also delete associated files (JSON backup for PNG, etc.)
            if file_path.suffix == ".png":
                json_path = file_path.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")
            return False

    def list_all(self) -> List[Dict[str, Any]]:
        """List all character cards in storage.

        Returns:
            List of character metadata dictionaries
        """
        characters = []

        for path in self.storage_dir.iterdir():
            if not path.is_file():
                continue

            try:
                char_data = self._read_metadata(path)
                characters.append(char_data)
            except Exception as e:
                logger.warning(f"Error reading {path.name}: {e}")

        # Sort by name
        characters.sort(key=lambda x: x.get("name", "").lower())
        return characters

    def exists(self, filename: str) -> bool:
        """Check if a character file exists.

        Args:
            filename: Name of the file to check

        Returns:
            True if file exists, False otherwise
        """
        return (self.storage_dir / filename).exists()

    # =============================================================================
    # Session Caching
    # =============================================================================

    def cache_session(self, session_id: str, character: CharacterCard) -> None:
        """Cache a character for a session.

        Args:
            session_id: Session identifier
            character: CharacterCard to cache
        """
        self._session_cache[session_id] = character

    def get_session(self, session_id: str) -> Optional[CharacterCard]:
        """Get cached character for a session.

        Args:
            session_id: Session identifier

        Returns:
            CharacterCard or None if not cached
        """
        return self._session_cache.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear cached character for a session.

        Args:
            session_id: Session identifier
        """
        self._session_cache.pop(session_id, None)

    # =============================================================================
    # Private File Format Handlers
    # =============================================================================

    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save character as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load character from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_yaml(self, data: Dict[str, Any], path: Path) -> None:
        """Save character as YAML."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load character from YAML."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _save_png(self, data: Dict[str, Any], path: Path) -> None:
        """Save character as PNG with embedded data."""
        # Create a simple placeholder image
        img_size = (400, 600)
        img = Image.new("RGB", img_size, color=(80, 80, 100))

        # Serialize character data
        char_data_json = json.dumps(data, ensure_ascii=False)

        # Create PNG info with embedded character data
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("chara", char_data_json)

        # Save the PNG with embedded data
        img.save(path, "PNG", pnginfo=pnginfo)

        # Also save as JSON backup
        json_path = path.with_suffix(".json")
        self._save_json(data, json_path)

    def _load_png(self, path: Path) -> Dict[str, Any]:
        """Load character from PNG with embedded data."""
        with Image.open(path) as img:
            # Check for embedded character data
            if "chara" in img.text:
                return json.loads(img.text["chara"])
            elif "li" in img.text:
                return json.loads(img.text["li"])
            elif "chara-v2" in img.text:
                return json.loads(img.text["chara-v2"])

        # Fallback to JSON backup
        json_path = path.with_suffix(".json")
        if json_path.exists():
            return self._load_json(json_path)

        raise ValueError(f"No character data found in {path}")

    def _read_metadata(self, path: Path) -> Dict[str, Any]:
        """Read metadata from a character file.

        Args:
            path: Path to character file

        Returns:
            Metadata dictionary
        """
        try:
            char_data = None

            if path.suffix == ".json":
                char_data = self._load_json(path)
                if "data" in char_data:
                    char_data = char_data["data"]
                elif "spec" in char_data and char_data["spec"] == "chara_card_v2":
                    char_data = char_data.get("data", char_data)
            elif path.suffix in (".png", ".webp"):
                with Image.open(path) as img:
                    if "chara" in img.text:
                        char_data_full = json.loads(img.text["chara"])
                        if "spec" in char_data_full and char_data_full["spec"] == "chara_card_v2":
                            char_data = char_data_full.get("data", char_data_full)
                        elif "data" in char_data_full:
                            char_data = char_data_full["data"]
                        else:
                            char_data = char_data_full

            if not char_data:
                char_data = {}

            name = char_data.get("name", path.stem)
            description = char_data.get("description", "")[:200] + "..." if char_data.get("description") else ""
            creator = char_data.get("creator", "")
            tags = char_data.get("tags", [])

            # Avatar URL: use the file itself if it's an image
            avatar_url = f"/api/v1/roleplay/character/avatar/{path.name}" if path.suffix in (".png", ".webp", ".jpg", ".jpeg", ".gif") else None

            return {
                "filename": path.name,
                "name": name,
                "description": description,
                "creator": creator,
                "tags": tags,
                "size": path.stat().st_size,
                "type": path.suffix,
                "avatar_url": avatar_url,
            }

        except Exception as e:
            return {
                "filename": path.name,
                "name": path.stem,
                "description": "[Error reading character]",
                "creator": "",
                "tags": [],
                "size": path.stat().st_size,
                "type": path.suffix,
                "avatar_url": None,
            }

    # =============================================================================
    # Avatar Image Access
    # =============================================================================

    def get_avatar(self, filename: str) -> Optional[tuple[bytes, str]]:
        """Get avatar image data and media type.

        Args:
            filename: Name of the image file

        Returns:
            Tuple of (image_bytes, media_type) or None if not found
        """
        file_path = self.storage_dir / filename

        if not file_path.exists():
            return None

        # Determine media type
        media_type_map = {
            ".png": "image/png",
            ".webp": "image/webp",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }
        media_type = media_type_map.get(file_path.suffix.lower(), "image/png")

        try:
            with open(file_path, "rb") as f:
                image_data = f.read()
            return (image_data, media_type)
        except Exception as e:
            logger.error(f"Failed to read avatar {filename}: {e}")
            return None


# =============================================================================
# Global Storage Instance
# =============================================================================

_default_storage: Optional[CharacterStorage] = None


def get_character_storage() -> CharacterStorage:
    """Get the global character storage instance.

    Returns:
        CharacterStorage instance
    """
    global _default_storage
    if _default_storage is None:
        _default_storage = CharacterStorage()
    return _default_storage
