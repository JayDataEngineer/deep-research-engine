"""Expressions & Sprites Manager

Manages character expression images and sprites for visual chat.
Similar to SillyTavern's expressions/sprites feature.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


# Default emotion categories for expressions
DEFAULT_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "fear",
    "disgust",
    "contempt",
    "excited",
    "bored",
    "thinking",
    "confused",
    "loving",
    "annoyed",
    "sleepy",
]


@dataclass
class Sprite:
    """A sprite/expression image for a character."""
    id: str
    character_name: str
    emotion: str
    image_path: str
    name: str = ""
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sprite":
        return cls(**data)


class ExpressionManager:
    """Manages character expressions and sprites."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            storage_path = get_inputs_root_dir() / "expressions"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sprites_dir = self.storage_path / "sprites"
        self.sprites_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"

        self._sprites: Dict[str, Sprite] = {}
        self._load_index()

        logger.info(f"Expression Manager initialized at: {self.storage_path}")

    def _load_index(self) -> None:
        """Load sprite index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    for sprite_data in data:
                        sprite = Sprite.from_dict(sprite_data)
                        self._sprites[sprite.id] = sprite
                logger.debug(f"Loaded {len(self._sprites)} sprites")
            except Exception as e:
                logger.error(f"Failed to load sprite index: {e}")
                self._sprites = {}

    def _save_index(self) -> None:
        """Save sprite index to disk."""
        try:
            data = [s.to_dict() for s in self._sprites.values()]
            with open(self.index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sprite index: {e}")

    def add_sprite(
        self,
        character_name: str,
        image_path: str,
        emotion: str = "neutral",
        name: str = "",
        notes: str = ""
    ) -> Sprite:
        """Add a sprite for a character.

        Args:
            character_name: Name of the character
            image_path: Path to the sprite image
            emotion: Emotion category
            name: Optional name for the sprite
            notes: Optional notes

        Returns:
            The created Sprite
        """
        # Copy image to storage
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Create character directory
        character_dir = self.sprites_dir / character_name
        character_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        sprite_id = str(uuid.uuid4())[:8]
        file_ext = image_path.suffix
        stored_filename = f"{emotion}_{sprite_id}{file_ext}"
        stored_path = character_dir / stored_filename

        # Copy image
        shutil.copy2(image_path, stored_path)

        # Create sprite record
        sprite = Sprite(
            id=sprite_id,
            character_name=character_name,
            emotion=emotion,
            image_path=str(stored_path),
            name=name or f"{emotion.capitalize()} expression",
            notes=notes,
            created_at=datetime.now().isoformat()
        )

        self._sprites[sprite_id] = sprite
        self._save_index()
        logger.info(f"Added sprite for {character_name}: {emotion}")
        return sprite

    def list_sprites(
        self,
        character_name: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> List[Sprite]:
        """List sprites.

        Args:
            character_name: Filter by character
            emotion: Filter by emotion

        Returns:
            List of Sprites
        """
        sprites = list(self._sprites.values())

        if character_name:
            sprites = [s for s in sprites if s.character_name == character_name]
        if emotion:
            sprites = [s for s in sprites if s.emotion == emotion]

        return sprites

    def get_sprite(self, sprite_id: str) -> Optional[Sprite]:
        """Get a sprite by ID.

        Args:
            sprite_id: The sprite ID

        Returns:
            The Sprite or None
        """
        return self._sprites.get(sprite_id)

    def get_sprite_for_emotion(
        self,
        character_name: str,
        emotion: str
    ) -> Optional[Sprite]:
        """Get the best matching sprite for a character and emotion.

        Args:
            character_name: Character name
            emotion: Desired emotion

        Returns:
            The best matching Sprite or None
        """
        # First try exact match
        for sprite in self._sprites.values():
            if sprite.character_name == character_name and sprite.emotion == emotion:
                return sprite

        # Try fuzzy match (e.g., "joyful" -> "happy")
        emotion_lower = emotion.lower()
        for sprite in self._sprites.values():
            if sprite.character_name == character_name:
                if sprite.emotion.lower() in emotion_lower or emotion_lower in sprite.emotion.lower():
                    return sprite

        # Fall back to neutral
        for sprite in self._sprites.values():
            if sprite.character_name == character_name and sprite.emotion == "neutral":
                return sprite

        return None

    def remove_sprite(self, sprite_id: str) -> bool:
        """Remove a sprite.

        Args:
            sprite_id: The sprite ID

        Returns:
            True if removed, False if not found
        """
        sprite = self._sprites.get(sprite_id)
        if not sprite:
            return False

        # Delete image file
        image_path = Path(sprite.image_path)
        if image_path.exists():
            image_path.unlink()

        # Remove from index
        del self._sprites[sprite_id]
        self._save_index()
        logger.info(f"Removed sprite: {sprite_id}")
        return True

    def import_from_folder(
        self,
        character_name: str,
        folder_path: str,
        naming_convention: str = "prefix"
    ) -> List[Sprite]:
        """Import sprites from a folder.

        Args:
            character_name: Character to assign sprites to
            folder_path: Path to folder containing sprite images
            naming_convention: How to determine emotion from filename
                - "prefix": emotion_name.ext (e.g., happy.png)
                - "suffix": name_emotion.ext (e.g., alice_happy.png)

        Returns:
            List of imported Sprites
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        imported = []
        for image_file in folder_path.iterdir():
            if not image_file.is_file():
                continue

            # Check file extension
            if image_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                continue

            # Determine emotion from filename
            stem = image_file.stem
            if naming_convention == "prefix":
                emotion = stem.split("_")[0] if "_" in stem else stem
            else:  # suffix
                parts = stem.split("_")
                emotion = parts[-1] if len(parts) > 1 else stem

            # Normalize emotion
            emotion = emotion.lower()

            # Import sprite
            try:
                sprite = self.add_sprite(
                    character_name=character_name,
                    image_path=str(image_file),
                    emotion=emotion,
                    name=f"{emotion.capitalize()} (imported)"
                )
                imported.append(sprite)
            except Exception as e:
                logger.error(f"Failed to import {image_file.name}: {e}")

        return imported

    def get_emotion_list(self) -> List[str]:
        """Get list of available emotions for a character."""
        return DEFAULT_EMOTIONS

    def get_stats(self) -> Dict[str, Any]:
        """Get expression statistics."""
        sprites = list(self._sprites.values())
        characters = set(s.character_name for s in sprites)
        emotions = set(s.emotion for s in sprites)

        return {
            "total_sprites": len(sprites),
            "characters": len(characters),
            "emotions_used": list(emotions),
            "storage_path": str(self.storage_path),
        }
