"""Voice Mapper - Maps character cards to TTS voices.

Handles character-to-voice associations, allowing character cards
to have default voices for roleplay mode.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceMapper:
    """Maps character cards to TTS voices.

    This class provides:
    - Character voice lookup by character name
    - Voice association storage in character card metadata
    - Default voice fallback
    """

    def __init__(self, default_voice_id: Optional[str] = None):
        """Initialize the voice mapper.

        Args:
            default_voice_id: Default voice to use when no character voice is set
        """
        self.default_voice_id = default_voice_id
        self._character_storage = None
        logger.info("VoiceMapper initialized")

    async def initialize(self, context):
        """Initialize the voice mapper with application context.

        Args:
            context: Application context
        """
        try:
            # Try to import character storage
            from src.extensions.libraries.character_engine.crud import get_character_storage
            self._character_storage = get_character_storage
            logger.info("VoiceMapper: Character storage loaded")
        except ImportError as e:
            logger.warning(f"VoiceMapper: Character storage not available: {e}")
            self._character_storage = None

    async def get_character_voice(
        self,
        character_name: str,
    ) -> Optional[str]:
        """Get the voice ID associated with a character.

        Args:
            character_name: Name of the character

        Returns:
            Voice ID if found, None otherwise
        """
        if not self._character_storage:
            logger.warning("VoiceMapper: Character storage not available")
            return None

        try:
            storage = self._character_storage()
            character = storage.load_character(character_name)

            if not character:
                logger.warning(f"VoiceMapper: Character '{character_name}' not found")
                return None

            # Check if character has a voice_id in metadata
            voice_id = character.get("voice_id")
            if voice_id:
                logger.info(f"VoiceMapper: Found voice '{voice_id}' for character '{character_name}'")
                return voice_id

            logger.info(f"VoiceMapper: No voice set for character '{character_name}'")
            return None

        except Exception as e:
            logger.error(f"VoiceMapper: Error loading character voice: {e}")
            return None

    async def get_character_voice_config(
        self,
        character_name: str,
    ) -> Optional[dict]:
        """Get the full voice configuration for a character.

        Args:
            character_name: Name of the character

        Returns:
            Dict with voice configuration (voice_id, extension, speed, pitch) or None
        """
        if not self._character_storage:
            return None

        try:
            storage = self._character_storage()
            character = storage.load_character(character_name)

            if not character:
                return None

            # Build voice config from character metadata
            config = {
                "voice_id": character.get("voice_id"),
                "extension": character.get("voice_extension", "kokoro"),
                "speed": character.get("voice_speed", 1.0),
                "pitch": character.get("voice_pitch", 1.0),
            }

            # Only return config if voice_id is set
            if config["voice_id"]:
                logger.info(f"VoiceMapper: Voice config for '{character_name}': {config}")
                return config

            return None

        except Exception as e:
            logger.error(f"VoiceMapper: Error loading character voice config: {e}")
            return None

    async def set_character_voice(
        self,
        character_name: str,
        voice_id: str,
        extension: str = "kokoro",
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> bool:
        """Associate a voice with a character.

        Args:
            character_name: Name of the character
            voice_id: Voice ID to use
            extension: TTS extension name
            speed: Speech speed multiplier
            pitch: Pitch adjustment

        Returns:
            True if successful, False otherwise
        """
        if not self._character_storage:
            logger.error("VoiceMapper: Character storage not available")
            return False

        try:
            storage = self._character_storage()
            character = storage.load_character(character_name)

            if not character:
                logger.error(f"VoiceMapper: Character '{character_name}' not found")
                return False

            # Update character with voice settings
            character["voice_id"] = voice_id
            character["voice_extension"] = extension
            character["voice_speed"] = speed
            character["voice_pitch"] = pitch

            # Save character
            storage.save_character(character, format="json")

            logger.info(f"VoiceMapper: Set voice '{voice_id}' for character '{character_name}'")
            return True

        except Exception as e:
            logger.error(f"VoiceMapper: Error setting character voice: {e}")
            return False

    async def resolve_voice(
        self,
        character_name: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve the appropriate voice ID to use.

        Priority:
        1. Explicitly provided voice_id
        2. Character's associated voice
        3. Default voice

        Args:
            character_name: Optional character name
            voice_id: Optional explicit voice ID

        Returns:
            Resolved voice ID or None
        """
        # Explicit voice_id takes precedence
        if voice_id:
            logger.info(f"VoiceMapper: Using explicit voice '{voice_id}'")
            return voice_id

        # Try character voice
        if character_name:
            character_voice = await self.get_character_voice(character_name)
            if character_voice:
                return character_voice

        # Fall back to default
        logger.info(f"VoiceMapper: Using default voice '{self.default_voice_id}'")
        return self.default_voice_id

    async def close(self):
        """Cleanup resources."""
        self._character_storage = None
        logger.info("VoiceMapper closed")
