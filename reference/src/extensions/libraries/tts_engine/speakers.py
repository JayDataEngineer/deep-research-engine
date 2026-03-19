"""
Speaker Management Service for IndexTTS2 External API.

Manages speaker-to-reference-audio mappings with JSON persistence.
Supports both file upload and manual file path configuration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SpeakerMapping(BaseModel):
    """A speaker mapping with metadata."""

    speaker_id: str = Field(..., description="Unique identifier for the speaker")
    name: str = Field(..., description="Display name for the speaker")
    audio_path: str = Field(..., description="Path to reference audio file")
    description: str | None = Field(None, description="Optional description")
    gender: str | None = Field(None, description="Speaker gender (male/female/neutral)")
    language: str | None = Field(None, description="Speaker language")
    is_active: bool = Field(True, description="Whether this speaker is active")

    @validator('audio_path')
    def validate_audio_path(cls, v: str, values: dict) -> str:
        """Validate that the audio path exists and is a valid audio file."""
        path = Path(v)
        if not path.exists():
            # Don't fail validation for new speakers, just log a warning
            logger.warning(f"Audio path does not exist: {v}")
        elif path.suffix.lower() not in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            raise ValueError(f"Invalid audio file format: {path.suffix}")
        return v

    @validator('speaker_id')
    def validate_speaker_id(cls, v: str) -> str:
        """Validate speaker ID format."""
        if not v or not v.strip():
            raise ValueError("Speaker ID cannot be empty")
        # Only allow alphanumeric, underscore, and hyphen
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Speaker ID can only contain letters, numbers, underscore, and hyphen")
        return v.strip()


class SpeakerManager:
    """Manages speaker-to-reference-audio mappings with JSON persistence."""

    def __init__(self, storage_dir: str | Path | None = None):
        """Initialize the speaker manager.

        Args:
            storage_dir: Directory to store speaker_mappings.json. Defaults to data/config/ (root level).
        """
        if storage_dir is None:
            # Default to data/config/ for the config file
            from src.constants import DATA_CONFIG_DIR, DATA_INPUTS_DIR
            storage_dir = DATA_CONFIG_DIR
            audio_storage_dir = DATA_INPUTS_DIR / "audio" / "reference"
        else:
            audio_storage_dir = Path(storage_dir) / "reference_audio"

        self.storage_dir = Path(storage_dir)
        self.mappings_file = self.storage_dir / "speaker_mappings.json"
        self.audio_storage_dir = Path(audio_storage_dir)

        # Ensure directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.audio_storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing mappings
        self._mappings: dict[str, dict[str, Any]] = {}
        self._load_mappings()

    def _load_mappings(self) -> None:
        """Load speaker mappings from JSON file."""
        if self.mappings_file.exists():
            try:
                with open(self.mappings_file) as f:
                    data = json.load(f)
                    # Validate each mapping
                    for speaker_id, mapping_data in data.items():
                        try:
                            SpeakerMapping(**mapping_data)
                            self._mappings[speaker_id] = mapping_data
                        except Exception as e:
                            logger.warning(f"Invalid mapping for speaker '{speaker_id}': {e}")
                logger.info(f"Loaded {len(self._mappings)} speaker mappings from {self.mappings_file}")
            except Exception as e:
                logger.error(f"Failed to load speaker mappings: {e}")
                self._mappings = {}
        else:
            logger.info(f"No speaker mappings file found at {self.mappings_file}, starting fresh")
            self._mappings = {}
            self._save_mappings()

        # Initialize default speakers if no speakers exist
        self._ensure_default_speakers()

    def _ensure_default_speakers(self) -> None:
        """Ensure default speakers are registered."""
        # Check if we have any speakers
        if self._mappings:
            return

        from src.constants import ROOT_DATA_DIR

        # Register default Asuna speaker
        asuna_audio_path = ROOT_DATA_DIR / "inputs" / "audio_samples" / "asuna_5s.wav"
        if asuna_audio_path.exists():
            try:
                self.add_speaker(
                    speaker_id="asuna",
                    name="Asuna",
                    audio_path=str(asuna_audio_path),
                    description="Default female voice - energetic and friendly",
                    gender="female",
                )
                logger.info("Registered default speaker: asuna")
            except Exception as e:
                logger.warning(f"Failed to register default Asuna speaker: {e}")

    def _save_mappings(self) -> None:
        """Save speaker mappings to JSON file."""
        try:
            with open(self.mappings_file, 'w') as f:
                json.dump(self._mappings, f, indent=2)
            logger.debug(f"Saved {len(self._mappings)} speaker mappings to {self.mappings_file}")
        except Exception as e:
            logger.error(f"Failed to save speaker mappings: {e}")
            raise

    def add_speaker(
        self,
        speaker_id: str,
        name: str,
        audio_path: str,
        description: str | None = None,
        gender: str | None = None,
        language: str | None = None,
    ) -> SpeakerMapping:
        """Add a new speaker mapping.

        Args:
            speaker_id: Unique identifier for the speaker
            name: Display name
            audio_path: Path to reference audio file
            description: Optional description
            gender: Optional gender (male/female/neutral)
            language: Optional language

        Returns:
            The created SpeakerMapping

        Raises:
            HTTPException: If speaker already exists or validation fails
        """
        if speaker_id in self._mappings:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Speaker '{speaker_id}' already exists"
            )

        # Validate audio path exists
        if not Path(audio_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference audio file not found: {audio_path}"
            )

        mapping = SpeakerMapping(
            speaker_id=speaker_id,
            name=name,
            audio_path=audio_path,
            description=description,
            gender=gender,
            language=language,
            is_active=True,
        )

        self._mappings[speaker_id] = mapping.dict()
        self._save_mappings()

        logger.info(f"Added speaker '{speaker_id}': {name} -> {audio_path}")
        return mapping

    def register_speaker(
        self,
        speaker_id: str,
        audio_path: str,
        display_name: str | None = None,
        gender: str = "unknown",
        description: str = "",
    ) -> SpeakerMapping:
        """Register a new speaker (alias for add_speaker with different parameter names).

        This method is an alias for add_speaker with parameter names matching
        the external API contract used by the TTSSettings component.

        Args:
            speaker_id: Unique identifier for the speaker
            audio_path: Path to reference audio file
            display_name: Display name (defaults to speaker_id if not provided)
            gender: Speaker gender (default: unknown)
            description: Optional description

        Returns:
            The created SpeakerMapping
        """
        return self.add_speaker(
            speaker_id=speaker_id,
            name=display_name or speaker_id,
            audio_path=audio_path,
            description=description or None,
            gender=gender,
        )

    def update_speaker(
        self,
        speaker_id: str,
        name: str | None = None,
        audio_path: str | None = None,
        description: str | None = None,
        gender: str | None = None,
        language: str | None = None,
        is_active: bool | None = None,
    ) -> SpeakerMapping:
        """Update an existing speaker mapping.

        Args:
            speaker_id: Speaker to update
            name: New display name
            audio_path: New audio path
            description: New description
            gender: New gender
            language: New language
            is_active: New active status

        Returns:
            The updated SpeakerMapping

        Raises:
            HTTPException: If speaker not found or validation fails
        """
        if speaker_id not in self._mappings:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Speaker '{speaker_id}' not found"
            )

        current = self._mappings[speaker_id]

        # Update fields
        if name is not None:
            current['name'] = name
        if audio_path is not None:
            if not Path(audio_path).exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Reference audio file not found: {audio_path}"
                )
            current['audio_path'] = audio_path
        if description is not None:
            current['description'] = description
        if gender is not None:
            current['gender'] = gender
        if language is not None:
            current['language'] = language
        if is_active is not None:
            current['is_active'] = is_active

        # Validate the updated mapping
        mapping = SpeakerMapping(**current)
        self._mappings[speaker_id] = mapping.dict()
        self._save_mappings()

        logger.info(f"Updated speaker '{speaker_id}'")
        return mapping

    def remove_speaker(self, speaker_id: str) -> None:
        """Remove a speaker mapping.

        Args:
            speaker_id: Speaker to remove

        Raises:
            HTTPException: If speaker not found
        """
        if speaker_id not in self._mappings:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Speaker '{speaker_id}' not found"
            )

        audio_path = self._mappings[speaker_id].get('audio_path', '')

        # Check if audio file is in our storage directory
        # If so, ask if we should delete it (for now, we'll keep it)
        audio_file = Path(audio_path)
        if audio_file.exists() and audio_file.is_relative_to(self.audio_storage_dir):
            logger.info(f"Audio file {audio_path} is in storage directory. Consider manual cleanup if needed.")

        del self._mappings[speaker_id]
        self._save_mappings()

        logger.info(f"Removed speaker '{speaker_id}'")

    def get_speaker(self, speaker_id: str) -> SpeakerMapping:
        """Get a speaker mapping by ID.

        Args:
            speaker_id: Speaker to retrieve

        Returns:
            The SpeakerMapping

        Raises:
            HTTPException: If speaker not found
        """
        if speaker_id not in self._mappings:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Speaker '{speaker_id}' not found"
            )

        return SpeakerMapping(**self._mappings[speaker_id])

    def speaker_exists(self, speaker_id: str) -> bool:
        """Check if a speaker exists.

        Args:
            speaker_id: Speaker to check

        Returns:
            True if speaker exists and is active, False otherwise
        """
        if speaker_id not in self._mappings:
            return False
        return self._mappings[speaker_id].get('is_active', True)

    def get_audio_path(self, speaker_id: str) -> str:
        """Get the audio path for a speaker.

        Args:
            speaker_id: Speaker to get audio path for

        Returns:
            The audio file path

        Raises:
            HTTPException: If speaker not found
        """
        if speaker_id not in self._mappings:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Speaker '{speaker_id}' not found"
            )

        return self._mappings[speaker_id]['audio_path']

    def list_speakers(self, active_only: bool = False) -> list[SpeakerMapping]:
        """List all speaker mappings.

        Args:
            active_only: If True, only return active speakers

        Returns:
            List of SpeakerMapping objects
        """
        mappings = []
        for _speaker_id, data in self._mappings.items():
            if active_only and not data.get('is_active', True):
                continue
            mappings.append(SpeakerMapping(**data))
        return mappings

    def get_speaker_dict(self) -> dict[str, str]:
        """Get a simple dict mapping speaker_id to audio_path.

        Useful for compatibility with existing code.

        Returns:
            Dict mapping speaker_id to audio_path for active speakers
        """
        return {
            speaker_id: data['audio_path']
            for speaker_id, data in self._mappings.items()
            if data.get('is_active', True)
        }

    def save_uploaded_audio(self, filename: str, content: bytes) -> str:
        """Save an uploaded audio file to storage.

        Args:
            filename: Original filename
            content: File content bytes

        Returns:
            Path to saved file

        Raises:
            HTTPException: If file format is invalid
        """
        # Validate file extension
        ext = Path(filename).suffix.lower()
        if ext not in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio file format: {ext}. Supported: .wav, .mp3, .flac, .ogg, .m4a"
            )

        # Generate safe filename
        import uuid
        safe_filename = f"{uuid.uuid4()}{ext}"
        output_path = self.audio_storage_dir / safe_filename

        # Save file
        with open(output_path, 'wb') as f:
            f.write(content)

        logger.info(f"Saved uploaded audio to {output_path}")
        return str(output_path)

    def get_storage_dir(self) -> Path:
        """Get the storage directory path."""
        return self.storage_dir

    def get_audio_storage_dir(self) -> Path:
        """Get the audio storage directory path."""
        return self.audio_storage_dir


# Global singleton instance
_speaker_manager: SpeakerManager | None = None


def get_speaker_manager() -> SpeakerManager:
    """Get the global speaker manager instance."""
    global _speaker_manager
    if _speaker_manager is None:
        _speaker_manager = SpeakerManager()
    return _speaker_manager


# Export singleton for direct import (backwards compatibility)
speaker_manager = get_speaker_manager()
