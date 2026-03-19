"""
Audio History Manager with LRU Cache.

This module manages audio file history with automatic cleanup when the
history exceeds a maximum size. Uses an LRU (Least Recently Used) cache
strategy to remove oldest files when needed.
"""

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import settings
from src.extensions.libraries.tts_engine.api.utils.storage import get_output_directory

logger = logging.getLogger(__name__)

# Maximum number of audio files to keep in history
MAX_HISTORY_ITEMS = 50


@dataclass
class AudioHistoryEntry:
    """Represents a single entry in the audio history."""

    audio_id: str
    format: str
    file_path: str
    created_at: datetime
    size_bytes: int

    def __hash__(self) -> int:
        """Make the entry hashable for use in sets/dicts."""
        return hash(f"{self.audio_id}.{self.format}")


class AudioHistoryManager:
    """
    Manages audio file history with LRU cache.

    This class tracks generated audio files and automatically removes old files
    when the history exceeds MAX_HISTORY_ITEMS. Uses an OrderedDict to maintain
    insertion order for efficient LRU operations.
    """

    def __init__(self, max_items: int = MAX_HISTORY_ITEMS):
        """Initialize the AudioHistoryManager.

        Args:
            max_items: Maximum number of audio files to keep in history
        """
        self.max_items = max_items
        # OrderedDict maintains insertion order (oldest first)
        self._history: OrderedDict[str, AudioHistoryEntry] = OrderedDict()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the history manager by scanning existing files."""
        if self._initialized:
            return

        logger.info("Initializing audio history manager...")

        try:
            output_dir = Path(get_output_directory())
            if not output_dir.exists():
                logger.info(f"Output directory does not exist: {output_dir}")
                self._initialized = True
                return

            # Scan all audio files in the output directory
            self._scan_directory(output_dir)

            # If we have more files than max_items, clean up oldest
            if len(self._history) > self.max_items:
                logger.info(
                    f"Found {len(self._history)} existing files, "
                    f"cleaning up to {self.max_items}"
                )
                self._cleanup_old_files()

            self._initialized = True
            logger.info(
                f"Audio history manager initialized with {len(self._history)} files"
            )

        except Exception as e:
            logger.error(f"Error initializing audio history manager: {e}")
            self._initialized = True  # Still mark as initialized to avoid retry loops

    def _scan_directory(self, directory: Path) -> None:
        """Scan a directory for audio files and add them to history.

        Args:
            directory: Directory path to scan
        """
        supported_formats = {".wav", ".mp3", ".ogg"}

        # Check root directory
        self._scan_path(directory, supported_formats)

        # Check date subdirectories
        for item in directory.iterdir():
            if item.is_dir():
                self._scan_path(item, supported_formats)

    def _scan_path(self, path: Path, supported_formats: set[str]) -> None:
        """Scan a path for audio files.

        Args:
            path: Path to scan
            supported_formats: Set of supported file extensions
        """
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                try:
                    stat = file_path.stat()
                    created_at = datetime.fromtimestamp(stat.st_ctime)
                    size = stat.st_size

                    # Extract audio_id (remove extension and optional "audio_" prefix)
                    filename = file_path.stem
                    if filename.startswith("audio_"):
                        audio_id = filename[6:]  # Remove "audio_" prefix
                    else:
                        audio_id = filename

                    key = f"{audio_id}.{file_path.suffix[1:]}"

                    entry = AudioHistoryEntry(
                        audio_id=audio_id,
                        format=file_path.suffix[1:],
                        file_path=str(file_path),
                        created_at=created_at,
                        size_bytes=size,
                    )

                    self._history[key] = entry

                except Exception as e:
                    logger.warning(f"Error scanning file {file_path}: {e}")

    def add_entry(self, audio_id: str, format: str, file_path: str) -> None:
        """Add a new audio file to the history.

        If adding this entry exceeds max_items, the oldest entry will be removed.

        Args:
            audio_id: Unique identifier for the audio
            format: Audio format (wav, mp3, ogg)
            file_path: Full path to the audio file
        """
        key = f"{audio_id}.{format}"

        try:
            file_stat = os.stat(file_path)
            entry = AudioHistoryEntry(
                audio_id=audio_id,
                format=format,
                file_path=file_path,
                created_at=datetime.now(),
                size_bytes=file_stat.st_size,
            )

            # If this key already exists, remove it first (will be re-added at the end)
            if key in self._history:
                del self._history[key]

            # Add new entry at the end (most recent)
            self._history[key] = entry

            # Clean up if we exceed max items
            if len(self._history) > self.max_items:
                self._cleanup_old_files()

            logger.debug(
                f"Added audio entry: {key}, history size: {len(self._history)}/{self.max_items}"
            )

        except Exception as e:
            logger.error(f"Error adding audio entry {key}: {e}")

    def _cleanup_old_files(self) -> None:
        """Remove oldest files from history and disk.

        Removes files in FIFO order (oldest first) until we're at max_items.
        """
        num_to_remove = len(self._history) - self.max_items

        if num_to_remove <= 0:
            return

        logger.info(f"Cleaning up {num_to_remove} old audio file(s)...")

        # Remove oldest entries (from the beginning of OrderedDict)
        for _ in range(num_to_remove):
            if not self._history:
                break

            # Get and remove the oldest entry (first item)
            key, entry = next(iter(self._history.items()))
            del self._history[key]

            # Delete the file from disk
            try:
                if os.path.exists(entry.file_path):
                    os.remove(entry.file_path)
                    logger.debug(f"Deleted old audio file: {entry.file_path}")

                    # Also try to remove empty parent directories
                    parent_dir = os.path.dirname(entry.file_path)
                    try:
                        if parent_dir != get_output_directory():
                            os.rmdir(parent_dir)
                            logger.debug(f"Removed empty directory: {parent_dir}")
                    except OSError:
                        pass  # Directory not empty, that's fine

            except Exception as e:
                logger.warning(f"Error deleting old audio file {entry.file_path}: {e}")

        logger.info(f"Cleanup complete. History size: {len(self._history)}")

    def remove_entry(self, audio_id: str, format: str) -> bool:
        """Remove an entry from history and delete the file.

        Args:
            audio_id: Unique identifier for the audio
            format: Audio format

        Returns:
            True if entry was found and removed, False otherwise
        """
        key = f"{audio_id}.{format}"

        if key not in self._history:
            return False

        entry = self._history[key]
        del self._history[key]

        # Delete the file
        try:
            if os.path.exists(entry.file_path):
                os.remove(entry.file_path)
                logger.debug(f"Removed audio file: {entry.file_path}")
                return True
        except Exception as e:
            logger.error(f"Error removing audio file {entry.file_path}: {e}")

        return False

    def get_entry(self, audio_id: str, format: str) -> Optional[AudioHistoryEntry]:
        """Get an entry from history without changing its LRU status.

        Args:
            audio_id: Unique identifier for the audio
            format: Audio format

        Returns:
            AudioHistoryEntry if found, None otherwise
        """
        key = f"{audio_id}.{format}"
        return self._history.get(key)

    def touch_entry(self, audio_id: str, format: str) -> bool:
        """Mark an entry as recently used (move to end of LRU).

        Args:
            audio_id: Unique identifier for the audio
            format: Audio format

        Returns:
            True if entry was found and updated, False otherwise
        """
        key = f"{audio_id}.{format}"

        if key not in self._history:
            return False

        # Move to end (most recently used)
        entry = self._history.pop(key)
        self._history[key] = entry
        return True

    def get_all_entries(self) -> list[AudioHistoryEntry]:
        """Get all entries in history, ordered from oldest to newest.

        Returns:
            List of AudioHistoryEntry objects
        """
        return list(self._history.values())

    def get_history_size(self) -> int:
        """Get the current number of entries in history.

        Returns:
            Number of entries
        """
        return len(self._history)

    def get_total_size_bytes(self) -> int:
        """Get the total size of all files in history.

        Returns:
            Total size in bytes
        """
        return sum(entry.size_bytes for entry in self._history.values())

    def clear_all(self) -> int:
        """Clear all entries and delete all files.

        Returns:
            Number of files deleted
        """
        count = 0

        for key, entry in list(self._history.items()):
            try:
                if os.path.exists(entry.file_path):
                    os.remove(entry.file_path)
                    count += 1
            except Exception as e:
                logger.error(f"Error deleting file {entry.file_path}: {e}")

        self._history.clear()
        logger.info(f"Cleared all audio history, deleted {count} file(s)")

        return count


# Global singleton instance
audio_history_manager = AudioHistoryManager()
