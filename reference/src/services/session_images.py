"""Session-based storage for last generated images.

Allows editing the last generated image in a chat session.
"""

import base64
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class SessionImageStore:
    """Stores the last generated image per session.

    Images are stored in memory with a TTL for cleanup.
    """

    def __init__(self, ttl_minutes: int = 60):
        """Initialize the image store.

        Args:
            ttl_minutes: Time to live for images (default 1 hour)
        """
        self._store: Dict[str, dict] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.RLock()

    def store_image(
        self,
        session_id: str,
        image_base64: str,
        prompt: str = "",
        image_path: str = "",
    ) -> None:
        """Store a generated image for a session.

        Args:
            session_id: Unique session identifier
            image_base64: Base64 encoded image data
            prompt: The prompt used to generate the image
            image_path: Optional path to saved image file
        """
        with self._lock:
            # Extract base64 data if it's a data URL
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            self._store[session_id] = {
                "image_base64": image_base64,
                "prompt": prompt,
                "image_path": image_path,
                "timestamp": datetime.now(),
            }
            logger.info(f"[SESSION IMAGE] Stored image for session {session_id}")

    def get_image(self, session_id: str) -> Optional[dict]:
        """Get the last generated image for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Dict with image_base64, prompt, image_path, or None if not found
        """
        with self._lock:
            data = self._store.get(session_id)
            if data is None:
                return None

            # Check if expired
            if datetime.now() - data["timestamp"] > self._ttl:
                del self._store[session_id]
                logger.info(f"[SESSION IMAGE] Expired image for session {session_id}")
                return None

            return data

    def has_image(self, session_id: str) -> bool:
        """Check if a session has a stored image.

        Args:
            session_id: Unique session identifier

        Returns:
            True if image exists and is not expired
        """
        return self.get_image(session_id) is not None

    def clear_image(self, session_id: str) -> bool:
        """Clear the stored image for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            True if image was removed, False if not found
        """
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.info(f"[SESSION IMAGE] Cleared image for session {session_id}")
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired images.

        Returns:
            Number of images removed
        """
        with self._lock:
            now = datetime.now()
            expired_keys = [
                k for k, v in self._store.items()
                if now - v["timestamp"] > self._ttl
            ]
            for key in expired_keys:
                del self._store[key]
            return len(expired_keys)


# Global singleton
_image_store: Optional[SessionImageStore] = None


def get_image_store() -> SessionImageStore:
    """Get the global image store instance."""
    global _image_store
    if _image_store is None:
        _image_store = SessionImageStore()
    return _image_store
