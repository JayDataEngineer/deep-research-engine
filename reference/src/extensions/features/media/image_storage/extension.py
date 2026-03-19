"""Image Storage Extension

Handles image extraction and storage from user messages for image editing functionality.
This is a stateful service that maintains a session-to-image mapping.
"""

import base64
import logging
from typing import Optional

from src.extensions.core import LogicExtension, ExtensionContext

logger = logging.getLogger(__name__)


class ImageStorageExtension(LogicExtension):
    """Handle image storage and extraction from messages.

    Maintains an in-memory mapping of session IDs to image bytes,
    used for character image editing functionality.
    """

    name = "image_storage"

    def __init__(self, config: dict, root_path):
        super().__init__(config, root_path)
        # Store images per session for image editing
        self._session_images: dict[str, bytes] = {}

    async def initialize(self, context: ExtensionContext):
        """Initialize the extension."""
        self.context = context
        logger.info("[IMAGE_STORAGE] Initialized")

    async def shutdown(self):
        """Clean up resources."""
        self._session_images.clear()
        logger.info("[IMAGE_STORAGE] Shutdown")

    async def process(self, input_data: dict, **kwargs):
        """Process method - not used for this service extension."""
        return {}

    # ==========================================================================
    # Image Storage API
    # ==========================================================================

    def extract_and_store_images(self, messages: list, session_id: str) -> None:
        """Extract images from user messages and store them for the session.

        Args:
            messages: List of message dicts
            session_id: Session ID to associate the image with
        """
        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")
            if not isinstance(content, list):
                continue

            for part in content:
                if part.get("type") == "image_url":
                    url_data = part.get("image_url", {})
                    url = url_data.get("url", "")

                    if not url:
                        continue

                    # Extract base64 data
                    if url.startswith("data:"):
                        # data:image/png;base64,iVBORw0KGgo...
                        try:
                            _, base64_data = url.split(",", 1)
                            image_bytes = base64.b64decode(base64_data)
                            self._session_images[session_id] = image_bytes
                            logger.info(f"[IMAGE] Stored image for session {session_id}: {len(image_bytes)} bytes")
                        except Exception as e:
                            logger.error(f"[IMAGE] Failed to decode image: {e}")

    def get_session_image(self, session_id: str) -> Optional[bytes]:
        """Get the stored image for a session.

        Args:
            session_id: Session ID

        Returns:
            Image bytes or None if no image stored
        """
        return self._session_images.get(session_id)

    def clear_session_image(self, session_id: str) -> bool:
        """Clear the stored image for a session.

        Args:
            session_id: Session ID

        Returns:
            True if image was cleared, False if no image was stored
        """
        if session_id in self._session_images:
            del self._session_images[session_id]
            logger.info(f"[IMAGE] Cleared image for session {session_id}")
            return True
        return False


# Singleton instance for module-level access
_instance: Optional[ImageStorageExtension] = None


def get_image_storage() -> Optional[ImageStorageExtension]:
    """Get the image storage extension instance.

    Returns:
        ImageStorageExtension instance or None if not activated
    """
    from src.extensions.core.manager import get_extension_manager
    global _instance

    if _instance is None:
        extension_manager = get_extension_manager()
        if extension_manager:
            _instance = extension_manager.get_extension("image_storage")
        else:
            logger.warning("[IMAGE_STORAGE] Extension manager not available")

    return _instance

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(ImageStorageExtension)
