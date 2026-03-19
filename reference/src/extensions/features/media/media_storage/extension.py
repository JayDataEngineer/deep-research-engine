"""Media Storage Extension

Handles media file uploads (images, audio, video) with validation and storage.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.extensions.core import LogicExtension, ExtensionContext
from src.config import settings
from src.utils.media import (
    save_uploaded_file,
    get_media_type,
    is_supported_format,
    validate_file_size,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
)

logger = logging.getLogger(__name__)


class MediaStorageExtension(LogicExtension):
    """Handle media file uploads and storage.

    Provides validation for file size and format, and stores uploaded files
    in the configured upload directory.
    """

    name = "media_storage"

    def __init__(self, config: dict, root_path):
        super().__init__(config, root_path)
        self.upload_dir: Optional[Path] = None

    async def initialize(self, context: ExtensionContext):
        """Initialize the extension (abstract method)."""
        self.context = context

    async def shutdown(self):
        """Shutdown the extension (abstract method)."""
        pass

    async def process(self, input_data: dict, **kwargs) -> dict:
        """Process method - not used for this API-driven extension."""
        return {}

    async def activate(self, context: ExtensionContext):
        """Initialize the extension."""
        await super().activate(context)
        self.upload_dir = Path(context.settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[MEDIA_STORAGE] Activated with upload_dir: {self.upload_dir}")

    async def deactivate(self):
        """Clean up resources."""
        await super().deactivate()
        logger.info("[MEDIA_STORAGE] Deactivated")

    @property
    def router(self) -> Optional[APIRouter]:
        """Get the FastAPI router for this extension."""
        router = APIRouter(prefix="/media", tags=["media"])

        @router.post("/upload")
        async def upload_media(
            file: UploadFile = File(...),
            type: str = Form(...),
        ):
            """Upload a media file (image, audio, or video)."""
            if type not in ["image", "audio", "video"]:
                raise HTTPException(status_code=400, detail="Invalid media type")

            file_data = await file.read()

            if not validate_file_size(file_data):
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.max_upload_size / (1024*1024):.0f}MB",
                )

            if not is_supported_format(file.filename, type):
                supported = {
                    "image": list(SUPPORTED_IMAGE_FORMATS),
                    "audio": list(SUPPORTED_AUDIO_FORMATS),
                    "video": list(SUPPORTED_VIDEO_FORMATS),
                }.get(type, set())
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Supported: {', '.join(supported)}",
                )

            try:
                url, path = save_uploaded_file(file_data, file.filename, type)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

            return {
                "url": url,
                "type": type,
                "size": len(file_data),
                "filename": file.filename,
            }

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(MediaStorageExtension)
