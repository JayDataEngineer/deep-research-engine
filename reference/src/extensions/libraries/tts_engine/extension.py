"""
TTS Engine Extension

Provides Text-to-Speech API endpoints using local TTS services.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from src.extensions.core.base import BaseExtension, ExtensionStatus


class TTSEngineExtension(BaseExtension):
    """
    TTS engine extension for managing text-to-speech services.

    Provides API endpoints for TTS generation, audio processing,
    emotion tagging, and voice management.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._status = ExtensionStatus.LOADED

    @property
    def status(self) -> ExtensionStatus:
        return self._status

    def register_routes(self, app) -> None:
        """Register TTS API routes."""
        try:
            from .api import create_tts_router, TTS_AVAILABLE
            if TTS_AVAILABLE:
                tts_router = create_tts_router()
                app.include_router(tts_router, tags=["tts"])
        except ImportError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"TTS routes not available: {e}")

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/api/v1/tts",
            "name": "tts",
            "tags": ["tts", "audio", "speech", "voice"],
            "description": "Text-to-Speech API with multiple engine support",
        }

    async def initialize(self, context) -> None:
        """Initialize the TTS engine."""
        self._status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Shutdown the TTS engine."""
        self._status = ExtensionStatus.UNLOADED
