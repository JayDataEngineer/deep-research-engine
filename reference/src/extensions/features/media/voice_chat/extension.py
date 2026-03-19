"""Voice Chat Extension.

Provides endpoints for two-way voice conversations:
1. Accept audio input (STT)
2. Return audio output (TTS)
3. Support all chat modes (direct, tool_call, deep_agent, deep_research)
4. Stream both text and audio chunks
"""

import json
import logging
import uuid
from pathlib import Path
from typing import AsyncIterator, Any

from fastapi import APIRouter
from src.extensions.core.base import LogicExtension, ExtensionStatus

logger = logging.getLogger(__name__)


class VoiceChatExtension(LogicExtension):
    """Voice chat extension with STT + TTS support."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router: APIRouter = None

    async def initialize(self, context):
        """Initialize the voice chat extension."""
        await super().initialize(context)
        logger.info("Voice chat extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(self, input_data: dict, **kwargs):
        """Process voice chat request."""
        action = input_data.get('action', 'chat')

        if action == 'chat':
            return await self.voice_chat(input_data)
        elif action == 'health':
            return {"status": "healthy", "service": "voice-chat"}
        elif action == 'list_characters':
            return await self.list_characters()
        else:
            return {"error": f"Unknown action: {action}"}

    @property
    def router(self) -> APIRouter:
        """Get the API router for this extension."""
        if self._router is None:
            from .routes import router
            self._router = router
        return self._router

    async def voice_chat(self, input_data: dict) -> dict:
        """Voice chat endpoint logic."""
        # This is handled by the routes module
        pass

    async def list_characters(self) -> dict:
        """List all available characters for voice chat."""
        try:
            from src.extensions.libraries.character_engine.crud import get_character_storage

            storage = get_character_storage()
            characters = storage.list_characters()

            # Return simplified list with voice info
            result = []
            for char in characters:
                name = char.get("name", "")
                if name:
                    # Check if character has voice configured
                    full_char = storage.load_character(name)
                    voice_id = full_char.get("voice_id") if full_char else None

                    result.append({
                        "name": name,
                        "has_voice": voice_id is not None,
                        "voice_id": voice_id,
                    })

            return {"characters": result}

        except Exception as e:
            logger.error(f"[VOICE CHAT] Error listing characters: {e}")
            return {"characters": [], "error": str(e)}

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(VoiceChatExtension)
