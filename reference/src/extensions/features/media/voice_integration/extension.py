"""Voice Integration Extension

Provides STT (Speech-to-Text) and TTS (Text-to-Speech) capabilities
for all chat modes without modifying their core logic.

This is a service extension that provides:
- STT processing for audio input
- TTS streaming for audio output
- Character-to-voice mapping
- Multipart streaming coordination
"""

import logging
from pathlib import Path
from typing import Any, Dict

from src.extensions.core.base import LogicExtension

from .tts_streamer import TTSStreamer
from .stt_processor import STTProcessor
from .voice_mapper import VoiceMapper

logger = logging.getLogger(__name__)


class VoiceIntegrationExtension(LogicExtension):
    """
    Voice integration extension.

    This is a service extension that provides:
    - STT processing for audio input
    - TTS streaming for audio output
    - Character-to-voice mapping
    - Multipart streaming coordination
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._tts_streamer = None
        self._stt_processor = None
        self._voice_mapper = None

        # Get configuration
        ext_config = config.get("config", {})
        self._default_voice_id = ext_config.get("default_voice_id")
        self._chunk_size = ext_config.get("chunk_size_bytes", 8192)

    async def initialize(self, context):
        """Initialize voice integration subsystems.

        Args:
            context: Application context
        """
        await super().initialize(context)

        # Initialize TTS streamer
        self._tts_streamer = TTSStreamer(chunk_size=self._chunk_size)
        await self._tts_streamer.initialize(context)

        # Initialize STT processor
        self._stt_processor = STTProcessor()
        await self._stt_processor.initialize(context)

        # Initialize voice mapper
        self._voice_mapper = VoiceMapper(default_voice_id=self._default_voice_id)
        await self._voice_mapper.initialize(context)

        logger.info("VoiceIntegrationExtension: All subsystems initialized")

    async def shutdown(self):
        """Cleanup subsystems."""
        if self._tts_streamer:
            await self._tts_streamer.close()
        if self._stt_processor:
            await self._stt_processor.close()
        if self._voice_mapper:
            await self._voice_mapper.close()
        logger.info("VoiceIntegrationExtension: Shutdown complete")

    async def process(self, input_data: dict, **kwargs):
        """
        Process voice chat request.

        This is NOT the main entry point - voice chat requests
        go through the dedicated /voice/chat endpoint.

        This method is provided for extension compatibility.

        Args:
            input_data: Input data dictionary
            **kwargs: Additional keyword arguments

        Yields:
            Status events
        """
        yield {"status": "voice_integration_active"}

    @property
    def tts_streamer(self) -> TTSStreamer:
        """Get the TTS streamer instance."""
        return self._tts_streamer

    @property
    def stt_processor(self) -> STTProcessor:
        """Get the STT processor instance."""
        return self._stt_processor

    @property
    def voice_mapper(self) -> VoiceMapper:
        """Get the voice mapper instance."""
        return self._voice_mapper

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(VoiceIntegrationExtension)
