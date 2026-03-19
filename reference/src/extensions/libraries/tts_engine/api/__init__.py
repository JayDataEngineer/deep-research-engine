"""
TTS API Integration - Local TTS Services.

This module provides TTS API endpoints using local services
extracted from CozyTTS (no submodule dependency).

Note: IndexTTS external API routes have been moved to the IndexTTS extension:
src/extensions/services/index_tts/routes/

Note: Emotion tagging has been moved to audio_producer extension:
src/extensions/features/audio_producer/api.py
"""

import logging
from pathlib import Path

from fastapi import APIRouter

# Import local TTS API routers
try:
    from .v1 import tts as tts_router
    from .v1 import models as models_router
    from .v1 import audio as audio_router
    from .v1 import audio_analysis as audio_analysis_router
    from .v1 import audio_qa as audio_qa_router
    from .v1 import batch as batch_router
    from .v1 import config as config_router
    from .v1 import emotion_tagger as emotion_tagger_router
    # emotion_tagging moved to audio_producer
    from .v1 import chunked_tts as chunked_tts_router
    from .v1 import voice_input as voice_input_router
    from .v1 import extension_process as extension_process_router
    from .v1 import chat_tts as chat_tts_router
    from .v1 import cli as cli_router

    TTS_AVAILABLE = True
except ImportError as e:
    TTS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"TTS services not available: {e}")
    # Set placeholders for when imports fail
    tts_router = None


def create_tts_router() -> APIRouter:
    """Create and return the TTS API router.

    Note: IndexTTS external routes (/api/v1/external/indextts/*) are handled
    by the IndexTTS extension at src/extensions/services/index_tts/routes/

    Returns:
        APIRouter with all TTS endpoints
    """
    router = APIRouter(prefix="/api/v1/tts", tags=["Text-to-Speech"])

    if not TTS_AVAILABLE:
        return router

    # Include all TTS routes
    router.include_router(tts_router.router)
    router.include_router(models_router.router)
    router.include_router(audio_router.router)
    router.include_router(audio_analysis_router.router)
    router.include_router(audio_qa_router.router)
    router.include_router(batch_router.router)
    router.include_router(config_router.router)
    router.include_router(emotion_tagger_router.router)
    # emotion_tagging moved to audio_producer extension
    router.include_router(chunked_tts_router.router)
    router.include_router(voice_input_router.router)
    router.include_router(extension_process_router.router)
    router.include_router(chat_tts_router.router)
    router.include_router(cli_router.router)

    return router


# Export the main router factory
__all__ = ["create_tts_router", "TTS_AVAILABLE"]
