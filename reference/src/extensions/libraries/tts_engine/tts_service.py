"""
TTS Service compatibility shim.

This module re-exports from router.py for backward compatibility.
The tts_service.py was renamed to router.py during refactoring.
"""

from .router import TTSService, get_tts_service, tts_service

__all__ = ["TTSService", "get_tts_service", "tts_service"]
