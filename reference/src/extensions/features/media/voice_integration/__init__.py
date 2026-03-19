"""Voice Integration Extension

Provides STT (Speech-to-Text) and TTS (Text-to-Speech) capabilities
for all chat modes without modifying their core logic.
"""

from .extension import VoiceIntegrationExtension, register_extension

__all__ = ["VoiceIntegrationExtension", "register_extension"]
