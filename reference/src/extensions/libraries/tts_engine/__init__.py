"""
TTS Engine Library - Text-to-Speech functionality.

This package contains all TTS-related services for extension management
and audio generation.

NOTE: The core ExtensionManager is now in src.extensions.core.
This package re-exports it for backward compatibility.
"""

from .base import BaseTTSExtension, SimpleModelExtension
from .router import TTSService, get_tts_service, tts_service
from .speakers import SpeakerManager, get_speaker_manager, speaker_manager
from .text_chunker import TextChunker
from .emotion_tagger import EmotionTagger
from .audio_utils import AudioProcessor
from .audio_history_manager import AudioHistoryManager, audio_history_manager
from .config_storage import TTSConfigStorage as ConfigStorage
# ⚠️ DEPRECATED: discover_extensions is deprecated. Use ExtensionManager from src.extensions.core.manager
from .extension_discovery import discover_extensions_from_path as discover_extensions  # noqa: F401
from .extension_process_manager import ExtensionProcessManager, extension_process_manager
from .extension_unload import generic_unload_extension as unload_extension
from .loading_tracker import LoadingTracker
# Optional import - psutil may not be available in all venvs
try:
    from .memory_monitor import MemoryMonitor
except ImportError:
    MemoryMonitor = None
from .model_interface import TTSModel as ModelInterface
from .model_validation_service import ModelValidationService
from .parameter_registry import ParameterRegistry

# Import from core extension system
from src.extensions.core.base import TTSExtensionInterface
from src.extensions.core import ExtensionManager

# Import audio_service from shared/tools (re-export for compatibility)
from src.shared.tools.audio_utils import AudioService, audio_service

# Re-export TTSExtensionInterface as TTSExtension for backward compatibility
TTSExtension = TTSExtensionInterface

__all__ = [
    # Core extension system
    "TTSExtensionInterface",
    "TTSExtension",  # Alias for TTSExtensionInterface for backward compatibility
    "BaseTTSExtension",
    "SimpleModelExtension",
    "ExtensionManager",  # Now imported from src.extensions.core
    # TTS service
    "TTSService",
    "get_tts_service",
    "tts_service",
    # Speaker management
    "SpeakerManager",
    "get_speaker_manager",
    "speaker_manager",
    # Utilities
    "TextChunker",
    "EmotionTagger",
    "AudioProcessor",
    "AudioHistoryManager",
    "audio_history_manager",
    "ConfigStorage",
    # Extension discovery and loading
    "discover_extensions",
    "ExtensionProcessManager",
    "extension_process_manager",
    "unload_extension",
    "LoadingTracker",
    "MemoryMonitor",
    "ModelInterface",
    "ModelValidationService",
    "ParameterRegistry",
    # Audio service (re-exported from shared/tools)
    "AudioService",
    "audio_service",
]
