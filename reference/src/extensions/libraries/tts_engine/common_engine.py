"""
Common TTS Engine Base Class

Provides a reusable base class for TTS engine implementations,
reducing boilerplate code across extensions.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.extensions.libraries.tts_engine.model_interface import TTSModel

logger = logging.getLogger(__name__)


class LazyImportMixin:
    """
    Mixin for lazy-loading heavy dependencies.

    Useful for extensions with large dependencies (torch, transformers, etc.)
    that shouldn't be imported at module load time.
    """

    def _ensure_imports(self) -> None:
        """
        Lazy-load heavy dependencies.

        Override this in subclasses to lazy-import dependencies.
        Example:
            def _ensure_imports(self):
                global torch, Model
                if torch is None:
                    import torch
                if Model is None:
                    from somewhere import Model
        """
        pass


class VoiceListMixin:
    """
    Mixin for extensions that define voices as a simple list.

    Extensions can define a VOICES list and get get_voices() for free.
    """

    # Override this in subclasses
    VOICES: list[dict[str, Any]] = []

    def get_voices(self) -> list[dict[str, Any]]:
        """Get available voices from the VOICES list."""
        return self.VOICES


class CommonTTSEngine(TTSModel, LazyImportMixin):
    """
    Common base class for TTS engine implementations.

    Provides standard implementations of common TTSModel methods,
    reducing boilerplate in engine.py files.

    Usage:
        class MyTTSEngine(CommonTTSEngine):
            def __init__(self):
                super().__init__("my_extension", 24000)

            def initialize(self, config: dict[str, Any]) -> bool:
                # Your initialization logic
                pass

            def synthesize(self, text: str, voice: str, **params) -> tuple[np.ndarray, int]:
                # Your synthesis logic
                pass
    """

    def __init__(self, name: str, default_sample_rate: int = 24000):
        """
        Initialize the common engine.

        Args:
            name: Extension name
            default_sample_rate: Default sample rate for audio output
        """
        self._name = name
        self._default_sample_rate = default_sample_rate
        self.model = None
        self.is_initialized = False

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> bool:
        """
        Initialize the model with configuration.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def synthesize(self, text: str, voice: str, **params) -> tuple:
        """
        Synthesize speech from text.

        Must be implemented by subclasses.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info(f"Cleaning up {self.get_name()} model")
        self.model = None
        self.is_initialized = False

    def get_name(self) -> str:
        """Return model name."""
        return self._name

    def get_sample_rate(self) -> int:
        """Return preferred sample rate."""
        return self._default_sample_rate

    def is_available(self) -> bool:
        """Check if model is available."""
        return True

    def get_voices(self) -> list[dict[str, Any]]:
        """
        Get available voices.

        Default implementation returns empty list.
        Override or use VoiceListMixin for voice support.
        """
        return []


class ConfigModelMixin:
    """
    Mixin for extensions that use config-based model initialization.

    Provides standard methods for getting model paths from config.
    """

    def _get_model_path(self, config: dict[str, Any], default_path: str) -> str:
        """
        Get model path from config with fallback.

        Args:
            config: Model configuration dict
            default_path: Default path if not in config
        """
        return config.get("model_path", config.get("model_dir", default_path))

    def _get_bool_config(self, config: dict[str, Any], key: str, default: bool = False) -> bool:
        """Get boolean config value with fallback."""
        value = config.get(key, default)
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def _get_float_config(self, config: dict[str, Any], key: str, default: float = 1.0) -> float:
        """Get float config value with fallback."""
        value = config.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


class StandardTTSEngine(CommonTTSEngine, VoiceListMixin, ConfigModelMixin):
    """
    Standard TTS engine with all common mixins.

    This is the recommended base class for new TTS extensions.

    Provides:
    - Lazy loading support
    - Voice list management
    - Config-based model initialization
    - Standard cleanup, get_name, get_sample_rate methods
    """

    pass
