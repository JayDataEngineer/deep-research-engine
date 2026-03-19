"""
Minimal model interface for TTS models.

This module defines a simplified interface for TTS models that separates
model logic from extension infrastructure concerns.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TTSModel(ABC):
    """Minimal interface for TTS models.

    This interface focuses only on the core TTS functionality,
    leaving infrastructure concerns to extension adapters.
    """

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> bool:
        """Initialize the model with configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def synthesize(
        self, text: str, voice: str, **params
    ) -> tuple[np.ndarray, int] | tuple["torch.Tensor", int]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice identifier to use
            **params: Additional model-specific parameters

        Returns:
            Tuple of (audio_data, sample_rate)
            audio_data can be either numpy array or torch tensor
        """
        pass

    @abstractmethod
    def get_voices(self) -> list[dict[str, Any]]:
        """Get available voice information.

        Returns:
            List of voice dictionaries with keys: id, name, language, gender, description
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def get_name(self) -> str:
        """Return model name."""
        return self.__class__.__name__

    def get_sample_rate(self) -> int:
        """Return preferred sample rate."""
        return 22050

    def is_available(self) -> bool:
        """Check if model is available and properly initialized."""
        return True


class BaseModelWrapper:
    """Base wrapper for TTS models that provides common functionality.

    This class bridges the gap between the minimal TTSModel interface
    and the full TTSExtension interface, handling common concerns like
    audio processing, parameter validation, and error handling.
    """

    def __init__(self, model: TTSModel):
        """Initialize the wrapper with a TTS model.

        Args:
            model: TTS model instance
        """
        self.model = model
        self.is_initialized = False
        self.config = {}

    def initialize(self, config: dict[str, Any]) -> bool:
        """Initialize the model with configuration.

        Args:
            config: Model configuration

        Returns:
            True if successful
        """
        try:
            self.config = config
            success = self.model.initialize(config)
            self.is_initialized = success
            return success
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return False

    def synthesize(
        self, text: str, voice: str, **params
    ) -> tuple[np.ndarray, int] | tuple["torch.Tensor", int]:
        """Synthesize speech using the model.

        Args:
            text: Text to synthesize
            voice: Voice to use
            **params: Additional parameters

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        try:
            return self.model.synthesize(text, voice, **params)
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            raise

    def get_voices(self) -> list[dict[str, Any]]:
        """Get available voices from the model.

        Returns:
            List of voice dictionaries
        """
        try:
            return self.model.get_voices()
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return []

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            self.model.cleanup()
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Error cleaning up model: {str(e)}")

    def get_sample_rate(self) -> int:
        """Get the model's preferred sample rate."""
        return self.model.get_sample_rate()

    def is_available(self) -> bool:
        """Check if the model is available."""
        return self.model.is_available() and self.is_initialized


class ModelFactory:
    """Factory for creating TTS model instances."""

    _models = {}

    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a model class with the factory.

        Args:
            name: Model name
            model_class: Model class
        """
        cls._models[name] = model_class

    @classmethod
    def create_model(cls, name: str, **kwargs) -> TTSModel | None:
        """Create a model instance by name.

        Args:
            name: Model name
            **kwargs: Model initialization parameters

        Returns:
            Model instance or None if not found
        """
        if name not in cls._models:
            logger.error(f"Model '{name}' not registered")
            return None

        try:
            model_class = cls._models[name]
            return model_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating model '{name}': {str(e)}")
            return None

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available model names.

        Returns:
            List of model names
        """
        return list(cls._models.keys())


# Utility function to create a simple model wrapper
def create_model_wrapper(model_class: type, **model_kwargs) -> BaseModelWrapper:
    """Create a model wrapper for the given model class.

    Args:
        model_class: Model class to wrap
        **model_kwargs: Model initialization parameters

    Returns:
        Model wrapper instance
    """
    model = model_class(**model_kwargs)
    return BaseModelWrapper(model)
