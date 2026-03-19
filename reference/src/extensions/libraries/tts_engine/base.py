"""
Base extension class for TTS extensions.

This module provides a simplified base class that handles common functionality
for TTS extensions, reducing boilerplate code and ensuring consistent behavior.

TTS extensions now extend ServiceExtension from the core system.
"""

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.extensions.core.base import ServiceExtension
from src.extensions.libraries.tts_engine.api.schemas.tts import VoiceInfo
from src.extensions.libraries.tts_engine.audio_utils import AudioProcessor
from src.extensions.libraries.tts_engine.model_interface import BaseModelWrapper, TTSModel

logger = logging.getLogger(__name__)


class BaseTTSExtension(ServiceExtension):
    """Base class for TTS extensions that reduces boilerplate.

    This class handles common functionality like audio processing,
    parameter validation, and error handling, allowing extensions
    to focus on model-specific logic.

    TTS extensions should extend this class and implement the abstract methods.
    """

    def __init__(self, config: dict[str, Any], root_path: Path):
        """Initialize the base extension.

        Args:
            config: Extension configuration dictionary
            root_path: Path to the extension directory
        """
        super().__init__(config, root_path)
        self.model_wrapper: BaseModelWrapper | None = None
        self.voices: list[VoiceInfo] = []
        self._model_config: dict[str, Any] = {}
        self.is_loaded = False

    async def initialize(self, context: Any = None) -> None:
        """Initialize the extension (for compatibility with core ExtensionManager).

        TTS extensions use their own loading mechanism via load()/unload().
        This method stores the context and optionally loads the model.

        Args:
            context: Extension context with LLM factory, settings, etc.
        """
        self.context = context
        # TTS extensions can be loaded on-demand via load()
        # Optionally load immediately if configured to do so
        pass

    async def shutdown(self) -> None:
        """Shutdown the extension (for compatibility with core ExtensionManager).

        TTS extensions use their own unloading mechanism via unload().

        Returns:
            None
        """
        if self.is_loaded:
            self.unload()

    # ==========================================================================
    # SERVICE EXTENSION METHODS (for subprocess communication)
    # These are implemented as no-ops for in-process TTS extensions.
    # For subprocess-based TTS extensions, override these methods.
    # ==========================================================================

    async def start_service(self) -> None:
        """Start the TTS service (no-op for in-process extensions).

        Subprocess-based extensions should override this to start their server.
        """
        self.port = None
        self.base_url = None

    async def send_request(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Send request to TTS service (no-op for in-process extensions).

        Subprocess-based extensions should override this to communicate with their server.
        """
        raise NotImplementedError("Use generate_audio() directly for in-process TTS extensions")

    async def stop_service(self) -> None:
        """Stop the TTS service (no-op for in-process extensions).

        Subprocess-based extensions should override this to stop their server.
        """
        pass

    # ==========================================================================
    # TTS EXTENSION INTERFACE METHODS
    # ==========================================================================

    def _create_model_wrapper(self, model: TTSModel) -> BaseModelWrapper:
        """Create a model wrapper for the given model.

        Args:
            model: TTS model instance

        Returns:
            Model wrapper instance
        """
        return BaseModelWrapper(model)

    def _get_model_config(self) -> dict[str, Any]:
        """Get model configuration from extension config.

        Returns:
            Model configuration dictionary
        """
        # This should be overridden by subclasses to extract
        # model-specific configuration from extension parameters
        return self._model_config

    def _convert_voices(self, model_voices: list[dict[str, Any]]) -> list[VoiceInfo]:
        """Convert model voice format to VoiceInfo objects.

        Args:
            model_voices: List of voice dictionaries from model

        Returns:
            List of VoiceInfo objects
        """
        voices = []
        for voice in model_voices:
            voices.append(
                VoiceInfo(
                    id=voice.get("id", "unknown"),
                    name=voice.get("name", "Unknown Voice"),
                    language=voice.get("language", "en-US"),
                    gender=voice.get("gender", "neutral"),
                    description=voice.get("description", ""),
                )
            )
        return voices

    def _map_voice(self, voice_id: str) -> str:
        """Map extension voice ID to model voice name.

        Args:
            voice_id: Extension voice ID

        Returns:
            Model voice name
        """
        # Default implementation - can be overridden for custom mapping
        return voice_id

    def _audio_to_bytes(
        self, audio_data: Union[np.ndarray, "torch.Tensor", bytes], sample_rate: int
    ) -> bytes:
        """Convert audio data to bytes.

        Args:
            audio_data: Audio data as numpy array, torch tensor, or bytes
            sample_rate: Sample rate in Hz

        Returns:
            Audio data as bytes
        """
        if isinstance(audio_data, bytes):
            # If audio data is already in bytes format, return as is
            return audio_data
        elif TORCH_AVAILABLE and isinstance(audio_data, torch.Tensor):
            return AudioProcessor.torch_to_wav_bytes(audio_data, sample_rate)
        else:
            return AudioProcessor.numpy_to_wav_bytes(audio_data, sample_rate)

    def _validate_voice(self, voice_id: str) -> bool:
        """Validate that a voice exists.

        Args:
            voice_id: Voice ID to validate

        Returns:
            True if voice exists, False otherwise
        """
        return any(voice.id == voice_id for voice in self.voices)

    def load(self) -> bool:
        """Load the extension and initialize the model.

        Returns:
            True if successful
        """
        try:
            if self.model_wrapper is None:
                logger.error(f"No model wrapper set for {self.name} extension")
                return False

            # Get model configuration
            model_config = self._get_model_config()

            # Initialize the model
            if self.model_wrapper.initialize(model_config):
                # Get voices from model
                model_voices = self.model_wrapper.get_voices()
                self.voices = self._convert_voices(model_voices)
                self.is_loaded = True
                logger.info(f"{self.name} extension loaded successfully")
                return True
            else:
                logger.error(f"Failed to initialize model for {self.name} extension")
                return False

        except Exception as e:
            logger.error(f"Failed to load {self.name} extension: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback to stderr
            return False

    def unload(self) -> bool:
        """Unload the extension and clean up resources.

        Returns:
            True if successful
        """
        try:
            if self.model_wrapper:
                self.model_wrapper.cleanup()

            self.is_loaded = False
            logger.info(f"{self.name} extension unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unload {self.name} extension: {str(e)}")
            return False

    def get_voices(self) -> list[VoiceInfo]:
        """Get available voices.

        Returns:
            List of available voices
        """
        # If voices are already loaded, return them
        if self.voices:
            return self.voices

        # Otherwise, try to get voices from the model wrapper
        # This allows extensions to provide voices without loading the full model
        if self.model_wrapper:
            try:
                model_voices = self.model_wrapper.get_voices()
                self.voices = self._convert_voices(model_voices)
                return self.voices
            except Exception as e:
                logger.debug(f"Could not get voices from model wrapper: {e}")

        return self.voices

    def generate_audio(
        self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 1.0, **kwargs
    ) -> tuple[bytes, float]:
        """Generate audio using the model.

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            speed: Speech speed multiplier
            pitch: Pitch adjustment
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (audio_bytes, duration_seconds)
        """
        if not self.is_loaded or not self.model_wrapper:
            raise RuntimeError(f"Extension {self.name} is not loaded")

        # For voice cloning extensions (index_tts, vibevoice, gpt_sovits), skip voice validation entirely
        # They use ref_audio_path instead of voice_id
        # Check if this extension has ref_audio_path parameter (voice cloning extensions)
        try:
            extension_config = self.get_config()
            parameters = getattr(extension_config, 'parameters', None) or []
            has_ref_audio_param = any(p.name == 'ref_audio_path' for p in parameters)
        except (AttributeError, TypeError):
            # If get_config() doesn't return an ExtensionConfig, check kwargs instead
            has_ref_audio_param = 'ref_audio_path' in kwargs

        if has_ref_audio_param:
            # These extensions don't use voice_id at all - skip it completely
            voice_id = None

        # Prepare model parameters FIRST to check if 'voice' is in kwargs
        model_params = self._prepare_model_params(voice_id, speed, pitch, **kwargs)

        # Only validate voice_id if we're going to use it (not when 'voice' is in params)
        if 'voice' not in model_params:
            # Validate voice exists only when using voice_id
            if voice_id and not self._validate_voice(voice_id):
                raise ValueError(f"Voice '{voice_id}' not found in {self.name} extension")

        # Map voice ID to model voice name
        model_voice = self._map_voice(voice_id)

        try:
            # Generate audio using model
            # Remove 'voice' from model_params if it exists to avoid conflict
            # since voice is passed as a separate parameter
            params = model_params.copy()
            if 'voice' in params:
                # If 'voice' exists in model_params, give preference to it over model_voice
                voice_to_use = params.pop('voice')
            else:
                # Otherwise use the mapped model_voice
                voice_to_use = model_voice

            audio_data, sample_rate = self.model_wrapper.synthesize(
                text, voice_to_use, **params
            )

            # Convert to bytes
            audio_bytes = self._audio_to_bytes(audio_data, sample_rate)

            # Calculate duration
            duration = AudioProcessor.get_wav_duration(audio_bytes)

            return audio_bytes, duration

        except Exception as e:
            logger.error(f"Error generating audio with {self.name}: {str(e)}")
            raise RuntimeError(f"Audio generation failed: {str(e)}") from e

    def _prepare_model_params(
        self, voice_id: str, speed: float, pitch: float, **kwargs
    ) -> dict[str, Any]:
        """Prepare parameters for the model.

        Args:
            voice_id: Voice ID
            speed: Speech speed
            pitch: Pitch adjustment
            **kwargs: Additional parameters

        Returns:
            Model parameters dictionary
        """
        # Default implementation - can be overridden for custom parameter handling
        params = {"speed": speed, "pitch": pitch, **kwargs}
        return params

    def get_supported_formats(self) -> list[str]:
        """Get supported audio formats.

        Returns:
            List of supported formats
        """
        # Check if subclass defines supported_formats as a class attribute
        if hasattr(self.__class__, 'supported_formats'):
            return self.__class__.supported_formats
        # Default implementation - can be overridden
        return ["wav", "mp3", "ogg"]

    def get_config(self) -> dict[str, Any]:
        """Get extension configuration.

        Returns the core config from ServiceExtension. This includes
        parameters from config.yaml that can be used for UI generation
        and runtime configuration.

        Returns:
            Extension configuration dictionary
        """
        # Return the core config with extension parameters
        return self.config.get('extension', {})

    def get_tts_config(self) -> dict[str, Any]:
        """Get TTS-specific configuration (required by TTSExtensionInterface).

        Returns configuration including dynamic parameters for the TTS extension.
        This is used by the core ExtensionManager to provide TTS capabilities
        to the frontend and other components.

        Returns:
            Extension configuration dictionary with parameters
        """
        # Build TTS config from core config + runtime state
        tts_config = {
            'name': self.name,
            'category': 'tts',
            'is_loaded': self.is_loaded,
            'voices': len(self.voices),
            'parameters': self.config.get('parameters', []),
        }
        return tts_config


class SimpleModelExtension(BaseTTSExtension):
    """Simplified extension for models with basic functionality.

    This class further reduces boilerplate for simple models that
    don't need complex parameter handling or voice mapping.

    Note: This class is deprecated. Extensions should now receive config
    and root_path through the core extension system and create their
    model wrapper in the initialize() method or provide it via set_model_wrapper().
    """

    def __init__(
        self, config: dict[str, Any], root_path: Path,
        model_class: type = None, **model_kwargs
    ):
        """Initialize the simple model extension.

        Args:
            config: Extension configuration dictionary
            root_path: Path to the extension directory
            model_class: TTS model class (optional, can be set later)
            **model_kwargs: Model initialization parameters
        """
        super().__init__(config, root_path)

        # Create model instance if class provided
        if model_class is not None:
            model = model_class(**model_kwargs)
            self.model_wrapper = self._create_model_wrapper(model)

    def set_model_wrapper(self, model_wrapper: BaseModelWrapper) -> None:
        """Set the model wrapper externally.

        This allows extensions to create the model wrapper after initialization.

        Args:
            model_wrapper: The model wrapper to use
        """
        self.model_wrapper = model_wrapper

    def _get_model_config(self) -> dict[str, Any]:
        """Get model configuration from extension parameters.

        Returns:
            Model configuration dictionary
        """
        # Return the stored model config
        return self._model_config

    def set_model_config(self, config: dict[str, Any]) -> None:
        """Set model configuration externally.

        Args:
            config: Model configuration dictionary
        """
        self._model_config = config

    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._model_config[key] = value

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._model_config.get(key, default)
