"""
TTS Extension Factory - DRY solution for TTS extensions.

This module provides a generic factory that reads extension config.yaml files
and auto-generates extension classes, eliminating code duplication across TTS extensions.

All TTS extensions now use the core ExtensionManager system.
"""

import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from src.constants import ROOT_MODELS_DIR
from src.extensions.core.base import ServiceExtension
from src.extensions.libraries.tts_engine.api.schemas.tts import ExtensionConfig
from src.extensions.libraries.tts_engine.model_interface import BaseModelWrapper, TTSModel

logger = logging.getLogger(__name__)


# TTS models base directory
TTS_MODELS_DIR = ROOT_MODELS_DIR / "tts"

# OPTIMIZED: Cache resolved engine classes to avoid repeated introspection
_ENGINE_CLASS_CACHE: dict[str, type] = {}


def get_extension_init_config(name: str, config: dict) -> dict:
    """
    Get extension-specific initialization config.

    Each TTS extension has different initialization requirements.
    This function maps the generic config to extension-specific configs.

    Args:
        name: Extension name
        config: Extension config from config.yaml

    Returns:
        Initialization config dict for the specific extension
    """
    # Try to get model path from config.yaml
    model_path = None
    if config.get('models'):
        model_path = config['models'][0].get('local_path')

    # Extension-specific configs with fallback to default paths
    if name == 'kokoro':
        model_dir = model_path or str(TTS_MODELS_DIR / 'kokoro')
        return {
            'model_dir': model_dir,
            'model_file': 'kokoro-v1.0.onnx',
            'voices_file': 'voices-v1.0.bin',
        }
    elif name == 'maya1':
        return {
            'model_path': model_path or str(TTS_MODELS_DIR / 'maya1'),
        }
    elif name == 'vibe_voice':
        # Try vibevoice-1.5b first, fall back to vibevoice-7b
        vibe_path = model_path
        if not vibe_path:
            if (TTS_MODELS_DIR / 'vibevoice-1.5b').exists():
                vibe_path = str(TTS_MODELS_DIR / 'vibevoice-1.5b')
            else:
                vibe_path = str(TTS_MODELS_DIR / 'vibevoice-7b')
        return {
            'model_path': vibe_path,
            'device': config.get('device', 'auto'),
        }
    elif name == 'gpt_sovits':
        return {
            'model_path': model_path or str(TTS_MODELS_DIR / 'gpt_sovits'),
            'device': config.get('device', 'auto'),
        }
    elif name == 'qwen3_tts':
        return {
            'model_path': model_path or str(TTS_MODELS_DIR / 'qwen3_tts'),
            'device': config.get('device', 'auto'),
        }
    elif name == 'index_tts_vllm':
        return {
            'model_path': model_path or str(TTS_MODELS_DIR / 'indextts'),
            'device': config.get('device', 'auto'),
        }
    elif name == 'espeak_ng':
        # espeak-ng doesn't need model files
        return {}
    else:
        # Generic fallback
        return {
            'model_path': model_path,
            'device': config.get('device', 'auto'),
        }


class TTSExtensionFactory:
    """
    Factory for creating TTS extension classes from config.yaml.

    All extensions created by this factory:
    - Extend ServiceExtension from the core system
    - Implement TTSExtensionInterface from the core system
    - Work with the unified ExtensionManager
    """

    @staticmethod
    def create_extension_class(
        name: str,
        engine_module_name: str,
        version: str = "2.0.0",
        config_getter: str = None,
        extra_paths: list[str] = None,
    ) -> type:
        """
        Create a TTS extension class dynamically using the core system.

        Args:
            name: Extension name (e.g., "kokoro", "maya1")
            engine_module_name: Name of the engine module to import (e.g., "kokoro.engine")
            version: Extension version
            config_getter: Optional name of config getter function
            extra_paths: Optional list of extra paths to add to sys.path before importing

        Returns:
            A dynamically created extension class that extends ServiceExtension and TTSExtensionInterface
        """

        def _init(self, config: dict[str, Any], root_path: Path):
            """Initialize the extension with core constructor signature."""
            # Call ServiceExtension.__init__ with config and root_path
            ServiceExtension.__init__(self, config, root_path)

            # Store extension state
            self.name = name
            self.model_wrapper: BaseModelWrapper | None = None
            self.voices = []
            self.is_loaded = False

            # Add extra paths to sys.path if provided
            if extra_paths:
                for path in extra_paths:
                    if path not in sys.path:
                        sys.path.insert(0, path)

            # Import and create engine
            engine_module = importlib.import_module(engine_module_name)

            # OPTIMIZED: Check cache first for resolved engine class
            cache_key = f"{engine_module_name}:{name}"
            if cache_key in _ENGINE_CLASS_CACHE:
                engine_class = _ENGINE_CLASS_CACHE[cache_key]
                logger.info(f"[{name}] Using cached engine class: {engine_class.__name__}")
            else:
                # Try different naming conventions for the engine class
                engine_class = None

                # Try: {Name}Engine (e.g., KokoroEngine)
                cap_name = name.replace("_", " ").title().replace(" ", "")
                engine_class = getattr(engine_module, f"{cap_name}Engine", None)

                # Special case: if name ends with _tts, try with TTSEngine suffix (e.g., Qwen3TTSEngine)
                if engine_class is None and name.endswith("_tts"):
                    tts_cap_name = name[:-4].replace("_", " ").title().replace(" ", "") + "TTS"
                    engine_class = getattr(engine_module, f"{tts_cap_name}Engine", None)

                # Try: {Name}TTSModel (e.g., KokoroTTSModel, Maya1TTSModel)
                if engine_class is None:
                    engine_class = getattr(engine_module, f"{cap_name}TTSModel", None)

                # Try: {Name}Model (without TTS prefix, e.g., VibeVoiceModel, GPTSoVITSModel)
                if engine_class is None:
                    engine_class = getattr(engine_module, f"{cap_name}Model", None)

                # Try: smart acronym handling
                if engine_class is None:
                    parts = name.split("_")
                    smart_parts = []
                    for part in parts:
                        if part.lower() == 'gpt':
                            smart_parts.append('GPT')
                        elif part.lower() == 'tts':
                            smart_parts.append('TTS')
                        elif 'vits' in part.lower():
                            if part.lower().startswith('sovits'):
                                smart_parts.append('SoVITS')
                            elif part.lower().endswith('sovits'):
                                smart_parts.append('SoVITS')
                            else:
                                p = part.title().replace('Tts', 'TTS').replace('Vits', 'VITS')
                                smart_parts.append(p)
                        else:
                            smart_parts.append(part.capitalize())
                    smart_name = "".join(smart_parts)
                    engine_class = getattr(engine_module, f"{smart_name}Model", None)

                # Try: all caps with underscore
                if engine_class is None:
                    all_caps_underscore = name.upper()
                    engine_class = getattr(engine_module, f"{all_caps_underscore}Model", None)

                # Try: Refactored{Name}Extension
                if engine_class is None:
                    engine_class = getattr(engine_module, f"Refactored{cap_name}Extension", None)

                # Try: Model (generic fallback)
                if engine_class is None:
                    engine_class = getattr(engine_module, "Model", None)

                if engine_class is None:
                    available_classes = []
                    for attr in dir(engine_module):
                        if not attr.startswith("_") and attr[0].isupper():
                            obj = getattr(engine_module, attr)
                            if isinstance(obj, type) and obj is not TTSModel:
                                available_classes.append(attr)

                    raise AttributeError(
                        f"No engine class found in {engine_module_name}. "
                        f"Tried: {cap_name}Engine, {cap_name}TTSModel, {cap_name}Model, "
                        f"{smart_name}Model, {all_caps_underscore}Model, Refactored{cap_name}Extension, Model. "
                        f"Available classes: {available_classes[:10]}"
                    )

                # Cache the resolved class for future use
                _ENGINE_CLASS_CACHE[cache_key] = engine_class
                logger.info(f"[{name}] Cached engine class: {engine_class.__name__}")

            engine = engine_class()
            self.model_wrapper = BaseModelWrapper(engine)

            # Initialize the model with extension-specific config
            init_config = get_extension_init_config(name, config)
            logger.info(f"[{name}] Initializing model with config: {init_config}")
            self.model_wrapper.initialize(init_config)

        def _load(self) -> bool:
            """Load the extension model (for subprocess compatibility).

            The extension_service in subprocess mode expects a load() method.
            Since the model is already loaded in __init__, this just returns True.
            """
            # Model is already loaded in __init__ via BaseModelWrapper
            if self.model_wrapper and self.model_wrapper.is_initialized:
                self.is_loaded = True
                return True
            return False

        def _unload(self) -> bool:
            """Unload the extension model."""
            if self.is_loaded and self.model_wrapper:
                self.model_wrapper.cleanup()
            self.is_loaded = False
            return True

        def _initialize(self, context: Any = None) -> None:
            """Initialize the extension (core interface)."""
            self.context = context
            # Model loading is done in __init__, nothing to do here

        def _shutdown(self) -> None:
            """Shutdown the extension."""
            if self.is_loaded and self.model_wrapper:
                self.model_wrapper.cleanup()
            self.is_loaded = False

        def _start_service(self) -> None:
            """Start service (no-op for in-process extensions)."""
            self.port = None
            self.base_url = None

        def _send_request(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
            """Send request (not used for in-process extensions)."""
            raise NotImplementedError("Use generate_audio() directly")

        def _stop_service(self) -> None:
            """Stop service (no-op for in-process extensions)."""
            pass

        def _get_voices(self) -> list:
            """Get available voices."""
            if self.model_wrapper:
                try:
                    return self.model_wrapper.get_voices()
                except Exception:
                    return []
            return []

        async def _generate_audio(
            self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 1.0, **kwargs
        ) -> tuple[bytes, float]:
            """Generate audio. Supports both sync and async models."""
            if not self.model_wrapper:
                raise RuntimeError(f"Extension {self.name} has no model wrapper")

            from src.extensions.libraries.tts_engine.audio_utils import AudioProcessor

            # Check if the model's synthesize method is async
            model = self.model_wrapper.model
            if asyncio.iscoroutinefunction(model.synthesize):
                # Async model (like IndexTTS vLLM)
                audio_data, sample_rate = await model.synthesize(text, voice_id, speed=speed, pitch=pitch, **kwargs)
            else:
                # Sync model (most TTS engines)
                audio_data, sample_rate = self.model_wrapper.synthesize(text, voice_id, speed=speed, pitch=pitch, **kwargs)

            # Convert to bytes
            if isinstance(audio_data, bytes):
                audio_bytes = audio_data
            else:
                import numpy as np
                try:
                    import torch
                    if isinstance(audio_data, torch.Tensor):
                        audio_bytes = AudioProcessor.torch_to_wav_bytes(audio_data, sample_rate)
                    else:
                        audio_bytes = AudioProcessor.numpy_to_wav_bytes(audio_data, sample_rate)
                except ImportError:
                    audio_bytes = AudioProcessor.numpy_to_wav_bytes(np.array(audio_data), sample_rate)

            duration = AudioProcessor.get_wav_duration(audio_bytes)
            return audio_bytes, duration

        def _get_supported_formats(self) -> list[str]:
            """Get supported formats."""
            return ["wav", "mp3", "ogg"]

        def _get_tts_config(self) -> dict:
            """Get TTS configuration."""
            return {
                'name': self.name,
                'category': 'tts',
                'is_loaded': self.is_loaded,
                'voices': len(self._get_voices()),
                'parameters': self.config.get('parameters', []),
            }

        def _validate_parameters(self, **kwargs) -> bool:
            """Validate parameters."""
            return True

        # Create the class dynamically
        parts = name.split('_')
        formatted_parts = []
        for part in parts:
            if part.lower() == 'ng':
                formatted_parts.append('NG')
            elif part.lower() == 'tts':
                formatted_parts.append('TTS')
            else:
                formatted_parts.append(part.capitalize())
        cls_name = f"{''.join(formatted_parts)}Extension"

        # Create class that extends both ServiceExtension and implements TTSExtensionInterface
        cls_dict = {
            "__init__": _init,
            "initialize": _initialize,
            "shutdown": _shutdown,
            "start_service": _start_service,
            "send_request": _send_request,
            "stop_service": _stop_service,
            "get_voices": _get_voices,
            "generate_audio": _generate_audio,
            "get_supported_formats": _get_supported_formats,
            "get_tts_config": _get_tts_config,
            "validate_parameters": _validate_parameters,
            "load": _load,  # For subprocess compatibility (extension_service expects load())
            "unload": _unload,  # For subprocess compatibility
            "__module__": f"{name}.extension",
        }

        # Create the class with ServiceExtension as base
        cls = type(cls_name, (ServiceExtension,), cls_dict)
        return cls

    @staticmethod
    def create_from_directory(ext_dir: Path) -> ServiceExtension:
        """
        Create an extension instance from an extension directory.

        Reads config.yaml and auto-generates the extension using the core system.

        Args:
            ext_dir: Path to the extension directory

        Returns:
            An instantiated extension that works with the core ExtensionManager
        """
        from src.constants import TTS_EXTENSIONS_DIR

        config_file = ext_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"No config.yaml found in {ext_dir}")

        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        ext_config = config_data.get("extension", {})
        name = ext_config.get("name")
        if not name:
            raise ValueError(f"No extension name in config.yaml")

        version = ext_config.get("version", "2.0.0")

        # Add TTS_EXTENSIONS_DIR to sys.path before importing
        if str(TTS_EXTENSIONS_DIR) not in sys.path:
            sys.path.insert(0, str(TTS_EXTENSIONS_DIR))

        # Check for extra_paths in config
        extra_paths = []
        models_section = config_data.get("models", [])
        if models_section:
            for model in models_section:
                local_path = model.get("local_path")
                if local_path:
                    extra_paths.append(str(ext_dir / local_path))

        # Add extra paths to sys.path
        for extra_path in extra_paths:
            if extra_path not in sys.path:
                sys.path.insert(0, extra_path)

        # Create the extension class
        engine_module = f"{name}.engine"
        extension_class = TTSExtensionFactory.create_extension_class(
            name=name,
            engine_module_name=engine_module,
            version=version,
            extra_paths=extra_paths,
        )

        # Instantiate with config and root_path (core pattern)
        return extension_class(config_data, ext_dir)


def create_extension_from_config(ext_dir: Path) -> ServiceExtension:
    """
    Convenience function to create an extension from its config.yaml.

    This replaces the need for each extension to have its own extension.py file.

    Usage in extension directory:
        # extension.py
        from src.extensions.libraries.tts_engine.extension_factory import create_extension_from_config
        from pathlib import Path

        def register_extension():
            return create_extension_from_config(Path(__file__).parent)

    Args:
        ext_dir: Path to the extension directory

    Returns:
        An instantiated extension that works with the core ExtensionManager
    """
    return TTSExtensionFactory.create_from_directory(ext_dir)
