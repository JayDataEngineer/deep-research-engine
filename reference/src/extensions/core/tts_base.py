"""
Unified TTS Base Class

Provides a common base for TTS subprocess extensions to reduce code duplication.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import ServiceExtension, ExtensionStatus, TTSExtensionInterface


class SubprocessServiceExtension(ServiceExtension):
    """
    Base class for subprocess-only service extensions.

    This class provides a unified pattern for extensions that run in isolated
    subprocesses (like TTS engines). It handles the common pattern of:
    - Placeholder registration (not loaded until activated)
    - Process lifecycle management
    - Shared process manager singleton
    """

    # Shared process manager singleton (class variable, override in subclasses)
    _shared_process_manager = None
    _process_manager_class = None

    def __init__(self, config: Dict[str, Any], root_path: Path):
        """
        Initialize the subprocess service extension.

        Args:
            config: Extension configuration
            root_path: Extension root directory
        """
        super().__init__(config, root_path)
        self._is_subprocess = True
        # Mark as REGISTERED initially - will be loaded on demand
        self._status = ExtensionStatus.REGISTERED

    @classmethod
    def _get_process_manager(cls):
        """
        Get or create the shared process manager singleton.

        Subclasses should override _process_manager_class with their
        specific process manager class.
        """
        if cls._shared_process_manager is None:
            if cls._process_manager_class is None:
                raise NotImplementedError(
                    f"{cls.__name__} must define _process_manager_class"
                )
            cls._shared_process_manager = cls._process_manager_class()
        return cls._shared_process_manager

    @property
    def status(self) -> ExtensionStatus:
        """Return current status."""
        return self._status

    @status.setter
    def status(self, value: ExtensionStatus):
        """Allow status updates."""
        self._status = value

    async def initialize(self, context) -> None:
        """
        Initialize is a no-op for subprocess extensions.

        The actual service starts when start_service() is called.

        Args:
            context: Extension context
        """
        self.context = context
        # Still marked as REGISTERED - not loaded until activate is called
        self._status = ExtensionStatus.REGISTERED

    async def shutdown(self) -> None:
        """Shutdown is a no-op for subprocess extensions."""
        pass

    async def start_service(self) -> None:
        """
        Start the actual service subprocess.

        Subclasses can override this for custom startup behavior.

        Raises:
            Exception: If the service fails to start
        """
        manager = self._get_process_manager()
        result = await manager.start_extension(self.name)

        # Check if the extension actually started successfully
        if result.get("status") == "error":
            raise Exception(f"Failed to start extension: {result.get('message')}")

        self._status = ExtensionStatus.LOADED

    async def send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request is handled by the subprocess.

        Subclasses should override this to communicate with their
        specific service endpoint.

        Args:
            endpoint: API endpoint path
            data: Request data

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement send_request() "
            "or use the extension process API directly"
        )

    async def stop_service(self) -> None:
        """Stop the service subprocess."""
        manager = self._get_process_manager()
        if manager.active_extension == self.name:
            await manager.stop_current()
        self._status = ExtensionStatus.REGISTERED

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return basic capabilities for discovery.

        Subclasses should override to provide specific capabilities.

        Returns:
            Dict with basic capability information
        """
        return {
            "name": self.name,
            "category": self.category,
            "subprocess_only": True,
            "supports_activation": True,
        }


class TTSExtensionBase(SubprocessServiceExtension, TTSExtensionInterface):
    """
    Unified base class for TTS extensions.

    This class combines the subprocess service pattern with the TTS interface
    to provide a consistent API for all TTS extensions.

    TTS extensions should:
    1. Set _process_manager_class to their process manager
    2. Implement get_voices() to return available voices
    3. Implement get_tts_config() to return TTS configuration
    4. Optionally override send_request() for custom API communication
    """

    def get_voices(self) -> List[Dict[str, Any]]:
        """
        Get available voices from this TTS extension.

        Reads voices from config.yaml by default.

        Returns:
            List of voice configuration dictionaries
        """
        return self.config.get('voices', [])

    def get_tts_config(self) -> Dict[str, Any]:
        """
        Get TTS configuration for this extension.

        Returns relevant config fields for the UI/API.

        Returns:
            TTS configuration dictionary
        """
        return {
            "name": self.name,
            "display_name": self.config.get('extension', {}).get('display_name', self.name),
            "category": self.category,
            "voices": self.get_voices(),
            "parameters": self.config.get('parameters', []),
            "models": self.config.get('models', []),
            "enabled": self.config.get('extension', {}).get('enabled', True),
        }

    async def generate_audio(
        self,
        text: str,
        voice: str,
        **kwargs
    ) -> Any:
        """
        Generate audio from text.

        Default implementation uses send_request(). Subclasses
        can override for custom behavior.

        Args:
            text: Input text to synthesize
            voice: Voice ID to use
            **kwargs: Additional generation parameters (speed, pitch, emotion, etc.)

        Returns:
            Generated audio data
        """
        return await self.send_request(
            "generate",
            {
                "text": text,
                "voice": voice,
                **kwargs
            }
        )

    def get_ui_config(self) -> Dict[str, Any]:
        """Return UI config for the extension."""
        return self.config.get('ui', {})

    async def send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to the TTS service subprocess.

        Default implementation raises NotImplementedError.
        Subclasses should override with actual process communication.

        Args:
            endpoint: API endpoint (e.g., "generate", "voices")
            data: Request data

        Returns:
            Response from the TTS service

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement send_request() "
            "to communicate with the TTS subprocess"
        )


def create_tts_extension_class(
    name: str,
    process_manager_class: type,
    voices: Optional[List[Dict[str, Any]]] = None,
    parameters: Optional[List[Dict[str, Any]]] = None,
) -> type:
    """
    Factory function to create a TTS extension class dynamically.

    This is useful for creating TTS extensions without writing a full class.

    Args:
        name: Extension name
        process_manager_class: The process manager class to use
        voices: Optional list of voice configurations
        parameters: Optional list of UI parameters

    Returns:
        A TTS extension class ready to use

    Example:
        ```python
        IndexTTSExtension = create_tts_extension_class(
            "index_tts_vllm",
            IndexTTSProcessManager,
            voices=[
                {"id": "male", "name": "Male Voice", "gender": "male"},
                {"id": "female", "name": "Female Voice", "gender": "female"},
            ]
        )
        ```
    """
    class DynamicTTSExtension(TTSExtensionBase):
        _process_manager_class = process_manager_class

        def __init__(self, config: Dict[str, Any], root_path: Path):
            super().__init__(config, root_path)

            # Add voices/parameters from factory if not in config
            if voices and 'voices' not in config:
                config['voices'] = voices
            if parameters and 'parameters' not in config:
                config['parameters'] = parameters

        def get_voices(self) -> List[Dict[str, Any]]:
            """Return voices from config or factory defaults."""
            config_voices = self.config.get('voices', [])
            return config_voices if config_voices else (voices or [])

    DynamicTTSExtension.__name__ = name
    return DynamicTTSExtension
