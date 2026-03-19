"""
TTS (Text-to-Speech) service for handling audio generation.

Primary: Uses LiteLLM shared infrastructure (OpenAI-compatible API)
Fallback: Local TTS extensions via subprocess
"""

import io
import logging
from typing import Optional

from src.extensions.libraries.tts_engine.api.schemas.tts import VoiceInfo
from src.constants import TTS_EXTENSION_URL, TTS_VIA_LITELLM
from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager

logger = logging.getLogger(__name__)


class TTSService:
    """Service for handling text-to-speech operations.
    
    Primary backend: LiteLLM shared infrastructure (http://litellm.local/v1)
    Fallback backend: Local TTS extensions via subprocess
    """

    def __init__(self, use_litellm: Optional[bool] = None):
        """Initialize the TTS service.
        
        Args:
            use_litellm: Use LiteLLM shared infrastructure (default: from TTS_VIA_LITELLM constant)
        """
        self.use_litellm = use_litellm if use_litellm is not None else TTS_VIA_LITELLM
        self.available_voices = []
        self._cached_extension: str | None = None  # Track which extension voices are cached for
        self._litellm_client = None
        
        # Initialize LiteLLM client if enabled
        if self.use_litellm:
            try:
                from src.shared.litellm_tts import LiteLLMTTSClient
                self._litellm_client = LiteLLMTTSClient()
                logger.info("TTS Service initialized (LiteLLM shared infrastructure)")
            except Exception as e:
                logger.warning(f"Failed to initialize LiteLLM TTS client: {e}. Falling back to local extensions.")
                self.use_litellm = False
        
        if not self.use_litellm:
            logger.info("TTS Service initialized (local extension mode)")

    async def refresh_voices(self):
        """Refresh voices from LiteLLM or subprocess extension."""
        if self.use_litellm and self._litellm_client:
            # Get voices from LiteLLM
            try:
                voices = self._litellm_client.get_voices()
                self.available_voices = [
                    VoiceInfo(
                        id=voice,
                        name=voice,
                        language="en-US",
                        gender="unknown",
                        description=f"Voice: {voice}",
                    )
                    for voice in voices
                ]
                logger.info(f"LiteLLM TTS voices: {len(self.available_voices)} available")
                return
            except Exception as e:
                logger.warning(f"Failed to get voices from LiteLLM: {e}")
        
        # Fallback to local extension
        current_extension = extension_process_manager.active_extension
        if current_extension == self._cached_extension and self.available_voices:
            logger.debug(f"Using cached voices for {current_extension} ({len(self.available_voices)} voices)")
            return

        self.available_voices = []
        self._cached_extension = current_extension

        if current_extension:
            try:
                response = await extension_process_manager.client.get(
                    f"{TTS_EXTENSION_URL}/voices",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    voices_list = data.get("voices", [])
                    self.available_voices = [
                        VoiceInfo(
                            id=v.get("id", ""),
                            name=v.get("name", ""),
                            language=v.get("language", "en-US"),
                            gender=v.get("gender", "unknown"),
                            description=v.get("description", ""),
                        )
                        for v in voices_list
                    ]
                    logger.info(f"Cached voices for {current_extension}: {len(self.available_voices)} available")
            except Exception as e:
                logger.warning(f"Failed to get voices from subprocess: {e}")
        else:
            logger.debug("No extension active, skipping voice refresh")

    def get_available_voices(self) -> list[VoiceInfo]:
        """Get list of available voices."""
        return self.available_voices

    def get_voice_by_id(self, voice_id: str) -> VoiceInfo | None:
        """Get voice information by ID."""
        for voice in self.available_voices:
            if voice.id == voice_id:
                return voice
        return None

    async def generate_audio(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
        pitch: float = 1.0,
        audio_format: str = "wav",
        model: Optional[str] = None,
        **kwargs
    ) -> tuple[io.BytesIO, float]:
        """Generate audio from text.
        
        Primary: Uses LiteLLM shared infrastructure (OpenAI-compatible API)
        Fallback: Local TTS extensions via subprocess

        Args:
            text: Text to convert to speech
            voice_id: Voice model to use
            speed: Speech speed multiplier
            pitch: Pitch adjustment (not used by LiteLLM)
            audio_format: Output audio format (wav, mp3, opus)
            model: TTS model name (default: from config)
            **kwargs: Additional parameters (emotion, etc.)

        Returns:
            Tuple of (audio_data_bytes, duration_seconds)

        Raises:
            RuntimeError: If generation fails
        """
        logger.info(
            f"Generating audio for text: '{text[:50]}...' with voice: {voice_id}"
        )

        # Try LiteLLM first (shared infrastructure)
        if self.use_litellm and self._litellm_client:
            try:
                logger.debug("Using LiteLLM TTS (shared infrastructure)")
                
                # Generate speech via LiteLLM
                audio_bytes = await self._litellm_client.generate_speech(
                    text=text,
                    model=model,
                    voice=voice_id,
                    speed=speed,
                    format=audio_format,
                    **kwargs
                )
                
                # Calculate duration (simple estimation for WAV)
                # For more accurate duration, could use librosa or wave module
                duration = len(audio_bytes) / (44100 * 2)  # Rough estimate for 16-bit audio
                
                logger.info(
                    f"Audio generated via LiteLLM, duration: {duration:.2f}s, {len(audio_bytes)} bytes"
                )
                return io.BytesIO(audio_bytes), duration
                
            except Exception as e:
                logger.warning(f"LiteLLM TTS failed: {e}. Falling back to local extension...")
                # Fall through to local extension

        # Fallback to local TTS extension
        logger.debug("Using local TTS extension (fallback)")
        
        # Validate voice_id only for extensions that use traditional voice selection
        voice_extensions_requiring_id = ["kokoro", "espeak_ng", "maya1"]
        current_extension = extension_process_manager.active_extension

        if current_extension in voice_extensions_requiring_id:
            if not voice_id or not isinstance(voice_id, str) or voice_id.strip() == "" or voice_id.strip() == "None":
                raise ValueError(f"Invalid voice ID: {voice_id}")

        # Check if subprocess extension is active
        if not extension_process_manager.active_extension:
            raise RuntimeError(
                "No extension is loaded and LiteLLM TTS is unavailable. "
                "Please load an extension via /api/v1/extension-process/load or start shared infrastructure."
            )

        # Refresh voices to ensure we have the latest
        await self.refresh_voices()

        # Use the subprocess extension to generate audio
        try:
            audio_bytes, duration = await extension_process_manager.generate_audio(
                text=text,
                voice_id=voice_id,
                speed=speed,
                pitch=pitch,
                output_format=audio_format,
                **kwargs
            )
            audio_buffer = io.BytesIO(audio_bytes)
            logger.info(
                f"Audio generated successfully via {extension_process_manager.active_extension}, duration: {duration:.2f}s"
            )
            return audio_buffer, duration

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e


# Global service instance
_tts_service_instance = None


def get_tts_service():
    """Get the TTS service instance, creating it if necessary."""
    global _tts_service_instance
    if _tts_service_instance is None:
        _tts_service_instance = TTSService()
    return _tts_service_instance


# For backward compatibility
tts_service = get_tts_service()
