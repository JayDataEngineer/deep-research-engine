"""LiteLLM TTS Client - Unified Text-to-Speech via Shared Infrastructure.

This module provides a simple, OpenAI-compatible interface for TTS generation
using the shared LiteLLM proxy infrastructure.

All TTS calls go through LiteLLM which routes to:
- Kokoro (fast, lightweight)
- IndexTTS2 (high quality, emotional)
- Other TTS models managed by the Orchestrator

Usage:
    from .litellm_tts import generate_speech, generate_speech_stream

    # Simple TTS generation
    audio_bytes = await generate_speech(
        text="Hello, this is a test.",
        voice="Cherry",
        model="kokoro"
    )

    # With custom parameters
    audio_bytes = await generate_speech(
        text="Emotional speech",
        model="indextts2",
        voice="female",
        emotion="happy",
        speed=1.0
    )

    # Streaming generation
    async for chunk in generate_speech_stream(text="Streaming test..."):
        process_audio_chunk(chunk)
"""

import io
import wave
from typing import Optional, Literal, AsyncIterator
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

from .logging_config import get_logger
from .litellm_config import (
    LITELLM_TTS_URL,
    LITELLM_API_KEY,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    DEFAULT_TTS_FORMAT,
    DEFAULT_TTS_SPEED,
    AVAILABLE_TTS_MODELS,
)

logger = get_logger(__name__)


# =============================================================================
# TTS Client
# =============================================================================

class LiteLLMTTSClient:
    """Async TTS client using LiteLLM proxy (OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        """Initialize TTS client.

        Args:
            base_url: LiteLLM proxy URL (default: http://litellm.local/v1)
            api_key: API key (default: sk-litellm-master)
            default_model: Default TTS model (default: kokoro)
        """
        self.base_url = base_url or LITELLM_TTS_URL
        self.api_key = api_key or LITELLM_API_KEY
        self.default_model = default_model or DEFAULT_TTS_MODEL

        # Create OpenAI client (LiteLLM is OpenAI-compatible)
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        logger.info(
            f"LiteLLM TTS Client initialized: {self.base_url}, "
            f"model: {self.default_model}"
        )

    async def generate_speech(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        format: Literal["wav", "mp3", "opus"] = DEFAULT_TTS_FORMAT,
        **kwargs,
    ) -> bytes:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            model: TTS model name (default: kokoro)
            voice: Voice ID (default: Cherry for kokoro, female for indextts2)
            speed: Speech speed multiplier (default: 1.0)
            format: Audio format (wav, mp3, opus)
            **kwargs: Additional parameters (emotion, pitch, etc.)

        Returns:
            Audio bytes

        Raises:
            OpenAIError: If TTS generation fails
        """
        model = model or self.default_model
        voice = voice or DEFAULT_TTS_VOICE
        speed = speed or DEFAULT_TTS_SPEED

        logger.debug(
            f"Generating TTS: model={model}, voice={voice}, "
            f"text_length={len(text)}, speed={speed}"
        )

        try:
            # OpenAI-compatible TTS API
            response = await self.client.audio.speech.create(
                model=model,
                input=text,
                voice=voice,
                speed=speed,
                response_format=format,
                extra_body=kwargs,  # Pass additional params like emotion
            )

            # Read audio bytes
            audio_bytes = response.content
            logger.debug(f"TTS generated: {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    async def generate_speech_stream(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        format: Literal["wav", "mp3", "opus"] = DEFAULT_TTS_FORMAT,
        chunk_size: int = 4096,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """Generate speech with streaming response.

        Args:
            text: Text to synthesize
            model: TTS model name
            voice: Voice ID
            speed: Speech speed
            format: Audio format
            chunk_size: Bytes per chunk
            **kwargs: Additional parameters

        Yields:
            Audio chunks
        """
        model = model or self.default_model
        voice = voice or DEFAULT_TTS_VOICE
        speed = speed or DEFAULT_TTS_SPEED

        try:
            response = await self.client.audio.speech.create(
                model=model,
                input=text,
                voice=voice,
                speed=speed,
                response_format=format,
                extra_body=kwargs,
            )

            # Stream the response
            async for chunk in response.iter_bytes(chunk_size=chunk_size):
                yield chunk

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise

    def get_available_models(self) -> dict:
        """Get available TTS models and their capabilities."""
        return AVAILABLE_TTS_MODELS

    def get_voices(self, model: Optional[str] = None) -> list:
        """Get available voices for a model.

        Args:
            model: TTS model name (default: current default model)

        Returns:
            List of voice IDs
        """
        model = model or self.default_model
        if model in AVAILABLE_TTS_MODELS:
            return AVAILABLE_TTS_MODELS[model].get("voices", [])
        return []


# =============================================================================
# Convenience Functions
# =============================================================================

# Global client instance (lazy initialization)
_tts_client: Optional[LiteLLMTTSClient] = None


def get_tts_client() -> LiteLLMTTSClient:
    """Get or create TTS client instance."""
    global _tts_client
    if _tts_client is None:
        _tts_client = LiteLLMTTSClient()
    return _tts_client


async def generate_speech(
    text: str,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    format: Literal["wav", "mp3", "opus"] = DEFAULT_TTS_FORMAT,
    **kwargs,
) -> bytes:
    """Generate speech from text using LiteLLM TTS.

    Args:
        text: Text to synthesize
        model: TTS model (default: kokoro)
        voice: Voice ID
        speed: Speech speed (default: 1.0)
        format: Audio format (wav, mp3, opus)
        **kwargs: Additional parameters (emotion, pitch, etc.)

    Returns:
        Audio bytes

    Example:
        >>> audio = await generate_speech("Hello world", voice="Cherry")
        >>> save_audio(audio, "output.wav")
    """
    client = get_tts_client()
    return await client.generate_speech(
        text=text,
        model=model,
        voice=voice,
        speed=speed,
        format=format,
        **kwargs,
    )


async def generate_speech_stream(
    text: str,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    format: Literal["wav", "mp3", "opus"] = DEFAULT_TTS_FORMAT,
    chunk_size: int = 4096,
    **kwargs,
) -> AsyncIterator[bytes]:
    """Generate speech with streaming.

    Args:
        text: Text to synthesize
        model: TTS model
        voice: Voice ID
        speed: Speech speed
        format: Audio format
        chunk_size: Bytes per chunk
        **kwargs: Additional parameters

    Yields:
        Audio chunks
    """
    client = get_tts_client()
    async for chunk in client.generate_speech_stream(
        text=text,
        model=model,
        voice=voice,
        speed=speed,
        format=format,
        chunk_size=chunk_size,
        **kwargs,
    ):
        yield chunk


def get_available_tts_models() -> dict:
    """Get available TTS models."""
    return get_tts_client().get_available_models()


def get_tts_voices(model: Optional[str] = None) -> list:
    """Get available voices for a model."""
    return get_tts_client().get_voices(model)
