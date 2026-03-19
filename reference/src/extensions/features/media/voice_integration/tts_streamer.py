"""TTS Streamer - Handles text-to-speech generation and streaming.

Provides async TTS generation with chunked streaming for real-time
audio delivery in voice chat conversations.
"""

import base64
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from src.constants import DATA_OUTPUTS_DIR

logger = logging.getLogger(__name__)


class TTSStreamer:
    """Handles TTS generation and streaming for chat responses.

    This class provides async TTS generation with support for:
    - Chunked audio streaming for real-time playback
    - Multiple TTS engines (Kokoro, IndexTTS, etc.)
    - Voice cloning via reference audio
    - Audio file persistence with URL generation
    """

    def __init__(self, chunk_size: int = 8192):
        """Initialize the TTS streamer.

        Args:
            chunk_size: Size of audio chunks for streaming (default: 8KB)
        """
        self.chunk_size = chunk_size
        self._tts_service = None
        logger.info("TTSStreamer initialized")

    async def initialize(self, context):
        """Initialize the TTS streamer with application context.

        Args:
            context: Application context containing services
        """
        try:
            from src.extensions.libraries.tts_engine.router import get_tts_service
            self._tts_service = get_tts_service()
            logger.info("TTSStreamer: TTS service loaded successfully")
        except ImportError as e:
            logger.warning(f"TTSStreamer: Failed to import TTS service: {e}")
            self._tts_service = None

    async def is_available(self) -> bool:
        """Check if TTS service is available.

        Returns:
            True if TTS service is loaded and ready
        """
        return self._tts_service is not None

    async def stream_tts(
        self,
        text: str,
        voice_id: Optional[str] = None,
        extension_name: str = "kokoro",
        speed: float = 1.0,
        pitch: float = 1.0,
        format: str = "wav",
        ref_audio_path: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Stream TTS generation as audio chunks.

        Args:
            text: Text to convert to speech
            voice_id: Voice model to use (for engines that require it)
            extension_name: TTS extension/engine to use
            speed: Speech speed multiplier
            pitch: Pitch adjustment
            format: Output audio format (wav, mp3, ogg)
            ref_audio_path: Path to reference audio for voice cloning

        Yields:
            Events containing either:
            - {"type": "chunk", "data": base64_encoded_audio}
            - {"type": "complete", "url": audio_url, "duration": seconds}
            - {"type": "error", "error": error_message}
        """
        if not self._tts_service:
            yield {"type": "error", "error": "TTS service not available"}
            return

        if not text or not text.strip():
            yield {"type": "error", "error": "No text provided for TTS"}
            return

        try:
            # Build kwargs for TTS generation
            kwargs = {
                "text": text,
                "voice_id": voice_id or "default",
                "speed": speed,
                "pitch": pitch,
                "audio_format": format,
            }

            # Add reference audio for voice cloning if provided
            if ref_audio_path:
                kwargs["ref_audio_path"] = ref_audio_path

            logger.info(f"TTSStreamer: Generating audio for text: '{text[:50]}...'")

            # Generate audio
            audio_buffer, duration = await self._tts_service.generate_audio(**kwargs)

            # Save audio to file and generate URL
            audio_url = await self._save_audio(audio_buffer, format)

            # Read audio data and chunk it
            audio_buffer.seek(0)
            audio_data = audio_buffer.read()

            # Stream audio in chunks
            total_chunks = (len(audio_data) + self.chunk_size - 1) // self.chunk_size
            for i in range(0, len(audio_data), self.chunk_size):
                chunk = audio_data[i:i + self.chunk_size]
                yield {
                    "type": "chunk",
                    "data": base64.b64encode(chunk).decode("utf-8"),
                    "format": format,
                    "chunk_index": i // self.chunk_size,
                    "total_chunks": total_chunks,
                }

            # Yield completion event with URL
            yield {
                "type": "complete",
                "url": audio_url,
                "duration": duration,
                "format": format,
            }

            logger.info(f"TTSStreamer: Audio generation complete, duration: {duration:.2f}s")

        except ValueError as e:
            logger.error(f"TTSStreamer: Validation error: {e}")
            yield {"type": "error", "error": str(e)}
        except RuntimeError as e:
            logger.error(f"TTSStreamer: Generation error: {e}")
            yield {"type": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"TTSStreamer: Unexpected error: {e}", exc_info=True)
            yield {"type": "error", "error": f"TTS generation failed: {str(e)}"}

    async def _save_audio(self, audio_buffer: io.BytesIO, format: str) -> str:
        """Save audio buffer to file and return URL.

        Args:
            audio_buffer: BytesIO containing audio data
            format: Audio format (wav, mp3, ogg)

        Returns:
            URL path to access the audio file
        """
        try:
            # Create output directory structure
            output_dir = DATA_OUTPUTS_DIR / "audio"
            date_dir = output_dir / datetime.now().strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            timestamp = int(time.time() * 1000)
            filename = f"audio_{timestamp}.{format}"
            filepath = date_dir / filename

            # Write audio file
            with open(filepath, "wb") as f:
                audio_buffer.seek(0)
                f.write(audio_buffer.read())

            # Return URL path
            url = f"/{filepath}"
            logger.info(f"TTSStreamer: Saved audio to {url}")
            return url

        except Exception as e:
            logger.error(f"TTSStreamer: Failed to save audio: {e}")
            # Return fallback URL even if save fails
            return f"/data/outputs/audio/fallback.{format}"

    async def close(self):
        """Cleanup resources."""
        self._tts_service = None
        logger.info("TTSStreamer closed")
