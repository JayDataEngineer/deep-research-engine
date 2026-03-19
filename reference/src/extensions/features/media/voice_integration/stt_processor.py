"""STT Processor - Handles speech-to-text transcription.

Wraps the existing faster-whisper transcription from voice_input.py
for use in voice chat conversations.
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

logger = logging.getLogger(__name__)


# Cache the Whisper model using the existing function
def _get_whisper_model():
    """Get or create cached Whisper model for transcription.

    Uses the same implementation as voice_input.py for consistency.
    """
    try:
        from faster_whisper import WhisperModel
        # Use base model for speed and accuracy balance
        # CPU with INT8 quantization for efficiency
        model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("STTProcessor: Whisper model loaded successfully")
        return model
    except ImportError:
        logger.warning("STTProcessor: faster-whisper not installed")
        return False
    except Exception as e:
        logger.error(f"STTProcessor: Failed to load Whisper model: {e}")
        return False


def _transcribe_audio_file(audio_path: str) -> Optional[str]:
    """Transcribe audio file and return the text.

    Uses the same transcription logic as voice_input.py.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcribed text or None if transcription failed
    """
    model = _get_whisper_model()
    if not model:
        return None

    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language=None,  # Auto-detect language
            vad_filter=True,  # Enable voice activity detection
        )

        # Combine all segments into one text
        full_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        logger.info(f"STTProcessor: Transcribed audio (language: {info.language}, probability: {info.language_probability:.2f})")

        return full_text
    except Exception as e:
        logger.error(f"STTProcessor: Failed to transcribe audio: {e}")
        return None


class STTProcessor:
    """Handles speech-to-text processing for voice chat.

    This class wraps the existing faster-whisper transcription
    functionality from voice_input.py.
    """

    def __init__(self):
        """Initialize the STT processor."""
        self._model = None  # Lazy load on first use
        logger.info("STTProcessor initialized")

    async def initialize(self, context):
        """Initialize the STT processor with application context.

        Args:
            context: Application context
        """
        # Pre-load the model to check availability
        self._model = _get_whisper_model()
        if self._model:
            logger.info("STTProcessor: Whisper model pre-loaded")
        else:
            logger.warning("STTProcessor: Whisper model not available")

    async def is_available(self) -> bool:
        """Check if STT service is available.

        Returns:
            True if Whisper model is loaded
        """
        if self._model is None:
            self._model = _get_whisper_model()
        return self._model is not False

    async def transcribe(
        self,
        audio_file: UploadFile,
    ) -> Optional[str]:
        """Transcribe an audio file.

        Args:
            audio_file: Uploaded audio file

        Returns:
            Transcribed text or None if transcription failed
        """
        # Validate file type
        allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        file_ext = os.path.splitext(audio_file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            logger.error(f"STTProcessor: Unsupported file type: {file_ext}")
            return None

        # Check file size (limit to 50MB)
        file_content = await audio_file.read()
        if len(file_content) > 50 * 1024 * 1024:  # 50MB
            logger.error("STTProcessor: File size too large")
            return None

        # Save to temporary file
        temp_dir = Path(tempfile.gettempdir())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_filename = f"stt_{timestamp}_{audio_file.filename}"
        temp_filepath = temp_dir / temp_filename

        try:
            with open(temp_filepath, "wb") as buffer:
                buffer.write(file_content)

            logger.info(f"STTProcessor: Saved audio to {temp_filepath}")

            # Transcribe the audio
            transcribed_text = _transcribe_audio_file(str(temp_filepath))

            if not transcribed_text:
                logger.warning("STTProcessor: Transcription returned empty result")
                return None

            logger.info(f"STTProcessor: Transcription successful: {transcribed_text[:100]}...")
            return transcribed_text

        except Exception as e:
            logger.error(f"STTProcessor: Error during transcription: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                if temp_filepath.exists():
                    os.remove(temp_filepath)
                    logger.debug(f"STTProcessor: Cleaned up {temp_filepath}")
            except Exception as e:
                logger.warning(f"STTProcessor: Failed to cleanup temp file: {e}")

    async def close(self):
        """Cleanup resources."""
        self._model = None
        logger.info("STTProcessor closed")
