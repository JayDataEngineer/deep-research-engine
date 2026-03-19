"""
Voice input API endpoints for handling audio uploads for voice cloning models.
Integrated with the extension system for proper parameter passing.
"""

import logging
import os
from datetime import datetime
from functools import lru_cache

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=1)
def _get_whisper_model():
    """Get or create cached Whisper model for transcription.

    Uses LRU cache with maxsize=1 to ensure only one model instance
    is loaded at a time, preventing unbounded memory growth.
    """
    try:
        from faster_whisper import WhisperModel
        # Use base model for speed and accuracy balance
        # CPU with INT8 quantization for efficiency
        model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
        return model
    except ImportError:
        logger.warning("faster-whisper not installed, skipping auto-transcription")
        return False
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return False


def _transcribe_audio_file(audio_path: str) -> str | None:
    """Transcribe audio file and return the text."""
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
        logger.info(f"Transcribed audio (language: {info.language}, probability: {info.language_probability:.2f})")

        return full_text
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        return None


@router.post("/upload-voice-sample")
async def upload_voice_sample(
    file: UploadFile = File(...),
    extension_name: str = Form(...),
    ref_text: str = Form("")
):
    """
    Upload a voice sample for voice cloning models.
    This endpoint stores the file temporarily and returns a path that can be used
    in subsequent TTS requests to pass as ref_audio_path parameter.

    Args:
        file: Audio file for voice cloning (typically 3-10 seconds)
        extension_name: Name of the extension that will use this audio
        ref_text: Transcript of the reference audio

    Returns:
        dict: Contains the path to the uploaded file and reference text
    """
    try:
        # Validate file type
        allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not supported: {file_ext}. Supported types: {', '.join(allowed_extensions)}"
            )

        # Check file size (limit to 50MB for voice cloning)
        file_content = await file.read()
        if len(file_content) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size too large. Maximum allowed size is 50MB."
            )

        # Use the application's temp directory for storing voice samples
        temp_dir = get_temp_directory()

        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_filename = f"{extension_name}_voice_sample_{timestamp}_{file.filename}"
        temp_filepath = os.path.join(temp_dir, temp_filename)

        with open(temp_filepath, "wb") as buffer:
            buffer.write(file_content)

        logger.info(f"Voice sample uploaded successfully: {temp_filepath}")

        # Auto-transcribe if ref_text is not provided
        transcribed_text = ref_text
        if not ref_text or not ref_text.strip():
            logger.info("No reference text provided, attempting auto-transcription...")
            transcribed_text = _transcribe_audio_file(temp_filepath)
            if transcribed_text:
                logger.info(f"Auto-transcribed text: {transcribed_text[:100]}...")
            else:
                logger.warning("Auto-transcription failed or unavailable")
                transcribed_text = ""  # Return empty string if transcription failed

        # Return the path to the uploaded file and reference text
        return {
            "success": True,
            "message": "Voice sample uploaded successfully" + (" (auto-transcribed)" if transcribed_text and not ref_text else ""),
            "ref_audio_path": temp_filepath,  # This path can be used in TTS requests
            "ref_text": transcribed_text or ref_text,
            "filename": file.filename
        }

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.error(f"Error uploading voice sample: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload voice sample"
        ) from e


@router.post("/transcribe-audio")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
):
    """
    Transcribe an audio file using Fast Whisper.
    Returns the transcribed text without storing the file.

    Args:
        file: Audio file to transcribe

    Returns:
        dict: Contains the transcribed text
    """
    try:
        # Validate file type
        allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not supported: {file_ext}. Supported types: {', '.join(allowed_extensions)}"
            )

        # Check file size (limit to 50MB)
        file_content = await file.read()
        if len(file_content) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size too large. Maximum allowed size is 50MB."
            )

        # Use the application's temp directory for storing temporary audio
        temp_dir = get_temp_directory()

        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_filename = f"transcribe_{timestamp}_{file.filename}"
        temp_filepath = os.path.join(temp_dir, temp_filename)

        with open(temp_filepath, "wb") as buffer:
            buffer.write(file_content)

        logger.info(f"Audio file saved for transcription: {temp_filepath}")

        # Transcribe the audio
        transcribed_text = _transcribe_audio_file(temp_filepath)

        # Clean up the temporary file
        try:
            os.remove(temp_filepath)
            logger.info(f"Cleaned up temporary file: {temp_filepath}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {temp_filepath}: {e}")

        if not transcribed_text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Transcription failed. Please ensure faster-whisper is installed."
            )

        logger.info(f"Transcription successful: {transcribed_text[:100]}...")

        return {
            "success": True,
            "text": transcribed_text,
            "filename": file.filename
        }

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transcribe audio: {str(e)}"
        ) from e
