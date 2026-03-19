"""
Voice Input Extension

Provides voice input transcription with WhisperX integration.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import ffmpeg
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.extensions.core.base import BaseExtension, ExtensionStatus

logger = logging.getLogger(__name__)


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    duration: Optional[float] = None
    language: Optional[str] = None


class VoiceInputExtension(BaseExtension):
    """
    Voice input extension.

    Provides voice-to-text transcription using WhisperX.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router = None
        self._whisper_model = None
        self._status = ExtensionStatus.LOADED

    @property
    def status(self) -> ExtensionStatus:
        return self._status

    def _get_whisper_model(self):
        """Get or initialize the WhisperX model."""
        if self._whisper_model is None:
            try:
                import whisperx
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Initializing WhisperX on device: {device}")
                self._whisper_model = whisperx.load_model("base", device=device, compute_type="float32")
                logger.info("WhisperX model loaded successfully")
            except ImportError as e:
                logger.error(f"WhisperX not installed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="WhisperX not installed. Please install it with: pip install whisperx"
                )
            except Exception as e:
                logger.error(f"Failed to load WhisperX model: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load WhisperX model: {str(e)}"
                )
        return self._whisper_model

    def _convert_to_wav(self, audio_data: bytes, input_format: str) -> bytes:
        """Convert audio to WAV format using ffmpeg."""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{input_format}", delete=False) as input_file:
                input_path = input_file.name
                input_file.write(audio_data)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                output_path = output_file.name

            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )

            with open(output_path, 'rb') as f:
                wav_data = f.read()

            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

            return wav_data
        except Exception as e:
            logger.error(f"Failed to convert audio: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to convert audio format: {str(e)}"
            )

    def _create_router(self) -> APIRouter:
        """Create the FastAPI router for voice input."""
        router = APIRouter(prefix="/api/v1/voice", tags=["voice-input"])

        @router.post("/transcribe", response_model=TranscriptionResponse)
        async def transcribe_audio(
            audio: UploadFile = File(..., description="Audio file to transcribe")
        ) -> TranscriptionResponse:
            """Transcribe audio file to text using WhisperX."""
            try:
                audio_data = await audio.read()
                content_type = audio.content_type or ""

                # Determine format
                if "webm" in content_type or audio.filename.endswith(".webm"):
                    input_format = "webm"
                elif "wav" in content_type or audio.filename.endswith(".wav"):
                    input_format = "wav"
                elif "mp3" in content_type or audio.filename.endswith(".mp3"):
                    input_format = "mp3"
                elif "ogg" in content_type or audio.filename.endswith(".ogg"):
                    input_format = "ogg"
                elif "mp4" in content_type or audio.filename.endswith(".mp4"):
                    input_format = "mp4"
                else:
                    if "." in audio.filename:
                        input_format = audio.filename.rsplit(".", 1)[-1]
                    else:
                        input_format = "webm"

                logger.info(f"Received audio file: {audio.filename}, format: {input_format}, size: {len(audio_data)} bytes")

                # Convert to WAV if needed
                if input_format != "wav":
                    audio_data = self._convert_to_wav(audio_data, input_format)

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(audio_data)

                try:
                    model = self._get_whisper_model()
                    result = model.transcribe(temp_path, batch_size=16)

                    text = result["segments"][0]["text"] if result.get("segments") else ""
                    language = result.get("language", "unknown")

                    duration = None
                    if result.get("segments"):
                        last_segment = result["segments"][-1]
                        duration = last_segment.get("end", 0)

                    logger.info(f"Transcription complete: {text[:50]}... (duration: {duration}s)")

                    return TranscriptionResponse(
                        text=text.strip(),
                        duration=duration,
                        language=language
                    )

                finally:
                    Path(temp_path).unlink(missing_ok=True)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {str(e)}"
                )

        @router.get("/health")
        async def health_check():
            """Check if WhisperX is available."""
            try:
                import whisperx
                return {
                    "status": "available",
                    "model_loaded": self._whisper_model is not None
                }
            except ImportError:
                return {
                    "status": "not_available",
                    "model_loaded": False,
                    "message": "WhisperX not installed. Install with: pip install whisperx"
                }

        return router

    def register_routes(self, app) -> None:
        """Register voice input API routes."""
        if self._router is None:
            self._router = self._create_router()
        app.include_router(self._router)

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/api/v1/voice",
            "name": "voice_input",
            "tags": ["voice", "transcription", "whisper", "audio"],
            "description": "Voice input transcription with WhisperX integration",
        }

    async def initialize(self, context) -> None:
        """Initialize the voice input extension."""
        self._status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Shutdown the voice input extension."""
        self._status = ExtensionStatus.UNLOADED
