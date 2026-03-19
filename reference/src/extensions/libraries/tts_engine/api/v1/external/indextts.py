"""
IndexTTS2 compatibility endpoint for external middleware integration.

This endpoint provides an API contract compatible with the IndexTTS2 middleware
that expects specific request/response formats different from the main Cozy TTS API.

IndexTTS2 Middleware Contract:
    POST /api/v1/external/indextts/synthesize
    Request: { text, emotion: float[8], format, sample_rate, speaker }
    Response: { audio: base64, format, sample_rate, duration_ms }
"""

import base64
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.extensions.libraries.tts_engine.speakers import get_speaker_manager
from src.extensions.libraries.tts_engine.router import tts_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["External API"])


class IndexTTS2SynthesizeRequest(BaseModel):
    """Request model compatible with IndexTTS2 middleware expectations."""

    text: str
    emotion: List[float] = Field(description="8-dimensional emotion vector")
    format: str = "wav"
    sample_rate: int = 24000
    speaker: str | None = None


class IndexTTS2SynthesizeResponse(BaseModel):
    """Response model compatible with IndexTTS2 middleware expectations."""

    audio: str = Field(description="base64 encoded audio")
    format: str
    sample_rate: int
    duration_ms: int


def _validate_emotion_vector(emotion: list[float]) -> None:
    """Validate the 8-dimensional emotion vector.

    Expected format: [valence, arousal, dominance, joy, sadness, anger, fear, surprise]

    Args:
        emotion: List of 8 float values

    Raises:
        HTTPException: If validation fails
    """
    if not isinstance(emotion, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Emotion must be a list of 8 float values"
        )

    if len(emotion) != 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Emotion vector must have 8 dimensions, got {len(emotion)}"
        )

    # Validate ranges
    # valence: -1.0 to 1.0
    if not -1.0 <= emotion[0] <= 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Valence must be between -1.0 and 1.0, got {emotion[0]}"
        )

    # Most other dimensions: 0.0 to 1.0
    for i, value in enumerate(emotion[1:], start=1):
        if not 0.0 <= value <= 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Emotion dimension {i} must be between 0.0 and 1.0, got {value}"
            )


def _emotion_to_indextts_params(emotion: list[float]) -> dict:
    """Convert IndexTTS2 middleware emotion vector to IndexTTS extension parameters.

    IndexTTS2 middleware emotion format:
    [valence, arousal, dominance, joy, sadness, anger, fear, surprise]

    IndexTTS extension emotion vector format:
    [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

    Mapping approximation:
        - joy → happy
        - anger → angry
        - sadness → sad
        - fear → afraid
        - surprise → surprised
        - valence positive → calm (inverse relationship)
        - Others map proportionally

    Args:
        emotion: 8-dimensional emotion vector from middleware

    Returns:
        Dictionary with emo_vector and other emotion-related parameters
    """
    [
        valence,    # -1.0 (negative) to 1.0 (positive)
        arousal,    # 0.0 (calm) to 1.0 (excited)
        dominance,  # 0.0 (submissive) to 1.0 (dominant)
        joy,        # 0.0 to 1.0
        sadness,    # 0.0 to 1.0
        anger,      # 0.0 to 1.0
        fear,       # 0.0 to 1.0
        surprise    # 0.0 to 1.0
    ] = emotion

    # Map to IndexTTS emotion vector format
    # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    indextts_emotion = [
        joy,                    # happy
        anger,                  # angry
        sadness,                # sad
        fear,                   # afraid (scared)
        0.0,                    # disgusted (no direct mapping)
        sadness * 0.7,          # melancholic (related to sadness)
        surprise,                # surprised
        max(0.0, valence * 0.8), # calm (positive valence = calm)
    ]

    return {
        "emo_vector": indextts_emotion,
        "use_emo_text": False,
        "emo_alpha": 1.0,  # Full emotion strength
    }


@router.post("/indextts/synthesize")
async def indextts2_synthesize(request: IndexTTS2SynthesizeRequest) -> IndexTTS2SynthesizeResponse:
    """
    IndexTTS2 compatibility endpoint for external middleware integration.

    This endpoint translates between the IndexTTS2 middleware API contract and
    the internal Cozy TTS API, then returns audio in the expected format.

    Args:
        request: IndexTTS2 middleware compatible request

    Returns:
        IndexTTS2 middleware compatible response with base64-encoded audio

    Raises:
        HTTPException: If validation fails or TTS generation errors
    """
    try:
        # Validate emotion vector
        _validate_emotion_vector(request.emotion)

        # Convert emotion vector to IndexTTS parameters
        emotion_params = _emotion_to_indextts_params(request.emotion)

        # Get the speaker manager
        speaker_manager = get_speaker_manager()

        # Determine voice ID from speaker parameter
        # IndexTTS uses reference audio for voice cloning, so speaker might
        # refer to a preset or reference audio identifier
        voice_id = request.speaker or "index2_ref_neutral"

        # Resolve reference audio path from speaker mapping
        ref_audio_path = None
        if request.speaker:
            # First, try the speaker manager (managed speakers)
            if speaker_manager.speaker_exists(request.speaker):
                ref_audio_path = speaker_manager.get_audio_path(request.speaker)
                logger.info(f"Using managed speaker '{request.speaker}': {ref_audio_path}")
            else:
                # Check if speaker might be a direct file path
                speaker_path = Path(request.speaker)
                if speaker_path.exists() and speaker_path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                    ref_audio_path = str(speaker_path)
                    logger.info(f"Using speaker as direct file path: {ref_audio_path}")
                else:
                    # Speaker not found - get list of available speakers for error message
                    available = speaker_manager.list_speakers(active_only=True)
                    available_names = [s.speaker_id for s in available]
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Speaker '{request.speaker}' not recognized. "
                        f"Available speakers: {available_names if available_names else 'none'}. "
                        f"Register a speaker with POST /api/v1/external/indextts/speakers or provide a valid audio file path."
                    )

        # Verify reference audio exists
        if ref_audio_path and not Path(ref_audio_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference audio file not found: {ref_audio_path}"
            )

        # Build extension parameters
        extension_params = {
            **emotion_params,
            "ref_audio_path": ref_audio_path,
        }

        # Log the synthesis request
        logger.info(
            f"IndexTTS2 compatibility synthesis: text='{request.text[:50]}...', "
            f"speaker='{request.speaker}', ref_audio='{ref_audio_path}', emotion={request.emotion}"
        )

        # Generate audio using the TTS service
        # Note: IndexTTS requires reference audio for voice cloning
        # We pass empty string for voice_id since IndexTTS uses ref_audio_path instead
        audio_bytes, duration_seconds = await tts_service.generate_audio(
            text=request.text,
            voice_id="",  # IndexTTS uses ref_audio_path, not voice_id
            **extension_params
        )

        # Encode audio to base64
        # audio_bytes is io.BytesIO, so read it first
        audio_data = audio_bytes.read()
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        # Convert duration to milliseconds
        duration_ms = int(duration_seconds * 1000)

        logger.info(
            f"IndexTTS2 synthesis complete: {len(audio_data)} bytes, "
            f"{duration_ms}ms, format={request.format}"
        )

        return IndexTTS2SynthesizeResponse(
            audio=audio_b64,
            format=request.format,
            sample_rate=request.sample_rate,
            duration_ms=duration_ms,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"IndexTTS2 synthesis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failed: {str(e)}"
        ) from e


@router.get("/indextts/speakers")
async def list_speakers():
    """
    List available speakers and their reference audio mappings.

    Returns information about configured speakers for the IndexTTS2
    compatibility endpoint, including which reference audio files
    are mapped to which speaker names.

    Note: This is a simplified endpoint for backward compatibility.
    For full speaker management, see the dedicated endpoints:
    - POST /api/v1/external/indextts/speakers - Register a new speaker
    - GET /api/v1/external/indextts/speakers - List all speakers (with metadata)
    - GET /api/v1/external/indextts/speakers/{id} - Get speaker details
    """
    speaker_manager = get_speaker_manager()
    speakers = speaker_manager.list_speakers(active_only=True)

    return {
        "available_speakers": [s.speaker_id for s in speakers],
        "mappings": {
            s.speaker_id: s.audio_path
            for s in speakers
        },
        "default_ref_audio": None,  # No default - must specify speaker or direct file path
        "note": "You can also pass a direct file path as the 'speaker' parameter, or register speakers with POST /api/v1/external/indextts/speakers"
    }
