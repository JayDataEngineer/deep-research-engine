"""
Simple CLI-friendly TTS endpoint.

Designed for easy terminal/curl usage without UI.
Returns audio directly or JSON with audio URL.
"""

import base64
import io
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.shared.tools.audio_utils import audio_service
from src.extensions.libraries.tts_engine.speakers import get_speaker_manager
from src.shared.factory import get_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cli", tags=["cli"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SpeakRequest(BaseModel):
    """Simple request for TTS generation."""

    text: str = Field(..., description="Text to speak", min_length=1, max_length=5000)
    speaker: str | None = Field(None, description="Speaker ID (uses default if not provided)")
    emotion: str | None = Field("neutral", description="Emotion preset: neutral, happy, sad, angry, etc.")
    format: str = Field("wav", description="Output format: wav, mp3, ogg")
    return_type: str = Field("url", description="Response type: url (JSON) or audio (binary)")


class SpeakResponse(BaseModel):
    """Response for TTS generation."""

    success: bool
    audio_url: str | None = None
    duration: float | None = None
    message: str


# ============================================================================
# Emotion presets (8-dimensional vectors)
# ============================================================================

EMOTION_PRESETS = {
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
    "happy": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2],
    "sad": [0.0, 0.0, 0.8, 0.2, 0.0, 0.6, 0.0, 0.1],
    "angry": [0.0, 0.9, 0.2, 0.3, 0.5, 0.1, 0.1, 0.0],
    "afraid": [0.0, 0.2, 0.3, 0.8, 0.1, 0.4, 0.6, 0.0],
    "surprised": [0.5, 0.1, 0.0, 0.2, 0.0, 0.0, 0.9, 0.1],
    "calm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
    "melancholic": [0.0, 0.0, 0.6, 0.1, 0.0, 0.8, 0.1, 0.3],
}


def get_emotion_vector(emotion: str) -> list[float]:
    """Get emotion vector for preset name."""
    return EMOTION_PRESETS.get(emotion.lower(), EMOTION_PRESETS["neutral"])


async def generate_tts_audio(
    text: str,
    speaker: str | None = None,
    emotion: str = "neutral",
    output_format: str = "wav"
) -> tuple[bytes, float]:
    """Generate TTS audio.

    Returns:
        Tuple of (audio_bytes, duration_seconds)
    """
    speaker_manager = get_speaker_manager()

    # Get emotion vector
    emotion_vector = get_emotion_vector(emotion)

    # Get speaker audio path
    ref_audio_path = None

    if speaker:
        if speaker_manager.speaker_exists(speaker):
            ref_audio_path = speaker_manager.get_audio_path(speaker)
        else:
            from pathlib import Path
            speaker_path = Path(speaker)
            if speaker_path.exists():
                ref_audio_path = str(speaker_path)

    # Use default speaker if none provided
    if not ref_audio_path:
        available_speakers = speaker_manager.list_speakers(active_only=True)
        if available_speakers:
            ref_audio_path = speaker_manager.get_audio_path(available_speakers[0].speaker_id)

    if not ref_audio_path:
        raise HTTPException(
            status_code=400,
            detail="No speaker available. Register a speaker first."
        )

    # Convert emotion to IndexTTS format
    happy, angry, sad, afraid, _, melancholic, surprised, calm = emotion_vector
    indextts_emotion = [
        max(-1.0, min(1.0, (calm - melancholic) * 2 - 1)),
        min(1.0, happy + surprised),
        0.5,
        happy,
        sad,
        angry,
        afraid,
        surprised,
    ]

    # Call IndexTTS
    import httpx
    from src.config import settings

    local_base_url = f"http://{settings.host}:{settings.port}"

    # Ensure IndexTTS is loaded
    async with httpx.AsyncClient(timeout=120.0) as client:
        status_response = await client.get(
            f"{local_base_url}/api/v1/tts/extension-process/status",
            timeout=5.0
        )
        status_data = status_response.json()
        if status_data.get("status") != "ready":
            logger.info("[CLI] Loading IndexTTS...")
            load_response = await client.post(
                f"{local_base_url}/api/v1/tts/extension-process/load",
                json={"extension": "index_tts"},
                timeout=60.0
            )
            load_data = load_response.json()
            if load_data.get("status") != "ready":
                raise HTTPException(
                    status_code=503,
                    detail=f"IndexTTS not ready: {load_data}"
                )

        # Generate audio
        response = await client.post(
            f"{local_base_url}/api/v1/tts/external/indextts/synthesize",
            json={
                "text": text,
                "emotion": indextts_emotion,
                "format": output_format,
                "sample_rate": 24000,
                "speaker": ref_audio_path,
            }
        )
        response.raise_for_status()
        data = response.json()

    # Decode audio
    audio_bytes = base64.b64decode(data["audio"])
    duration_ms = data["duration_ms"]
    duration_seconds = duration_ms / 1000.0

    return audio_bytes, duration_seconds


@router.post("/speak")
async def speak(request: SpeakRequest) -> SpeakResponse:
    """
    Simple TTS endpoint - returns JSON with audio URL.

    Easy to use from terminal:

    ```bash
    curl -X POST http://localhost:8787/api/v1/tts/cli/speak \\
      -H "Content-Type: application/json" \\
      -d '{"text": "Hello world", "emotion": "happy"}'
    ```
    """
    try:
        audio_bytes, duration = await generate_tts_audio(
            text=request.text,
            speaker=request.speaker,
            emotion=request.emotion or "neutral",
            output_format=request.format
        )

        # Save audio
        audio_buffer = io.BytesIO(audio_bytes)
        audio_url, _ = await audio_service.save_and_get_url(audio_buffer, request.format)

        return SpeakResponse(
            success=True,
            audio_url=audio_url,
            duration=duration,
            message=f"Generated {len(audio_bytes)} bytes of audio"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CLI] TTS failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speak/audio")
async def speak_audio(request: SpeakRequest) -> Response:
    """
    TTS endpoint that returns audio directly (binary response).

    Easy to use from terminal:

    ```bash
    # Play directly with paplay (Linux)
    curl -X POST http://localhost:8787/api/v1/tts/cli/speak/audio \\
      -H "Content-Type: application/json" \\
      -d '{"text": "Hello world", "emotion": "happy"}' \\
      --output spoken.wav && paplay spoken.wav

    # Or with aplay
    curl -X POST http://localhost:8787/api/v1/tts/cli/speak/audio \\
      -H "Content-Type: application/json" \\
      -d '{"text": "Hello world"}' | aplay

    # Save to file
    curl -X POST http://localhost:8787/api/v1/tts/cli/speak/audio \\
      -H "Content-Type: application/json" \\
      -d '{"text": "Hello world"}' -o hello.wav
    ```
    """
    try:
        audio_bytes, duration = await generate_tts_audio(
            text=request.text,
            speaker=request.speaker,
            emotion=request.emotion or "neutral",
            output_format=request.format
        )

        # Map format to content type
        content_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
        }
        content_type = content_types.get(request.format, "audio/wav")

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.format}",
                "X-Audio-Duration": str(duration),
                "X-Audio-Size": str(len(audio_bytes)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CLI] TTS audio failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speak")
async def speak_get(
    text: str = Query(..., description="Text to speak"),
    emotion: str = Query("neutral", description="Emotion preset"),
    speaker: str | None = Query(None, description="Speaker ID"),
    format: str = Query("wav", description="Output format"),
):
    """
    GET version of speak for even simpler terminal usage.

    ```bash
    curl "http://localhost:8787/api/v1/tts/cli/speak?text=Hello%20world&emotion=happy" -o hello.wav
    ```
    """
    request = SpeakRequest(text=text, speaker=speaker, emotion=emotion, format=format)
    return await speak(request)


@router.get("/emotions")
async def list_emotions():
    """List available emotion presets."""
    return {
        "emotions": list(EMOTION_PRESETS.keys()),
        "description": "Use any of these emotion names with the speak endpoint"
    }
