"""
Chat TTS API endpoints.

Provides TTS generation for chat messages with automatic emotion detection.
"""

import asyncio
import base64
import io
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.shared.tools.audio_utils import audio_service
from src.extensions.libraries.tts_engine.speakers import get_speaker_manager
from src.shared.factory import get_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat-tts"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatTTSRequest(BaseModel):
    """Request for TTS generation with automatic emotion detection."""

    text: str = Field(..., description="Text to convert to speech", min_length=1)
    speaker: str | None = Field(None, description="Speaker ID for voice cloning (uses default if not provided)")
    voice_sample: str | None = Field(None, description="Base64-encoded voice sample for one-shot voice cloning")
    personality: str | None = Field(
        "neutral",
        description="Optional personality hint for emotion detection (e.g., 'friendly', 'professional', 'casual')"
    )
    format: str = Field("wav", description="Audio output format (wav, mp3, ogg)")
    # Roleplay voice switching (only used in roleplay mode)
    voice_role: str | None = Field(None, description="Voice role for roleplay: 'character' or 'narrator'")
    character_speaker: str | None = Field(None, description="Character speaker ID (for roleplay mode)")
    narrator_speaker: str | None = Field(None, description="Narrator speaker ID (for roleplay mode)")
    emotion_config_id: str | None = Field(None, description="Emotion config ID from emotion-tagging (IndexTTS only)")


class ChatTTSResponse(BaseModel):
    """Response for TTS generation."""

    success: bool
    audio_url: str | None = None
    duration: float | None = None
    emotion_vector: list[float] | None = None
    message: str


# ============================================================================
# Emotion Detection (LLM-based)
# ============================================================================

async def detect_emotion_llm(text: str, personality: str = "neutral") -> list[float]:
    """Use LLM to detect emotion for text.

    Returns 8-dimensional emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    Each value is 0.0 to 1.0 intensity.

    Args:
        text: Text to analyze
        personality: Optional personality hint for context

    Returns:
        List of 8 float values representing emotion intensity
    """
    personality_hints = {
        "friendly": "You are a warm, friendly assistant. You speak with genuine warmth and approachability.",
        "professional": "You are a professional, competent assistant. You speak with clarity and confidence.",
        "casual": "You are a casual, relaxed assistant. You speak informally and naturally.",
        "neutral": "You are a neutral speaker providing clear information.",
    }

    personality_prompt = personality_hints.get(personality, personality_hints["neutral"])

    prompt = f"""{personality_prompt}

Analyze the EMOTION of this text and respond with ONLY an 8-float list specifying the intensity of each emotion, in the following order:
[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

Each value should be between 0.0 and 1.0. Complex emotions can have multiple non-zero values.

Text: "{text[:200]}"

Respond with ONLY the 8 floats, like: [0.1, 0.8, 0.3, 0.2, 0.0, 0.1, 0.4, 0.0]
Nothing else. No explanation."""

    try:
        llm = await get_llm(temperature=0.1)
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()

        # Parse the list from response
        import ast
        emotion_vector = ast.literal_eval(content)

        if isinstance(emotion_vector, list) and len(emotion_vector) == 8:
            # Ensure all values are floats 0-1
            emotion_vector = [max(0.0, min(1.0, float(v))) for v in emotion_vector]
            logger.debug(f"Detected emotion vector: {emotion_vector}")
            return emotion_vector
        else:
            logger.warning(f"Invalid emotion response: {content}, using neutral")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]

    except Exception as e:
        logger.error(f"LLM emotion detection failed: {e}")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]


# ============================================================================
# Retry Logic
# ============================================================================

async def retry_with_backoff(func, max_retries: int = 3, *args, **kwargs):
    """Retry a function with exponential backoff."""
    import random

    last_exception = None
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise

            delay = min(2.0 * (2.0 ** attempt), 10.0)
            delay *= (0.5 + random.random() * 0.5)
            logger.info(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
            await asyncio.sleep(delay)

    raise last_exception


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/generate", response_model=ChatTTSResponse)
async def generate_chat_tts(request: ChatTTSRequest) -> ChatTTSResponse:
    """
    Generate TTS for chat messages with automatic emotion detection.

    This endpoint:
    1. Detects emotion from the text using the LLM
    2. Generates audio using IndexTTS with the emotion vector
    3. Returns a URL to the generated audio

    Args:
        request: ChatTTSRequest with text and optional parameters

    Returns:
        ChatTTSResponse with audio URL and emotion vector
    """
    speaker_manager = get_speaker_manager()

    try:
        # Step 1: Detect emotion using LLM (or use emotion config)
        logger.info(f"[CHAT TTS] Detecting emotion for text: '{request.text[:50]}...'")

        # Check for emotion config from emotion-tagging
        if request.emotion_config_id:
            # Load saved emotion config
            try:
                from src.extensions.libraries.tts_engine.config_storage import TTSConfigStorage
                storage = TTSConfigStorage()
                config = await storage.get_emotion_preset(request.emotion_config_id)
                if config:
                    # Use the personality from config for detection
                    personality = config.get("personality", request.personality or "neutral")
                    emotion_vector = await detect_emotion_llm(request.text, personality)
                    logger.info(f"[CHAT TTS] Using emotion config '{request.emotion_config_id}'")
                else:
                    emotion_vector = await detect_emotion_llm(request.text, request.personality or "neutral")
            except Exception as e:
                logger.warning(f"[CHAT TTS] Failed to load emotion config: {e}, using default")
                emotion_vector = await detect_emotion_llm(request.text, request.personality or "neutral")
        else:
            emotion_vector = await detect_emotion_llm(request.text, request.personality or "neutral")

        # Step 2: Determine speaker/voice (with roleplay support)
        ref_audio_path = None

        # Handle roleplay voice switching
        if request.voice_role:
            role_speaker = None
            if request.voice_role == "character" and request.character_speaker:
                role_speaker = request.character_speaker
                logger.info(f"[CHAT TTS] Using CHARACTER voice: {role_speaker}")
            elif request.voice_role == "narrator" and request.narrator_speaker:
                role_speaker = request.narrator_speaker
                logger.info(f"[CHAT TTS] Using NARRATOR voice: {role_speaker}")

            if role_speaker:
                if speaker_manager.speaker_exists(role_speaker):
                    ref_audio_path = speaker_manager.get_audio_path(role_speaker)
                else:
                    from pathlib import Path
                    speaker_path = Path(role_speaker)
                    if speaker_path.exists():
                        ref_audio_path = str(speaker_path)

        # Fall through to normal voice selection if roleplay didn't set it
        if not ref_audio_path and request.voice_sample:
            # Decode base64 voice sample and save to temp file
            voice_data = base64.b64decode(request.voice_sample)
            ref_audio_path = await audio_service.save_temp_audio(voice_data, "voice_sample.wav")
            logger.info(f"[CHAT TTS] Using provided voice sample: {ref_audio_path}")

        elif not ref_audio_path and request.speaker:
            # Use speaker from speaker manager
            if speaker_manager.speaker_exists(request.speaker):
                ref_audio_path = speaker_manager.get_audio_path(request.speaker)
                logger.info(f"[CHAT TTS] Using speaker '{request.speaker}': {ref_audio_path}")
            else:
                # Check if it's a direct file path
                from pathlib import Path
                speaker_path = Path(request.speaker)
                if speaker_path.exists():
                    ref_audio_path = str(speaker_path)
                    logger.info(f"[CHAT TTS] Using direct file path: {ref_audio_path}")
                else:
                    available = [s.speaker_id for s in speaker_manager.list_speakers(active_only=True)]
                    logger.warning(f"Speaker '{request.speaker}' not found. Available: {available}")

        # Use default speaker if none provided
        if not ref_audio_path:
            available_speakers = speaker_manager.list_speakers(active_only=True)
            if available_speakers:
                ref_audio_path = speaker_manager.get_audio_path(available_speakers[0].speaker_id)
                logger.info(f"[CHAT TTS] Using default speaker: {available_speakers[0].speaker_id}")

        if not ref_audio_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No speaker available. Please register a speaker or provide a voice sample."
            )

        # Step 3: Call IndexTTS synthesis
        # Import here to avoid circular imports
        import httpx

        from src.config import settings

        # Use local backend URL for IndexTTS endpoint
        # The llm_base_url points to external LLM API, but IndexTTS is local
        local_base_url = f"http://{settings.host}:{settings.port}"

        # Call the external IndexTTS endpoint
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Ensure IndexTTS extension is loaded
            try:
                status_response = await client.get(f"{local_base_url}/api/v1/tts/extension-process/status", timeout=5.0)
                status_data = status_response.json()
                if status_data.get("status") != "ready":
                    logger.info("[CHAT TTS] IndexTTS not loaded, loading...")
                    load_response = await client.post(
                        f"{local_base_url}/api/v1/tts/extension-process/load",
                        json={"extension": "index_tts"},
                        timeout=60.0
                    )
                    load_data = load_response.json()
                    if load_data.get("status") != "ready":
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"IndexTTS extension not ready: {load_data}"
                        )
                    logger.info("[CHAT TTS] IndexTTS loaded successfully")
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Timeout checking IndexTTS status"
                )

            # Convert emotion to IndexTTS2 format: [valence, arousal, dominance, joy, sadness, anger, fear, surprise]
            # Our format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            happy, angry, sad, afraid, _, melancholic, surprised, calm = emotion_vector

            # Map to IndexTTS2 format
            indextts_emotion = [
                max(-1.0, min(1.0, (calm - melancholic) * 2 - 1)),  # valence (calm=positive, sad=negative)
                min(1.0, happy + surprised),  # arousal (excited)
                0.5,  # dominance (neutral)
                happy,  # joy
                sad,  # sadness
                angry,  # anger
                afraid,  # fear
                surprised,  # surprise
            ]

            response = await client.post(
                f"{local_base_url}/api/v1/tts/external/indextts/synthesize",
                json={
                    "text": request.text,
                    "emotion": indextts_emotion,
                    "format": request.format,
                    "sample_rate": 24000,
                    "speaker": ref_audio_path,
                }
            )
            response.raise_for_status()
            data = response.json()

        # Decode base64 audio
        audio_data = base64.b64decode(data["audio"])
        duration_ms = data["duration_ms"]
        duration_seconds = duration_ms / 1000.0

        # Step 4: Save audio and get URL
        audio_buffer = io.BytesIO(audio_data)
        audio_url, _ = await audio_service.save_and_get_url(audio_buffer, request.format)

        logger.info(f"[CHAT TTS] Generated {len(audio_data)} bytes of audio: {audio_url}")

        return ChatTTSResponse(
            success=True,
            audio_url=audio_url,
            duration=duration_seconds,
            emotion_vector=emotion_vector,
            message="TTS generated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHAT TTS] Generation failed: {e}", exc_info=True)
        return ChatTTSResponse(
            success=False,
            audio_url=None,
            duration=None,
            emotion_vector=None,
            message=f"TTS generation failed: {str(e)}",
        )


@router.get("/speakers")
async def list_chat_speakers():
    """List available speakers for chat TTS."""
    speaker_manager = get_speaker_manager()
    speakers = speaker_manager.list_speakers(active_only=True)

    return {
        "speakers": [
            {
                "id": s.speaker_id,
                "name": s.name,
                "gender": s.gender,
                "description": s.description,
            }
            for s in speakers
        ],
        "count": len(speakers),
        "default": speakers[0].speaker_id if speakers else None,
    }


class SpeakerRegisterRequest(BaseModel):
    """Request model for registering a speaker."""
    speaker_id: str
    display_name: str | None = None
    gender: str = "unknown"
    description: str = ""
    voice_sample: str = Field(..., description="Base64-encoded voice sample")


@router.post("/speakers/register")
async def register_chat_speaker(request: SpeakerRegisterRequest):
    """
    Register a new speaker for chat TTS.

    Args:
        speaker_id: Unique identifier for the speaker
        display_name: Human-readable name
        gender: Speaker gender (male, female, unknown)
        description: Optional description
        voice_sample: Base64-encoded audio file for voice cloning

    Returns:
        Success status
    """
    speaker_id = request.speaker_id
    display_name = request.display_name
    gender = request.gender
    description = request.description
    voice_sample = request.voice_sample
    try:
        # Decode and save voice sample
        speaker_manager = get_speaker_manager()
        voice_data = base64.b64decode(voice_sample)
        audio_path = await audio_service.save_temp_audio(voice_data, f"{speaker_id}.wav")

        # Register speaker
        speaker_manager.register_speaker(
            speaker_id=speaker_id,
            audio_path=audio_path,
            display_name=display_name or speaker_id,
            gender=gender,
            description=description,
        )

        logger.info(f"[CHAT TTS] Registered speaker '{speaker_id}': {audio_path}")

        return {
            "success": True,
            "speaker_id": speaker_id,
            "audio_path": audio_path,
        }

    except Exception as e:
        logger.error(f"[CHAT TTS] Failed to register speaker: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register speaker: {str(e)}"
        )
