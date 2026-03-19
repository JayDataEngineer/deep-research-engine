"""
LLM Integration API endpoints.

Provides simple endpoints for external LLM services to generate TTS audio.
Designed for integration with unified-llm-interface and similar services.
"""

import io
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.extensions.libraries.tts_engine.router import tts_service

logger = logging.getLogger(__name__)

router = APIRouter()


class LLMTTSRequest(BaseModel):
    """Simple TTS request for LLM integration."""

    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=5000)
    voice: str | None = Field(None, description="Voice ID to use (optional, uses default if not specified)")
    extension: str = Field("kokoro", description="Extension to use for TTS (kokoro, vibevoice, maya1, etc.)")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.25, le=4.0)
    format: Literal["wav", "mp3", "ogg"] = Field("wav", description="Audio output format")

    model_config = {"extra": "allow"}


class LLMTTSResponse(BaseModel):
    """TTS response for LLM integration."""

    success: bool
    message: str
    audio_url: str | None = None
    duration: float | None = None


@router.post("/llm/tts", response_model=LLMTTSResponse)
async def llm_tts_generate(request: LLMTTSRequest):
    """Generate TTS audio from text - designed for LLM integration.

    This is a simplified endpoint for external LLM services to generate TTS audio.
    It uses sensible defaults and is designed to be called programmatically.

    Args:
        request: TTS request with text and optional parameters

    Returns:
        TTS response with audio URL

    Example:
        POST /api/v1/llm/tts
        {
            "text": "Hello, world!",
            "voice": "af_bella",
            "extension": "kokoro",
            "format": "wav"
        }
    """
    logger.info(f"LLM TTS Request: text='{request.text[:50]}...', extension={request.extension}, voice={request.voice}")

    try:
        # Validate text length
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too long. Maximum length is 5000 characters.",
            )

        # Prepare parameters
        all_params = {
            "text": request.text,
            "voice_id": request.voice,
            "speed": request.speed,
            "pitch": 1.0,  # Default pitch
            "audio_format": request.format,
            # Pass extension_name as an extra parameter
            "extension_name": request.extension,
        }

        # Add any extra fields from the request (extension-specific parameters)
        if hasattr(request, "model_extra") and request.model_extra:
            for key, value in request.model_extra.items():
                if not key.startswith("_"):
                    all_params[key] = value

        logger.info(f"Calling generate_audio with parameters: {list(all_params.keys())}")

        # Generate audio
        try:
            audio_buffer, duration = await tts_service.generate_audio(**all_params)
            audio_size = len(audio_buffer.getvalue()) if audio_buffer else 0
            logger.info(f"LLM TTS Generation Success: duration={duration}s, audio_size={audio_size} bytes")
        except Exception as e:
            logger.error(f"Error in tts_service.generate_audio: {str(e)}", exc_info=True)
            return LLMTTSResponse(
                success=False,
                message=f"Failed to generate speech: {str(e)}",
                audio_url=None,
                duration=None
            )

        # Save the audio file
        from datetime import datetime

        from src.extensions.libraries.tts_engine.api.utils.storage import get_audio_url, save_audio_file

        audio_id = str(datetime.now().timestamp()).replace(".", "")
        audio_bytes = audio_buffer.getvalue()
        save_audio_file(audio_bytes, audio_id, request.format)
        audio_url = get_audio_url(audio_id, request.format)

        logger.info(f"LLM TTS audio saved: {audio_url}")

        return LLMTTSResponse(
            success=True,
            message="Speech generated successfully",
            audio_url=audio_url,
            duration=duration,
        )

    except Exception as e:
        logger.error(f"Error generating LLM TTS: {str(e)}", exc_info=True)

        if isinstance(e, HTTPException):
            raise

        return LLMTTSResponse(
            success=False,
            message=f"Failed to generate speech: {str(e)}",
            audio_url=None,
            duration=None
        )


@router.post("/llm/tts/stream")
async def llm_tts_stream(request: LLMTTSRequest):
    """Generate TTS audio and stream directly - designed for LLM integration.

    This endpoint returns the audio directly as a stream, avoiding file storage.
    Useful for real-time applications where you want the audio immediately.

    Args:
        request: TTS request with text and optional parameters

    Returns:
        Streaming audio response

    Example:
        POST /api/v1/llm/tts/stream
        {
            "text": "Hello, world!",
            "voice": "af_bella",
            "extension": "kokoro",
            "format": "wav"
        }
    """
    logger.info(f"LLM TTS Stream Request: text='{request.text[:50]}...', extension={request.extension}, voice={request.voice}")

    try:
        # Validate text length
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too long. Maximum length is 5000 characters.",
            )

        # Prepare parameters
        all_params = {
            "text": request.text,
            "voice_id": request.voice,
            "speed": request.speed,
            "pitch": 1.0,  # Default pitch
            "audio_format": request.format,
            # Pass extension_name as an extra parameter
            "extension_name": request.extension,
        }

        # Add any extra fields from the request
        if hasattr(request, "model_extra") and request.model_extra:
            for key, value in request.model_extra.items():
                if not key.startswith("_"):
                    all_params[key] = value

        # Generate audio
        try:
            audio_buffer, duration = await tts_service.generate_audio(**all_params)
            audio_size = len(audio_buffer.getvalue()) if audio_buffer else 0
            logger.info(f"LLM TTS Stream Success: duration={duration}s, audio_size={audio_size} bytes")
        except Exception as e:
            logger.error(f"Error in tts_service.generate_audio for stream: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate speech: {str(e)}",
            ) from e

        # Set appropriate media type based on format
        media_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}.get(
            request.format, "audio/wav"
        )

        logger.info(f"Streaming LLM TTS audio for text: '{request.text[:50]}...'")

        # Return streaming response
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.format}",
                "X-Audio-Duration": str(duration),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating LLM TTS stream: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate speech stream",
        ) from e


@router.get("/llm/voices")
async def llm_get_voices(extension: str | None = None):
    """Get available voices - designed for LLM integration.

    Returns a simple list of available voices, optionally filtered by extension.

    Args:
        extension: Optional extension name to filter voices (e.g., "kokoro", "vibevoice")

    Returns:
        List of available voices

    Example:
        GET /api/v1/llm/voices?extension=kokoro
    """
    try:
        from src.extensions.libraries.tts_engine.api.schemas.tts import VoiceInfo

        # Refresh voices
        await tts_service.refresh_voices()

        all_voices = tts_service.get_available_voices()

        # Filter by extension if specified
        if extension:
            # Extension names now consistently use underscores
            # Get extension_manager from FastAPI app state
    from src.api.router_utils import get_extension_manager
    extension_manager = get_extension_manager(request)

            # Get the extension
            ext = extension_manager.extensions.get(extension)
            if ext:
                # Get voices from this extension
                ext_voices = ext.get_voices()
                voices = []
                for v in ext_voices:
                    # Handle both dict and VoiceInfo Pydantic model types
                    if isinstance(v, dict):
                        voices.append(VoiceInfo(
                            id=v.get("id", ""),
                            name=v.get("name", ""),
                            language=v.get("language", "en-US"),
                            gender=v.get("gender", "unknown"),
                            description=v.get("description", ""),
                        ))
                    else:
                        # v is already a VoiceInfo or similar Pydantic model
                        voices.append(VoiceInfo(
                            id=getattr(v, "id", ""),
                            name=getattr(v, "name", ""),
                            language=getattr(v, "language", "en-US"),
                            gender=getattr(v, "gender", "unknown"),
                            description=getattr(v, "description", ""),
                        ))
                logger.info(f"Retrieved {len(voices)} voices for extension: {extension}")
                return {"voices": voices, "extension": extension}
            else:
                logger.warning(f"Extension '{extension}' not found")
                return {"voices": [], "extension": extension}

        # Return all voices with their extensions
        logger.info(f"Retrieved {len(all_voices)} total voices")
        return {"voices": all_voices, "extension": "all"}

    except Exception as e:
        logger.error(f"Error retrieving voices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve voices",
        ) from e


@router.get("/llm/extensions")
async def llm_get_extensions():
    """Get available TTS extensions - designed for LLM integration.

    Returns a simple list of available extensions with their display names.

    Example:
        GET /api/v1/llm/extensions
    """
    try:
        # Get extension_manager from FastAPI app state
    from src.api.router_utils import get_extension_manager
    extension_manager = get_extension_manager(request)

        extensions = []
        for name, ext in extension_manager.extensions.items():
            try:
                config = ext.get_config()
                # Handle both dict and object types for config
                if isinstance(config, dict):
                    ext_name = config.get("extension_name", name)
                    display_name = config.get("display_name", name)
                    description = config.get("description", "")
                    requires_audio = config.get("requires_audio_sample", False)
                else:
                    ext_name = getattr(config, "extension_name", name)
                    display_name = getattr(config, "display_name", name)
                    description = getattr(config, "description", "")
                    requires_audio = getattr(config, "requires_audio_sample", False) if hasattr(config, "requires_audio_sample") else False

                extensions.append({
                    "name": ext_name,
                    "display_name": display_name,
                    "description": description,
                    "requires_audio_sample": requires_audio,
                })
            except Exception as e:
                logger.warning(f"Error getting config for extension '{name}': {str(e)}")
                # Add minimal info even if config fails
                extensions.append({
                    "name": name,
                    "display_name": name,
                    "description": "TTS extension",
                    "requires_audio_sample": False,
                })

        logger.info(f"Retrieved {len(extensions)} extensions")
        return {"extensions": extensions}

    except Exception as e:
        logger.error(f"Error retrieving extensions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve extensions",
        ) from e


@router.get("/llm/health")
async def llm_health():
    """Health check for LLM integration.

    Simple endpoint to check if the TTS service is operational.

    Example:
        GET /api/v1/llm/health
    """
    # Get extension_manager from FastAPI app state
    from src.api.router_utils import get_extension_manager
    extension_manager = get_extension_manager(request)

    loaded_extensions = list(extension_manager.loaded_extensions.keys())

    return {
        "status": "healthy",
        "service": "Cozy TTS",
        "loaded_extensions": loaded_extensions,
        "total_extensions": len(extension_manager.extensions),
    }
