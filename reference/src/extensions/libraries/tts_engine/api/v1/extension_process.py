"""
API endpoints for managing extension subprocesses.

Provides endpoints for loading/unloading extensions and checking their status.
Each extension runs in its own subprocess with its own venv to avoid
dependency conflicts. Only ONE extension is active at a time on port TTS_EXTENSION_PORT.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.constants import TTS_EXTENSION_PORT
from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/extension-process", tags=["extension-process"])


# Request/Response Models
class LoadExtensionRequest(BaseModel):
    extension: str = Field(..., description="Extension name to load (e.g., index_tts, kokoro)")
    config: dict = Field(default_factory=dict, description="Optional config parameters to pass to the extension")


class LoadExtensionResponse(BaseModel):
    status: str = Field(..., description="Status: ready, error, loading")
    extension: str | None = None
    port: int = Field(default=TTS_EXTENSION_PORT, description=f"Service port (always {TTS_EXTENSION_PORT})")
    voices: int | None = None
    message: str | None = None


class StatusResponse(BaseModel):
    status: str = Field(..., description="Status: ready, loading, no_extension, error")
    active_extension: str | None = None
    voices: int | None = None
    model_loaded: bool = False  # Whether the TTS model is loaded
    service_ready: bool = False  # Whether the service is ready for requests


class GenerateRequest(BaseModel):
    text: str
    voice_id: str
    speed: float = 1.0
    pitch: float = 1.0
    output_format: str = "wav"
    extra_params: dict = Field(default_factory=dict)


class AvailableExtensionsResponse(BaseModel):
    active_extension: str | None
    available_extensions: list[str]


@router.post(
    "/load",
    response_model=LoadExtensionResponse,
    responses={
        500: {
            "description": "Extension failed to load",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "Failed to load extension"
                        }
                    }
                }
            }
        }
    }
)
async def load_extension(request: LoadExtensionRequest):
    """
    Load an extension in its own subprocess on port TTS_EXTENSION_PORT.

    This will:
    1. Clean up any leftover processes from previous runs (if this is the first load)
    2. Stop any currently active extension (frees GPU memory)
    3. Start a new subprocess for the requested extension with its venv
    4. Load the extension's model into GPU memory
    5. Return when the extension is ready (2-3 second load time)

    The extension will remain loaded until another extension is loaded
    or the extension is explicitly unloaded.

    Frontend should show a loading spinner during this operation.
    """
    logger.info(f"Loading extension: {request.extension}")

    # Clean up any leftover processes from previous runs
    # This is important if the backend was restarted while an extension was running
    await extension_process_manager.cleanup_leftover_processes()

    result = await extension_process_manager.start_extension(
        request.extension,
        config=request.config
    )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Failed to load extension")
        )

    return LoadExtensionResponse(**result)


@router.post("/unload", response_model=StatusResponse)
async def unload_extension():
    """
    Unload the current extension and free GPU memory.

    This stops the subprocess and frees all GPU memory.
    Also cleans up any orphaned processes from previous runs.
    """
    # Clean up any orphaned processes first (in case tracking is out of sync)
    await extension_process_manager.cleanup_leftover_processes()

    # Then stop the currently tracked extension
    result = await extension_process_manager.stop_current()
    return StatusResponse(
        status=result["status"],
        active_extension=None,
        voices=None
    )


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get the status of the active extension.

    Returns information about which extension is currently loaded,
    whether it's ready, and the voice count.
    """
    result = await extension_process_manager.get_status()
    return StatusResponse(**result)


@router.get("/extensions", response_model=AvailableExtensionsResponse)
async def get_available_extensions(request: Request):
    """
    Get the list of available TTS extensions and the currently active one.

    Filters to only return TTS extensions (those with 'tts' tag and subprocess_only=true).
    """
    try:
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)

        # Filter to only TTS extensions
        tts_extensions = []
        for name, ext in extension_manager.extensions.items():
            # Access extension config
            if hasattr(ext, 'config') and ext.config:
                ext_config = ext.config.get('extension', {})
                ext_tags = ext.config.get('tags', [])
                is_subprocess_only = ext_config.get('subprocess_only', False)

                # Include if it has 'tts' tag and is subprocess_only
                if 'tts' in ext_tags and is_subprocess_only:
                    tts_extensions.append(name)

        return AvailableExtensionsResponse(
            active_extension=extension_process_manager.active_extension,
            available_extensions=tts_extensions
        )
    except Exception as e:
        logger.error(f"[TTS EXTENSIONS] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting extensions: {str(e)}"
        )


@router.post("/generate")
async def generate_audio(request: GenerateRequest):
    """
    Generate audio using the active extension subprocess.

    Note: This endpoint requires an extension to be loaded first.
    """
    if not extension_process_manager.active_extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No extension is currently loaded. Call POST /load first."
        )

    try:
        # Build params, avoiding duplicate keys from extra_params
        params = {
            "text": request.text,
            "voice_id": request.voice_id,
            "speed": request.speed,
            "pitch": request.pitch,
            "output_format": request.output_format,
        }
        # Add extra_params, but don't override explicit params
        for key, value in request.extra_params.items():
            if key not in params:
                params[key] = value

        audio_bytes, duration = await extension_process_manager.generate_audio(**params)

        from fastapi.responses import Response
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Duration": str(duration),
                "X-Extension": extension_process_manager.active_extension,
                "Content-Disposition": f'attachment; filename="output.{request.output_format}"'
            }
        )
    except RuntimeError as e:
        logger.error(f"Error generating audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
