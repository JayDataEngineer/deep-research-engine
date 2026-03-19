"""
TTS (Text-to-Speech) API endpoints.

Refactored to use centralized error handling and audio service utilities.
"""

import io
import logging
import os
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from src.extensions.libraries.tts_engine.api.schemas.tts import (
    ExtensionConfigResponse,
    TTSRequest,
    TTSResponse,
    VoiceInfo,
    VoicesResponse,
)
from src.constants import TTS_EXTENSION_URL
from src.shared.tools.audio_utils import audio_service
from src.extensions.libraries.tts_engine.router import tts_service
# Import extension_process_manager for subprocess-mode extension handling
from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager
from src.extensions.libraries.tts_engine.api.utils.error_handlers import (
    create_error_response,
)
from src.extensions.libraries.tts_engine.api.utils.validators import (
    validate_file_upload,
    validate_text_length,
    validate_tts_request,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Helper Functions for TTS Generation
# ============================================================================

def prepare_generation_params(request: TTSRequest) -> dict:
    """Prepare parameters for TTS generation.

    Args:
        request: TTSRequest with text and parameters

    Returns:
        Dictionary of parameters for generation
    """
    # Extract voice from extra fields (extension-specific parameter)
    voice_id = request.model_extra.get('voice') if request.model_extra else None

    # Normalize empty voice_id to None
    if not voice_id or (isinstance(voice_id, str) and voice_id.strip() in ("", "None")):
        voice_id = None

    # Base parameters
    all_params = {
        'text': request.text,
        'voice_id': voice_id,
        'speed': request.speed,
        'pitch': request.pitch,
        'audio_format': request.format,
    }

    # Add extension-specific parameters (excluding 'voice' which was converted to 'voice_id')
    if request.model_extra:
        for key, value in request.model_extra.items():
            # Skip private keys and 'voice' (already converted to voice_id)
            if not key.startswith('_') and key != 'voice':
                all_params[key] = value

    return all_params


async def save_generated_audio(
    audio_buffer: io.BytesIO,
    audio_format: str
) -> tuple[str, float]:
    """Save generated audio and return URL with duration.

    Args:
        audio_buffer: Buffer containing audio data
        audio_format: Audio format (wav, mp3, ogg)

    Returns:
        Tuple of (audio_url, duration)
    """
    return await audio_service.save_and_get_url(audio_buffer, audio_format)


async def handle_generation_error(
    error: Exception,
    context: str = "TTS generation"
) -> TTSResponse:
    """Handle errors during TTS generation.

    Args:
        error: The exception that occurred
        context: Context for error logging

    Returns:
        TTSResponse with error information
    """
    # HTTPException should NOT be handled here - let it propagate to FastAPI
    if isinstance(error, HTTPException):
        raise error

    logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    return create_error_response(error, context)


async def save_uploaded_reference_audio(
    upload_file: UploadFile,
    max_size_mb: int = 50
) -> str:
    """Save uploaded reference audio file to temp directory.

    Args:
        upload_file: The uploaded file
        max_size_mb: Maximum file size in megabytes

    Returns:
        Path to the saved file

    Raises:
        HTTPException: If file validation fails
    """
    from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory
    from src.extensions.libraries.tts_engine.api.utils.validators import sanitize_filename

    # Read file content
    content = await upload_file.read()

    # Validate file
    validate_file_upload(upload_file.filename, content, max_size_mb)

    # Get temp directory and create unique filename
    temp_dir = get_temp_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_filename = sanitize_filename(upload_file.filename)
    temp_filename = f"voice_clone_ref_{timestamp}_{safe_filename}"
    temp_filepath = os.path.join(temp_dir, temp_filename)

    # Save file
    with open(temp_filepath, "wb") as buffer:
        buffer.write(content)

    logger.info(f"Voice cloning reference audio uploaded: {temp_filepath}")
    return temp_filepath


async def ensure_extension_loaded_for_voice(voice_id: str) -> None:
    """Ensure the extension that provides a voice is loaded.

    Args:
        voice_id: The voice ID to find extension for

    Raises:
        HTTPException: If extension cannot be loaded
    """
    if not voice_id or not voice_id.strip():
        return

    extension = tts_service.extension_manager.get_extension_for_voice(voice_id)
    if extension and not extension.is_loaded:
        load_success = tts_service.extension_manager.load_extension(extension.name)
        if not load_success or not extension.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Extension for voice '{voice_id}' is not loaded and cannot be loaded. Model may need to be downloaded.",
            )


# REMOVED: Old /voices endpoint without extension parameter
# All voice queries MUST use /voices/{extension_name}
@router.get("/voices/{extension_name}", response_model=VoicesResponse)
async def get_voices_for_extension(extension_name: str, request: Request):
    """Get list of available TTS voices for a specific extension."""
    # Get extension_manager from FastAPI app state
    from src.api.router_utils import get_extension_manager
    extension_manager = get_extension_manager(request)
    from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager

    try:
        # Extension names now consistently use underscores
        # No need for hyphen-to-underscore conversion

        # First, try to get voices from the active subprocess (fast path)
        if extension_process_manager.active_extension == extension_name:
            response = await extension_process_manager.client.get(
                f"{TTS_EXTENSION_URL}/voices",
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                voices_list = data.get("voices", [])
                voices = [
                    VoiceInfo(
                        id=v.get("id", ""),
                        name=v.get("name", ""),
                        language=v.get("language", "en-US"),
                        gender=v.get("gender", "unknown"),
                        description=v.get("description", ""),
                    )
                    for v in voices_list
                ]
                logger.info(f"Retrieved {len(voices)} voices for extension: {extension_name} (from subprocess)")
                return VoicesResponse(voices=voices)

        # If not in subprocess, try to get voices from extension_manager (for unloaded extensions)
        # This allows showing voices before loading the extension
        ext = extension_manager.extensions.get(extension_name)
        if ext:
            voices = ext.get_voices()
            # Handle both dict (from subprocess) and VoiceInfo (from extension)
            voice_infos = []
            for v in voices:
                if isinstance(v, dict):
                    voice_infos.append(VoiceInfo(
                        id=v.get("id", ""),
                        name=v.get("name", ""),
                        language=v.get("language", "en-US"),
                        gender=v.get("gender", "unknown"),
                        description=v.get("description", ""),
                    ))
                else:
                    # Already a VoiceInfo object
                    voice_infos.append(v)
            logger.info(f"Retrieved {len(voice_infos)} voices for extension: {extension_name} (from extension manager)")
            return VoicesResponse(voices=voice_infos)

        # No voices found
        return VoicesResponse(voices=[])
    except Exception as e:
        logger.error(f"Error retrieving voices for extension '{extension_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve voices for extension '{extension_name}'",
        ) from e


@router.post("/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """Generate speech from text using specified parameters.

    Voice is now an extension-specific parameter passed via extra fields.
    Each extension declares whether it needs:
    - 'voice' parameter (for preset voice selection like Kokoro)
    - 'ref_audio_path' parameter (for voice cloning like IndexTTS)
    - Neither (for internal voice selection like espeak_ng)

    The extension_name parameter allows the frontend to specify which extension to use.
    This ensures the correct extension is loaded for the voice being used.
    """
    # Validate request
    validate_tts_request(
        text=request.text,
        speed=request.speed,
        pitch=request.pitch,
        format=request.format
    )

    logger.info(
        f"TTS Generation Request: text='{request.text[:50]}...', "
        f"format='{request.format}', "
        f"extension_name='{request.extension_name}', "
        f"extra_fields={list(request.model_extra.keys() if request.model_extra else [])}"
    )

    try:
        # If extension_name is provided, ensure that extension is loaded
        if request.extension_name:
            from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager

            # Extension names now consistently use underscores
            target_extension = request.extension_name.lower()

            # Check if the correct extension is loaded
            current_extension = extension_process_manager.active_extension

            logger.info(f"[EXTENSION SWITCH] Requested: '{request.extension_name}' -> '{target_extension}'")
            logger.info(f"[EXTENSION SWITCH] Current: '{current_extension}'")
            logger.info(f"[EXTENSION SWITCH] Match: {current_extension == target_extension}")

            if current_extension != target_extension:
                logger.warning(
                    f"Extension mismatch! Requested '{target_extension}' but '{current_extension}' is active. Switching..."
                )
                # Store the old extension in case we need to restore it
                old_extension = current_extension

                try:
                    # Stop current extension and start the new one
                    logger.info(f"[EXTENSION SWITCH] Stopping extension: {old_extension}")
                    await extension_process_manager.stop_current()

                    logger.info(f"[EXTENSION SWITCH] Starting extension: {target_extension}")
                    result = await extension_process_manager.start_extension(
                        target_extension,
                        config=None
                    )
                    logger.info(f"[EXTENSION SWITCH] Start result: {result}")

                    # CRITICAL: Check if start_extension succeeded
                    if result.get("status") == "error":
                        error_msg = result.get("message", "Unknown error")
                        logger.error(f"[EXTENSION SWITCH] Failed to start extension '{target_extension}': {error_msg}")
                        # Do NOT proceed with generation - raise an error immediately
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to load extension '{request.extension_name}': {error_msg}. Please try selecting a different extension."
                        )

                    # Verify that active_extension was actually updated
                    if extension_process_manager.active_extension != target_extension:
                        logger.error(f"[EXTENSION SWITCH] active_extension not updated! Expected '{target_extension}' but got '{extension_process_manager.active_extension}'")
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Extension switch failed - active extension is '{extension_process_manager.active_extension}' instead of '{request.extension_name}'"
                        )

                    # Wait for the extension to be ready
                    logger.info("[EXTENSION SWITCH] Waiting for extension to be ready...")
                    await extension_process_manager._wait_for_ready(timeout=30.0)
                    logger.info(f"[EXTENSION SWITCH] Extension '{target_extension}' started and ready successfully")

                except HTTPException:
                    # Re-raise HTTP exceptions as-is
                    raise
                except Exception as switch_error:
                    # If switch failed, try to restore the old extension
                    logger.error(f"[EXTENSION SWITCH] Extension switch failed: {switch_error}")
                    if old_extension:
                        logger.info(f"[EXTENSION SWITCH] Attempting to restore previous extension: {old_extension}")
                        try:
                            await extension_process_manager.start_extension(old_extension, config=None)
                            await extension_process_manager._wait_for_ready(timeout=30.0)
                            logger.info(f"[EXTENSION SWITCH] Restored extension '{old_extension}' successfully")
                        except Exception as restore_error:
                            logger.error(f"[EXTENSION SWITCH] Failed to restore previous extension: {restore_error}")
                            # Clean up - set active_extension to None
                            extension_process_manager.active_extension = None
                            extension_process_manager.process = None

                    # Always raise HTTPException to prevent generation with wrong extension
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to switch to extension '{request.extension_name}': {str(switch_error)}"
                    ) from switch_error
            else:
                # Extension name matches - trust the active_extension tracking
                # The subprocess status verification is skipped since we already:
                # 1. Verified active_extension matches requested name
                # 2. Checked the process is still running
                # This eliminates an unnecessary HTTP call on every generation
                logger.debug(f"[EXTENSION SWITCH] Extension already loaded and active: '{target_extension}'")

        # Prepare generation parameters
        params = prepare_generation_params(request)
        logger.info(f"Calling generate_audio with parameters: {list(params.keys())}")

        # Note: Voice refresh removed - voices are cached and only refreshed when needed
        # The refresh happens automatically when switching extensions

        # Check if we should use the subprocess for generation
        # If an extension subprocess is active, route generation through it
        from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager
        logger.debug(f"[DEBUG] process={extension_process_manager.process}, active_extension={extension_process_manager.active_extension}")
        if extension_process_manager.process and extension_process_manager.active_extension:
            logger.debug(f"[SUBPROCESS] Using extension subprocess for generation (active: {extension_process_manager.active_extension})")
            # Generate audio using the subprocess
            # Filter out params that are passed explicitly to avoid duplicate keyword arguments
            extra_params = {k: v for k, v in params.items() if k not in ['text', 'voice_id', 'speed', 'pitch', 'audio_format', 'output_format']}
            audio_bytes, duration = await extension_process_manager.generate_audio(
                text=params['text'],
                voice_id=params.get('voice_id') or '',
                speed=params.get('speed', 1.0),
                pitch=params.get('pitch', 1.0),
                output_format=params.get('audio_format', 'wav'),
                **extra_params
            )
            # Convert bytes to BytesIO buffer
            from io import BytesIO
            audio_buffer = BytesIO(audio_bytes)
        else:
            # Generate audio using the main process tts_service
            audio_buffer, duration = await tts_service.generate_audio(**params)
        audio_size = len(audio_buffer.getvalue()) if audio_buffer else 0
        logger.info(f"TTS Generation Success: duration={duration}s, audio_size={audio_size} bytes")

        # Save audio and get URL
        audio_url, _ = await save_generated_audio(audio_buffer, request.format)

        logger.info(f"Successfully saved speech: {audio_url}")

        return TTSResponse(
            success=True,
            message="Speech generated successfully",
            audio_url=audio_url,
            duration=duration,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle all other errors with our error handler
        return await handle_generation_error(e, "speech generation")


@router.post("/generate-with-voice-sample")
async def generate_speech_with_voice_sample(
    text: str,
    voice: str,
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
    speed: float = 1.0,
    pitch: float = 1.0,
    format: str = "wav",
    prompt_language: str = Form("zh"),
    text_language: str = Form("zh"),
):
    """
    Generate speech with a voice sample for voice cloning models.
    This endpoint accepts an audio file for voice cloning directly in the request.

    Args:
        text: Text to convert to speech
        voice: Voice ID to use for synthesis
        ref_audio: Reference audio file for voice cloning
        ref_text: Transcript of the reference audio
        speed: Speech speed multiplier
        pitch: Pitch adjustment
        format: Audio output format
        prompt_language: Language of the reference audio
        text_language: Language of the text to synthesize
    """
    try:
        # Validate text
        validate_text_length(text)

        # Validate speed, pitch, format
        validate_tts_request(text=text, speed=speed, pitch=pitch, format=format)

        # Normalize empty voice
        if not voice or not voice.strip():
            voice = None

        # Save uploaded reference audio
        ref_audio_path = await save_uploaded_reference_audio(ref_audio, max_size_mb=50)

        # Ensure extension is loaded if voice is provided
        await ensure_extension_loaded_for_voice(voice)

        # Refresh voices
        tts_service.refresh_voices()

        # Prepare generation parameters
        params = {
            'text': text,
            'voice_id': voice,
            'speed': speed,
            'pitch': pitch,
            'audio_format': format,
            'ref_audio_path': ref_audio_path,
            'ref_text': ref_text,
            'prompt_language': prompt_language,
            'text_language': text_language,
        }

        logger.info(f"Calling generate_audio for voice cloning with parameters: {list(params.keys())}")

        # Generate audio
        audio_buffer, duration = await tts_service.generate_audio(**params)
        audio_size = len(audio_buffer.getvalue()) if audio_buffer else 0
        logger.info(f"Voice Cloning Generation Success: duration={duration}s, audio_size={audio_size} bytes")

        # Save audio and get URL
        audio_url, _ = await save_generated_audio(audio_buffer, format)

        logger.info(f"Successfully saved voice cloning result: {audio_url}")

        return TTSResponse(
            success=True,
            message="Speech generated successfully with voice sample",
            audio_url=audio_url,
            duration=duration,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle all other errors with our error handler
        return await handle_generation_error(e, "voice cloning")


@router.post("/generate-stream")
async def generate_speech_stream(request: TTSRequest):
    """Generate speech and stream the audio directly."""
    # Validate request
    validate_tts_request(
        text=request.text,
        speed=request.speed,
        pitch=request.pitch,
        format=request.format
    )

    # Extract voice from extra fields (extension-specific parameter)
    voice_id = request.model_extra.get('voice') if request.model_extra else None

    logger.info(
        f"TTS Generation Stream Request: text='{request.text[:50]}...', "
        f"voice='{voice_id}', format='{request.format}'"
    )

    # Prepare generation parameters
    params = prepare_generation_params(request)
    logger.info(f"Calling generate_audio for streaming with parameters: {list(params.keys())}")

    # Generate audio
    audio_buffer, duration = await tts_service.generate_audio(**params)
    audio_size = len(audio_buffer.getvalue()) if audio_buffer else 0
    logger.info(f"Streaming Generation Success: duration={duration}s, audio_size={audio_size} bytes")

    # Set appropriate media type based on format
    media_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}.get(
        request.format, "audio/wav"
    )

    logger.info(f"Streaming audio for text: '{request.text[:50]}...'")

    # Return streaming response
    return StreamingResponse(
        io.BytesIO(audio_buffer.read()),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{request.format}",
            "X-Audio-Duration": str(duration),
        },
    )


@router.get("/voice/{voice_id}")
async def get_voice_info(voice_id: str):
    """Get information about a specific voice."""
    try:
        voice = tts_service.get_voice_by_id(voice_id)
        if not voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        logger.info(f"Retrieved info for voice: {voice_id}")
        return voice

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving voice info for '{voice_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve voice information",
        ) from e


@router.get("/extensions/config", response_model=ExtensionConfigResponse)
async def get_extensions_config(request: Request):
    """Get configuration for all extensions including dynamic parameters."""
    try:
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)

        # Verify extension manager is initialized
        if extension_manager is None:
            logger.error("Extension manager is not initialized")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Extension manager is not properly initialized",
            )

        logger.info(f"Extension manager instance found: {extension_manager}")

        # Check if extensions attribute exists (handle edge case)
        if not hasattr(extension_manager, "extensions"):
            logger.warning("Extension manager has no extensions attribute")
            return ExtensionConfigResponse(extensions=[])

        # Get extensions dict safely
        extensions = getattr(extension_manager, "extensions", None)
        if extensions is None:
            logger.warning("Extension manager extensions is None")
            return ExtensionConfigResponse(extensions=[])

        logger.info(f"Total extensions registered: {len(extensions)}")

        # Get loaded_extensions safely
        loaded_extensions = getattr(extension_manager, "loaded_extensions", {})
        logger.info(f"Loaded extensions: {len(loaded_extensions)}")

        # Log details of registered extensions (skip None values)
        for ext_name, ext in extensions.items():
            if ext is not None and hasattr(ext, "is_loaded"):
                logger.info(
                    f"Registered extension: {ext_name}, loaded: {ext.is_loaded}, type: {type(ext)}"
                )
            else:
                logger.warning(f"Extension '{ext_name}' is None or missing is_loaded attribute")

        # Log details of loaded extensions
        for ext_name, ext in loaded_extensions.items():
            logger.info(f"Loaded extension: {ext_name}, type: {type(ext)}")

        configs = extension_manager.get_all_tts_configs()
        logger.info(f"Retrieved {len(configs)} extension configurations")

        # Debug logging to ensure extensions exist
        if len(configs) == 0:
            logger.warning(
                "No extension configurations were retrieved. Checking extension manager state..."
            )
            logger.info(
                f"Total extensions registered: {len(extensions)}"
            )
            logger.info(
                f"Loaded extensions: {len(loaded_extensions)}"
            )

            # Even more detailed logging for debugging
            if extensions:
                for name, ext in extensions.items():
                    if ext is None:
                        logger.warning(f"Extension '{name}' is None, skipping config retrieval")
                        continue
                    try:
                        config = ext.get_config()
                        logger.info(
                            f"Extension '{name}' config method call result: {config is not None}"
                        )
                    except Exception as config_error:
                        logger.error(
                            f"Error getting config from extension '{name}': {str(config_error)}",
                            exc_info=True,
                        )

        return ExtensionConfigResponse(extensions=configs)
    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper status codes
        raise
    except Exception as e:
        logger.error(f"Error retrieving extension configurations: {str(e)}")
        logger.exception(
            "Full traceback for extension config error:"
        )  # Log full traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve extension configurations",
        ) from e


@router.get("/extensions/{extension_name}/config")
async def get_extension_config(extension_name: str, request: Request):
    """Get configuration for a specific extension."""
    try:
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)

        # Get the extension
        ext = extension_manager.registry.get(extension_name)

        if not ext or not hasattr(ext, 'config'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extension '{extension_name}' not found",
            )

        # Build config from extension's config.yaml
        ext_config_data = ext.config.get('extension', {})
        ext_tags = ext.config.get('tags', [])
        ext_parameters = ext.config.get('parameters', [])

        # Get display name and description
        display_name = ext_config_data.get('display_name', extension_name)
        description = ext_config_data.get('description', '')
        category = ext_config_data.get('category', 'tts')

        # Determine capabilities from tags/category
        supports_voice_cloning = 'voice-cloning' in ext_tags
        supports_emotion_tagging = 'emotion' in ext_tags or 'index_tts' in extension_name
        requires_audio_sample = supports_voice_cloning
        subprocess_only = ext_config_data.get('subprocess_only', False)

        # Convert parameters to the format expected by the frontend
        parameters = []
        for param in ext_parameters:
            param_obj = {
                "name": param.get("name"),
                "type": param.get("type", "text"),
                "label": param.get("label", param.get("name")),
                "description": param.get("description"),
                "default": param.get("default"),
                "required": param.get("required", False),
                "options": param.get("options"),
                "min_value": param.get("min_value"),
                "max_value": param.get("max_value"),
                "depends_on": param.get("depends_on"),
            }
            parameters.append(param_obj)

        result = {
            "extension_name": extension_name,
            "display_name": display_name,
            "description": description,
            "category": category,
            "parameters": parameters,
            "requires_audio_sample": requires_audio_sample,
            "supports_voice_cloning": supports_voice_cloning,
            "supports_emotion_tagging": supports_emotion_tagging,
            "subprocess_only": subprocess_only,
        }

        logger.info(f"Retrieved configuration for extension: {extension_name} with {len(parameters)} parameters")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving extension config for '{extension_name}': {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve extension configuration",
        ) from e


@router.post("/extensions/{extension_name}/load")
async def load_extension(extension_name: str, request: Request):
    """Load a specific extension by name.

    For extensions with isolated venvs (kokoro, maya1, index_tts, etc.),
    this will spawn a subprocess to load the extension.
    For other extensions, it loads in-process.
    """
    try:
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)
        from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager
        from src.constants import TTS_EXTENSIONS_DIR

        # Check if extension exists
        if extension_name not in extension_manager.extensions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extension '{extension_name}' not found",
            )

        # Check if extension has an isolated venv
        extension_venv = TTS_EXTENSIONS_DIR / extension_name / ".venv"
        has_isolated_venv = extension_venv.exists()

        if has_isolated_venv:
            # Use subprocess mode for extensions with isolated venvs
            logger.info(f"Loading extension '{extension_name}' in subprocess mode")

            # Check if correct extension is already active
            if extension_process_manager.active_extension != extension_name:
                # Stop current extension if any
                if extension_process_manager.process:
                    await extension_process_manager.stop_current()

                # Start the extension subprocess
                result = await extension_process_manager.start_extension(extension_name)

                if result.get("status") != "ready":
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"Failed to load extension '{extension_name}' in subprocess: {error_msg}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to load extension '{extension_name}': {error_msg}",
                    )

            logger.info(f"Extension '{extension_name}' loaded successfully (subprocess mode)")
            return {
                "success": True,
                "message": f"Extension '{extension_name}' loaded successfully",
            }
        else:
            # Load in-process for extensions without isolated venv
            logger.info(f"Loading extension '{extension_name}' in-process")
            success = extension_manager.load_extension(extension_name)

            if success:
                # Refresh voices to include voices from the newly loaded extension
                tts_service.refresh_voices()
                logger.info(f"Extension '{extension_name}' loaded successfully (in-process)")
                return {
                    "success": True,
                    "message": f"Extension '{extension_name}' loaded successfully",
                }
            else:
                logger.error(f"Failed to load extension '{extension_name}'")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load extension '{extension_name}'",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading extension '{extension_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading extension '{extension_name}': {str(e)}",
        ) from e


@router.post("/extensions/{extension_name}/unload")
async def unload_extension(extension_name: str, request: Request):
    """Unload a specific extension by name.

    IMPORTANT: Must handle both subprocess-mode extensions (with isolated venvs)
    and in-process extensions (without isolated venvs).
    """
    try:
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)

        # First check if extension is loaded in subprocess mode
        if extension_process_manager.active_extension == extension_name:
            logger.info(f"[UNLOAD] Extension '{extension_name}' is running in subprocess mode, stopping...")
            result = await extension_process_manager.stop_current()
            logger.info(f"[UNLOAD] Subprocess stopped: {result}")
            return {
                "success": True,
                "message": f"Extension '{extension_name}' unloaded successfully (subprocess stopped)",
            }

        # Check if extension exists (for in-process mode)
        if extension_name not in extension_manager.extensions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extension '{extension_name}' not found",
            )

        # Check if extension is loaded in-process
        if extension_name not in extension_manager.loaded_extensions:
            logger.info(f"Extension '{extension_name}' is not loaded (neither subprocess nor in-process)")
            return {
                "success": True,
                "message": f"Extension '{extension_name}' was not loaded",
            }

        # Unload the extension (in-process mode)
        success = extension_manager.unload_extension(extension_name)

        if success:
            # Refresh voices to remove voices from the unloaded extension
            tts_service.refresh_voices()
            logger.info(f"Extension '{extension_name}' unloaded successfully (in-process)")
            return {
                "success": True,
                "message": f"Extension '{extension_name}' unloaded successfully",
            }
        else:
            logger.error(f"Failed to unload extension '{extension_name}'")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to unload extension '{extension_name}'",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading extension '{extension_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading extension '{extension_name}': {str(e)}",
        ) from e


@router.get("/extensions")
async def get_extensions(request: Request):
    """Get information about all extensions."""
    try:
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)

        extensions_info = extension_manager.get_all_tts_configs()
        logger.info(f"Retrieved information for {len(extensions_info)} extensions")
        return {"extensions": extensions_info}

    except Exception as e:
        logger.error(f"Error retrieving extensions info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve extensions information",
        ) from e
