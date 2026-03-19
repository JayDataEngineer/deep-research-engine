"""
Unified TTS Configuration API.

Provides endpoints for managing TTS settings across the application:
- User preferences (extension, voice, personality)
- Emotion presets (IndexTTS only)
- Roleplay voice configurations
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.extensions.libraries.tts_engine.config_storage import (
    get_tts_config_storage,
    TTSUserConfig,
    EmotionPreset,
    RoleplayVoiceConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["tts-config"])


# ============================================================================
# Request/Response Models
# ============================================================================

class TTSUserConfigRequest(BaseModel):
    """Request model for user TTS config."""
    user_id: str = "default"
    active_extension: Optional[str] = None
    voice_id: Optional[str] = None
    personality: str = "neutral"
    emotion_config_id: Optional[str] = None
    autoplay: bool = False


class TTSUserConfigResponse(BaseModel):
    """Response model for user TTS config."""
    user_id: str
    active_extension: Optional[str] = None
    voice_id: Optional[str] = None
    personality: str = "neutral"
    emotion_config_id: Optional[str] = None
    autoplay: bool = False


class EmotionPresetRequest(BaseModel):
    """Request model for emotion preset."""
    name: str = Field(..., description="Unique preset name (no spaces)")
    display_name: str = Field(..., description="Display name for UI")
    description: str = ""
    chunking_strategy: str = "sentence_boundary"
    emotion_mode: str = "llm"
    prompt_template: str = "neutral"
    personality: str = "neutral"
    llm_temperature: float = 0.1
    general_prompt: Optional[str] = None
    character_prompt: Optional[str] = None
    voice_sample_path: Optional[str] = None
    include_previous: bool = False
    include_next: bool = False


class EmotionPresetResponse(BaseModel):
    """Response model for emotion preset."""
    name: str
    display_name: str
    description: str
    chunking_strategy: str
    emotion_mode: str
    prompt_template: str
    personality: str
    llm_temperature: float
    general_prompt: Optional[str] = None
    character_prompt: Optional[str] = None
    voice_sample_path: Optional[str] = None
    include_previous: bool
    include_next: bool
    created_at: str


class EmotionPresetsListResponse(BaseModel):
    """Response model for emotion presets list."""
    presets: list[EmotionPresetResponse]


class ApplyEmotionConfigResponse(BaseModel):
    """Response model for applying emotion config."""
    success: bool
    config_name: str
    message: str


# ============================================================================
# User Config Endpoints
# ============================================================================

@router.get("/user", response_model=TTSUserConfigResponse)
async def get_user_config(user_id: str = "default") -> TTSUserConfigResponse:
    """Get user TTS preferences."""
    storage = get_tts_config_storage()
    config = await storage.load_user_config(user_id)

    if config:
        return TTSUserConfigResponse(**config.to_dict())

    # Return default config if none exists
    return TTSUserConfigResponse(
        user_id=user_id,
        active_extension=None,
        voice_id=None,
        personality="neutral",
        emotion_config_id=None,
        autoplay=False
    )


@router.post("/user", response_model=TTSUserConfigResponse)
async def save_user_config(request: TTSUserConfigRequest) -> TTSUserConfigResponse:
    """Save user TTS preferences."""
    storage = get_tts_config_storage()
    config = TTSUserConfig(
        user_id=request.user_id,
        active_extension=request.active_extension,
        voice_id=request.voice_id,
        personality=request.personality,
        emotion_config_id=request.emotion_config_id,
        autoplay=request.autoplay
    )

    await storage.save_user_config(request.user_id, config)

    return TTSUserConfigResponse(**config.to_dict())


@router.delete("/user", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_config(user_id: str = "default"):
    """Delete user TTS preferences."""
    storage = get_tts_config_storage()
    await storage.delete_user_config(user_id)


# ============================================================================
# Emotion Preset Endpoints (IndexTTS only)
# ============================================================================

@router.get("/emotion-presets", response_model=EmotionPresetsListResponse)
async def list_emotion_presets() -> EmotionPresetsListResponse:
    """List all IndexTTS emotion tagging presets."""
    storage = get_tts_config_storage()
    presets = await storage.list_emotion_presets()

    return EmotionPresetsListResponse(
        presets=[EmotionPresetResponse(**p.to_dict()) for p in presets]
    )


@router.get("/emotion-presets/{name}", response_model=EmotionPresetResponse)
async def get_emotion_preset(name: str) -> EmotionPresetResponse:
    """Get a specific IndexTTS emotion tagging preset."""
    storage = get_tts_config_storage()
    preset = await storage.load_emotion_preset(name)

    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emotion preset '{name}' not found"
        )

    return EmotionPresetResponse(**preset.to_dict())


@router.post("/emotion-presets", response_model=EmotionPresetResponse)
async def create_emotion_preset(request: EmotionPresetRequest) -> EmotionPresetResponse:
    """Create a new IndexTTS emotion tagging preset."""
    storage = get_tts_config_storage()
    preset = EmotionPreset(**request.model_dump())

    await storage.save_emotion_preset(preset)

    return EmotionPresetResponse(**preset.to_dict())


@router.delete("/emotion-presets/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_emotion_preset(name: str):
    """Delete an IndexTTS emotion tagging preset."""
    storage = get_tts_config_storage()
    deleted = await storage.delete_emotion_preset(name)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emotion preset '{name}' not found"
        )


# ============================================================================
# Apply Emotion Config (for auto-apply from emotion-tagging page)
# ============================================================================

@router.post("/emotion-presets/{name}/apply", response_model=ApplyEmotionConfigResponse)
async def apply_emotion_config(name: str, user_id: str = "default") -> ApplyEmotionConfigResponse:
    """
    Apply an emotion config as the active preset for IndexTTS.

    This creates a link between emotion-tagging experiments and Voice Settings.
    When a user saves a config in emotion-tagging, it can be auto-applied here.
    """
    storage = get_tts_config_storage()

    # Load the emotion preset
    preset = await storage.load_emotion_preset(name)
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emotion preset '{name}' not found"
        )

    # Update user config to reference this emotion preset
    user_config = await storage.load_user_config(user_id)
    if not user_config:
        user_config = TTSUserConfig(user_id=user_id)

    user_config.emotion_config_id = name
    user_config.personality = preset.personality

    # Also set active_extension to index_tts if emotion config is being used
    if user_config.active_extension is None:
        user_config.active_extension = "index_tts"

    await storage.save_user_config(user_id, user_config)

    logger.info(f"Applied emotion config '{name}' for user '{user_id}'")

    return ApplyEmotionConfigResponse(
        success=True,
        config_name=name,
        message=f"Emotion config '{name}' is now active"
    )
