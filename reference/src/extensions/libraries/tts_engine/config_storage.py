"""
Unified TTS Configuration Storage Service.

Centralizes storage for all TTS-related configurations including:
- User preferences (extension, voice, personality, etc.)
- Emotion tagging presets (IndexTTS only)
- Roleplay voice configurations
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================

try:
    from src.constants import ROOT_DATA_DIR
except ImportError:
    # Fallback: go up to project root (backend/ -> project root)
    ROOT_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"

TTS_CONFIG_DIR = ROOT_DATA_DIR / "config" / "tts"
USER_CONFIGS_DIR = TTS_CONFIG_DIR / "user_configs"
EMOTION_PRESETS_DIR = TTS_CONFIG_DIR / "emotion_presets"
ROLEPLAY_CONFIGS_DIR = TTS_CONFIG_DIR / "roleplay_configs"

# Create directories
for dir_path in [TTS_CONFIG_DIR, USER_CONFIGS_DIR, EMOTION_PRESETS_DIR, ROLEPLAY_CONFIGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Models
# ============================================================================

class TTSUserConfig:
    """User TTS preferences."""

    def __init__(
        self,
        user_id: str,
        active_extension: Optional[str] = None,
        voice_id: Optional[str] = None,
        personality: str = "neutral",
        emotion_config_id: Optional[str] = None,
        autoplay: bool = False,
        **kwargs
    ):
        self.user_id = user_id
        self.active_extension = active_extension
        self.voice_id = voice_id
        self.personality = personality
        self.emotion_config_id = emotion_config_id  # IndexTTS only
        self.autoplay = autoplay
        self.extra = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "active_extension": self.active_extension,
            "voice_id": self.voice_id,
            "personality": self.personality,
            "emotion_config_id": self.emotion_config_id,
            "autoplay": self.autoplay,
            **self.extra
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TTSUserConfig":
        return cls(**data)


class EmotionPreset:
    """Emotion tagging preset for IndexTTS."""

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str = "",
        chunking_strategy: str = "sentence_boundary",
        emotion_mode: str = "llm",
        prompt_template: str = "neutral",
        personality: str = "neutral",
        llm_temperature: float = 0.1,
        general_prompt: Optional[str] = None,
        character_prompt: Optional[str] = None,
        voice_sample_path: Optional[str] = None,
        include_previous: bool = False,
        include_next: bool = False,
        created_at: Optional[str] = None,
        **kwargs
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.chunking_strategy = chunking_strategy
        self.emotion_mode = emotion_mode
        self.prompt_template = prompt_template
        self.personality = personality
        self.llm_temperature = llm_temperature
        self.general_prompt = general_prompt
        self.character_prompt = character_prompt
        self.voice_sample_path = voice_sample_path
        self.include_previous = include_previous
        self.include_next = include_next
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.extra = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "chunking_strategy": self.chunking_strategy,
            "emotion_mode": self.emotion_mode,
            "prompt_template": self.prompt_template,
            "personality": self.personality,
            "llm_temperature": self.llm_temperature,
            "general_prompt": self.general_prompt,
            "character_prompt": self.character_prompt,
            "voice_sample_path": self.voice_sample_path,
            "include_previous": self.include_previous,
            "include_next": self.include_next,
            "created_at": self.created_at,
            **self.extra
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmotionPreset":
        return cls(**data)


class RoleplayVoiceConfig:
    """Voice configuration for roleplay mode."""

    def __init__(
        self,
        character_id: str,
        character_voice: Optional[dict[str, Any]] = None,
        narrator_voice: Optional[dict[str, Any]] = None,
        auto_switch: bool = False,
        **kwargs
    ):
        self.character_id = character_id
        self.character_voice = character_voice or {}
        self.narrator_voice = narrator_voice or {}
        self.auto_switch = auto_switch
        self.extra = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "character_voice": self.character_voice,
            "narrator_voice": self.narrator_voice,
            "auto_switch": self.auto_switch,
            **self.extra
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoleplayVoiceConfig":
        return cls(**data)


# ============================================================================
# Storage Service
# ============================================================================

class TTSConfigStorage:
    """Centralized storage for all TTS configurations."""

    # User Config Methods
    async def save_user_config(self, user_id: str, config: TTSUserConfig) -> str:
        """Save user TTS preferences."""
        config_path = USER_CONFIGS_DIR / f"{user_id}.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved user config for {user_id}")
        return str(config_path)

    async def load_user_config(self, user_id: str) -> Optional[TTSUserConfig]:
        """Load user TTS preferences."""
        config_path = USER_CONFIGS_DIR / f"{user_id}.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            return TTSUserConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load user config for {user_id}: {e}")
            return None

    async def delete_user_config(self, user_id: str) -> bool:
        """Delete user TTS preferences."""
        config_path = USER_CONFIGS_DIR / f"{user_id}.json"
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Deleted user config for {user_id}")
            return True
        return False

    # Emotion Preset Methods
    async def save_emotion_preset(self, preset: EmotionPreset) -> str:
        """Save emotion tagging preset (IndexTTS only)."""
        preset_path = EMOTION_PRESETS_DIR / f"{preset.name}.json"
        with open(preset_path, "w") as f:
            json.dump(preset.to_dict(), f, indent=2)
        logger.info(f"Saved emotion preset: {preset.name}")
        return str(preset_path)

    async def load_emotion_preset(self, name: str) -> Optional[EmotionPreset]:
        """Load emotion tagging preset."""
        preset_path = EMOTION_PRESETS_DIR / f"{name}.json"
        if not preset_path.exists():
            return None
        try:
            with open(preset_path, "r") as f:
                data = json.load(f)
            return EmotionPreset.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load emotion preset {name}: {e}")
            return None

    async def list_emotion_presets(self) -> list[EmotionPreset]:
        """List all emotion tagging presets."""
        presets = []
        for preset_file in EMOTION_PRESETS_DIR.glob("*.json"):
            try:
                with open(preset_file, "r") as f:
                    data = json.load(f)
                presets.append(EmotionPreset.from_dict(data))
            except Exception as e:
                logger.error(f"Failed to load preset {preset_file}: {e}")
        return presets

    async def delete_emotion_preset(self, name: str) -> bool:
        """Delete emotion tagging preset."""
        preset_path = EMOTION_PRESETS_DIR / f"{name}.json"
        if preset_path.exists():
            preset_path.unlink()
            logger.info(f"Deleted emotion preset: {name}")
            return True
        return False

    # Roleplay Config Methods
    async def save_roleplay_config(self, config: RoleplayVoiceConfig) -> str:
        """Save roleplay voice configuration."""
        config_path = ROLEPLAY_CONFIGS_DIR / f"{config.character_id}.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved roleplay config for {config.character_id}")
        return str(config_path)

    async def load_roleplay_config(self, character_id: str) -> Optional[RoleplayVoiceConfig]:
        """Load roleplay voice configuration."""
        config_path = ROLEPLAY_CONFIGS_DIR / f"{character_id}.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            return RoleplayVoiceConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load roleplay config for {character_id}: {e}")
            return None


# ============================================================================
# Singleton Instance
# ============================================================================

_storage_instance: Optional[TTSConfigStorage] = None


def get_tts_config_storage() -> TTSConfigStorage:
    """Get the singleton TTSConfigStorage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = TTSConfigStorage()
    return _storage_instance
