"""Configuration management for the application."""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from dotenv import load_dotenv

# Import constants - handle both module and direct import
try:
    from src.constants import (
        BACKEND_ROOT, PROJECT_ROOT, DATA_INPUTS_DIR, DATA_OUTPUTS_DIR,
        ROOT_DATA_DIR, ROOT_MODELS_DIR, DEFAULT_PORT, DEFAULT_HOST, ENV_HOST, ENV_PORT,
        ENV_LLM_API_KEY, ENV_LLM_BASE_URL, ENV_LLM_MODEL,
        DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL,
        DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PROVIDER,
        ENV_EMBEDDING_MODEL, ENV_EMBEDDING_PROVIDER,
        DEFAULT_CORS_ORIGINS, DEFAULT_REQUEST_TIMEOUT,
        DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P,
        MAX_UPLOAD_SIZE, SUPPORTED_AUDIO_FORMATS, SUPPORTED_IMAGE_FORMATS,
        SUPPORTED_VIDEO_FORMATS, DEFAULT_AUDIO_VOICE, DEFAULT_AUDIO_FORMAT,
        DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_STEPS,
        ENV_COMFYUI_ADDRESS, DEFAULT_COMFYUI_URL, ENV_UPLOAD_DIR,
        ENV_MODEL_DIR, ENV_CORS_ORIGINS,
    )
except ImportError:
    # Fallback for when running from backend directory
    # Add project root to path if not already there
    backend_src = Path(__file__).parent
    project_root = backend_src.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.constants import (
        BACKEND_ROOT, PROJECT_ROOT, DATA_INPUTS_DIR, DATA_OUTPUTS_DIR,
        ROOT_DATA_DIR, ROOT_MODELS_DIR, DEFAULT_PORT, DEFAULT_HOST, ENV_HOST, ENV_PORT,
        ENV_LLM_API_KEY, ENV_LLM_BASE_URL, ENV_LLM_MODEL,
        DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL,
        DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PROVIDER,
        ENV_EMBEDDING_MODEL, ENV_EMBEDDING_PROVIDER,
        DEFAULT_CORS_ORIGINS, DEFAULT_REQUEST_TIMEOUT,
        DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P,
        MAX_UPLOAD_SIZE, SUPPORTED_AUDIO_FORMATS, SUPPORTED_IMAGE_FORMATS,
        SUPPORTED_VIDEO_FORMATS, DEFAULT_AUDIO_VOICE, DEFAULT_AUDIO_FORMAT,
        DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_STEPS,
        ENV_COMFYUI_ADDRESS, DEFAULT_COMFYUI_URL, ENV_UPLOAD_DIR,
        ENV_MODEL_DIR, ENV_CORS_ORIGINS,
    )

load_dotenv()


# =============================================================================
# Configuration Base Classes
# =============================================================================

class StorageDirsMixin:
    """Mixin for storage directory properties.

    Provides common storage directory paths used across multiple
    configuration classes (Settings, TTS services, etc.).
    """

    def _get_base_output_dir(self) -> Path:
        """Get the base output directory."""
        return get_output_dir()

    @property
    def storage_base_dir(self) -> Path:
        """Base storage directory."""
        return self._get_base_output_dir()

    @property
    def user_directory(self) -> Path:
        """User directory for user-specific data."""
        return self._get_base_output_dir() / "users"

    @property
    def output_directory(self) -> Path:
        """Output directory for generated audio files."""
        return self._get_base_output_dir() / "audio"

    @property
    def temp_directory(self) -> Path:
        """Temp directory for temporary files."""
        return self._get_base_output_dir() / "temp"


# =============================================================================
# Unified Input/Output Directories
# =============================================================================

def get_inputs_dir() -> Path:
    """Get the unified input directory (data/inputs/)."""
    return DATA_INPUTS_DIR


def get_output_dir() -> Path:
    """Get the unified output directory (data/outputs/)."""
    return DATA_OUTPUTS_DIR


def get_data_dir() -> Path:
    """Get the unified data directory (root level data/)."""
    return ROOT_DATA_DIR


def get_model_path(model_name: str) -> Path | None:
    """Get the absolute path for a model directory.

    Args:
        model_name: Name of the model (e.g., "indextts", "vibevoice")

    Returns:
        Path to the model directory, or None if not found

    Note:
        Uses ROOT_MODELS_DIR from constants.py as the base.
        For cozytts models, use: get_model_path("cozytts/indextts")
    """
    # ROOT_MODELS_DIR is already imported from constants above
    # Support both "indextts" and "cozytts/indextts" formats
    model_path = ROOT_MODELS_DIR / model_name

    return model_path if model_path.exists() else None


class Settings(StorageDirsMixin):
    """Application settings.

    Note: Model routing is now handled by LiteLLM proxy.
    See src/shared/litellm_factory.py for LLM configuration.
    """

    # LLM Configuration (for backward compatibility, but prefer LiteLLM)
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv(ENV_LLM_BASE_URL, DEFAULT_LLM_BASE_URL)
    llm_model: str = os.getenv(ENV_LLM_MODEL, DEFAULT_LLM_MODEL)
    show_llm_config: bool = os.getenv("SHOW_LLM_CONFIG", "false").lower() == "true"

    # Embedding Configuration
    embedding_model: str = os.getenv(ENV_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL)
    embedding_provider: str = os.getenv(ENV_EMBEDDING_PROVIDER, DEFAULT_EMBEDDING_PROVIDER)

    # Audio Configuration
    default_audio_voice: str = os.getenv("DEFAULT_AUDIO_VOICE", DEFAULT_AUDIO_VOICE)
    default_modalities: str = os.getenv("DEFAULT_MODALITIES", "text,audio")
    audio_format: str = os.getenv("AUDIO_FORMAT", DEFAULT_AUDIO_FORMAT)

    # Upload Configuration
    upload_dir: Path = DATA_OUTPUTS_DIR / "uploads"
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(MAX_UPLOAD_SIZE)))

    # Server Configuration
    host: str = os.getenv(ENV_HOST, DEFAULT_HOST)
    port: int = int(os.getenv(ENV_PORT, str(DEFAULT_PORT)))
    cors_origins: List[str] = os.getenv(
        ENV_CORS_ORIGINS, DEFAULT_CORS_ORIGINS
    ).replace("'", '"')

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "info")
    log_json: bool = os.getenv("LOG_JSON", "true").lower() == "true"

    # Companion Brain Configuration
    enable_persistence: bool = os.getenv("ENABLE_PERSISTENCE", "true").lower() == "true"

    # IndexTTS Configuration
    indextts_url: str = os.getenv("INDEXTTS_URL", "")
    indextts_default_speaker: str = os.getenv("INDEXTTS_DEFAULT_SPEAKER", "")

    def __init__(self):
        """Create data directory structure."""
        # Use centralized directory initialization from helpers
        try:
            from .api.shared.helpers import initialize_data_directories
            initialize_data_directories()
        except ImportError:
            # Fallback if running as standalone module
            try:
                from src.shared.utils.api_helpers import initialize_data_directories
                initialize_data_directories()
            except ImportError:
                # Skip initialization if helpers not available
                pass


settings = Settings()
