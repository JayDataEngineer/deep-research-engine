"""Centralized LiteLLM Configuration.

This module provides a single source of truth for all LiteLLM-related constants
and configuration. All other modules should import from here instead of defining
their own constants.

Usage:
    from src.shared.litellm_config import (
        LITELLM_BASE_URL,
        LITELLM_API_KEY,
        LITELLM_TTS_URL,
        get_litellm_config,
    )
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# Core LiteLLM Configuration
# =============================================================================

# Main LiteLLM endpoint for LLM inference
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://litellm.local/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-litellm-master")

# TTS endpoint (same LiteLLM proxy, different endpoint)
LITELLM_TTS_URL = os.getenv("LITELLM_TTS_URL", "http://litellm.local/v1")

# Health check endpoints
LITELLM_HEALTH_URL = os.getenv("LITELLM_HEALTH_URL", "http://litellm.local/health")
LITELLM_MODELS_URL = os.getenv("LITELLM_MODELS_URL", "http://litellm.local/v1/models")


# =============================================================================
# Model Name Mappings
# =============================================================================

# Model name mapping (old/legacy -> LiteLLM)
MODEL_MAPPING: Dict[str, str] = {
    # ZAI Cloud models
    "glm-4.5": "zai-glm-4-5",
    "glm-4.5-air": "zai-glm-4-5-air",
    "glm-4.6": "zai-glm-4-6",
    "glm-4.7": "zai-glm-4-7",
    "glm-5": "zai-glm-5",
    # Local thinking models (via llama.cpp)
    "GLM-4.7-Flash-GGUF": "qwen-35-27",
    # Local fast models
    "glm-flash-*": "qwen-35-27-fast",
    "qwen-flash-*": "qwen-35-27-fast",
}

# Non-thinking variants (used when enable_thinking=False)
THINKING_TO_FAST: Dict[str, str] = {
    "qwen-35-27": "qwen-35-27-fast",
    "qwen-35-27-vision": "qwen-35-27-vision-fast",
}


# =============================================================================
# TTS Configuration
# =============================================================================

# Default TTS settings
DEFAULT_TTS_MODEL = os.getenv("TTS_MODEL", "kokoro")
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", "Cherry")
DEFAULT_TTS_FORMAT = os.getenv("TTS_FORMAT", "wav")  # wav, mp3, opus
DEFAULT_TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# Available TTS models via LiteLLM
AVAILABLE_TTS_MODELS: Dict[str, Dict[str, Any]] = {
    "kokoro": {
        "description": "Fast, lightweight TTS (82M parameters)",
        "voices": ["Cherry", "Serena", "Ethan", "Chelsie", "Jessica", "Daniel"],
        "speed_range": (0.5, 2.0),
        "quality": "good",
        "latency": "low",
    },
    "indextts2": {
        "description": "High-quality emotional TTS",
        "voices": ["female", "male"],
        "speed_range": (0.7, 1.5),
        "quality": "excellent",
        "latency": "medium",
    },
}


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class LiteLLMConfig:
    """LiteLLM configuration container.

    Attributes:
        base_url: LiteLLM base URL for LLM inference
        api_key: LiteLLM API key
        tts_url: LiteLLM URL for TTS
        health_url: Health check endpoint
        models_url: Models listing endpoint
        model_mapping: Model name mapping dictionary
        thinking_to_fast: Thinking to fast model mapping
        default_tts_model: Default TTS model name
        default_tts_voice: Default TTS voice
        default_tts_format: Default audio format
        default_tts_speed: Default TTS speed
        available_tts_models: Available TTS models info
    """

    base_url: str = field(default_factory=lambda: LITELLM_BASE_URL)
    api_key: str = field(default_factory=lambda: LITELLM_API_KEY)
    tts_url: str = field(default_factory=lambda: LITELLM_TTS_URL)
    health_url: str = field(default_factory=lambda: LITELLM_HEALTH_URL)
    models_url: str = field(default_factory=lambda: LITELLM_MODELS_URL)
    model_mapping: Dict[str, str] = field(default_factory=lambda: MODEL_MAPPING.copy())
    thinking_to_fast: Dict[str, str] = field(
        default_factory=lambda: THINKING_TO_FAST.copy()
    )
    default_tts_model: str = field(default_factory=lambda: DEFAULT_TTS_MODEL)
    default_tts_voice: str = field(default_factory=lambda: DEFAULT_TTS_VOICE)
    default_tts_format: str = field(default_factory=lambda: DEFAULT_TTS_FORMAT)
    default_tts_speed: float = field(default_factory=lambda: DEFAULT_TTS_SPEED)
    available_tts_models: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: AVAILABLE_TTS_MODELS.copy()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "tts_url": self.tts_url,
            "health_url": self.health_url,
            "models_url": self.models_url,
        }


# =============================================================================
# Helper Functions
# =============================================================================


def get_litellm_config() -> LiteLLMConfig:
    """Get LiteLLM configuration.

    Returns:
        LiteLLMConfig instance with current settings
    """
    return LiteLLMConfig()


def get_litellm_headers(include_auth: bool = True) -> Dict[str, str]:
    """Get headers for LiteLLM API requests.

    Args:
        include_auth: Whether to include Authorization header

    Returns:
        Dictionary of headers
    """
    headers = {"Content-Type": "application/json"}
    if include_auth:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"
    return headers


def map_model_name(model: str, enable_thinking: bool = True) -> str:
    """Map old model names to LiteLLM names.

    Args:
        model: Original model name
        enable_thinking: Whether thinking mode is enabled

    Returns:
        LiteLLM-compatible model name
    """
    mapped = model  # Default: use as-is

    # Check exact match first
    if model in MODEL_MAPPING:
        mapped = MODEL_MAPPING[model]

    # Check wildcard patterns
    if mapped == model:  # No exact match found
        for pattern, replacement in MODEL_MAPPING.items():
            if pattern.endswith("*") and model.startswith(pattern[:-1]):
                mapped = replacement
                break

    # Handle thinking mode
    if not enable_thinking and mapped in THINKING_TO_FAST:
        mapped = THINKING_TO_FAST[mapped]

    return mapped


# =============================================================================
# Export for backward compatibility
# =============================================================================

# These are exported for modules that directly import the constants
__all__ = [
    # Core constants
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    "LITELLM_TTS_URL",
    "LITELLM_HEALTH_URL",
    "LITELLM_MODELS_URL",
    # Model mappings
    "MODEL_MAPPING",
    "THINKING_TO_FAST",
    # TTS constants
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TTS_VOICE",
    "DEFAULT_TTS_FORMAT",
    "DEFAULT_TTS_SPEED",
    "AVAILABLE_TTS_MODELS",
    # Helper functions
    "get_litellm_config",
    "get_litellm_headers",
    "map_model_name",
    # Config class
    "LiteLLMConfig",
]
