"""Shared configuration for LLM access via LiteLLM.

This module provides configuration constants for connecting to LiteLLM proxy.
All LLM calls are routed through LiteLLM for:
- Unified model routing
- Automatic Langfuse tracing
- Rate limiting
"""

import os

# =============================================================================
# Core LiteLLM Configuration
# =============================================================================

# Primary: Model Orchestrator (direct, manages GPU lifecycle)
ORCHESTRATOR_BASE_URL = os.getenv("ORCHESTRATOR_BASE_URL", "http://localhost:18880/v1")
ORCHESTRATOR_API_KEY = os.getenv("ORCHESTRATOR_API_KEY", "not-needed")

# Fallback: LiteLLM proxy (for cloud models, Langfuse tracing)
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", ORCHESTRATOR_BASE_URL)
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", ORCHESTRATOR_API_KEY)

# Default models
DEFAULT_MODEL = os.getenv("MAIN_MODEL", "qwen35-35b-a3b-vision")
TOOL_MODEL = os.getenv("TOOL_MODEL", "qwen35-35b-a3b-vision")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen35-35b-a3b-vision")

# Ingestion-specific model (tiny model for NER, identity resolution, VLM)
# Defaults to gemma-4-e4b — a small model optimized for fast ingestion tasks
INGESTION_MODEL = os.getenv("INGESTION_MODEL", "gemma-4-e4b")


# =============================================================================
# Model Name Mappings
# =============================================================================

# Model name mapping (old/legacy -> LiteLLM)
MODEL_MAPPING: dict[str, str] = {
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
THINKING_TO_FAST: dict[str, str] = {
    "qwen-35-27": "qwen-35-27-fast",
    "qwen-35-27-vision": "qwen-35-27-vision-fast",
    "qwen-35-35b-a3b-vision": "qwen-35-35b-a3b-vision-fast",
}


# =============================================================================
# Helper Functions
# =============================================================================


def map_model_name(model: str | None, enable_thinking: bool = True) -> str:
    """Map old model names to LiteLLM names.

    Args:
        model: Original model name
        enable_thinking: Whether thinking mode is enabled

    Returns:
        LiteLLM-compatible model name
    """
    if not model:
        return DEFAULT_MODEL

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


__all__ = [
    "DEFAULT_MODEL",
    "LITELLM_API_KEY",
    "LITELLM_BASE_URL",
    "MODEL_MAPPING",
    "ORCHESTRATOR_API_KEY",
    "ORCHESTRATOR_BASE_URL",
    "THINKING_TO_FAST",
    "TOOL_MODEL",
    "VISION_MODEL",
    "map_model_name",
]
