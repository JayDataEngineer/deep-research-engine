"""Shared configuration for LLM access.

Routes text LLM calls to the configured provider:
- Cloud models (deepseek/, openai/, etc.) → OpenRouter or direct API
- Local models → llama.cpp direct or LiteLLM proxy
"""

import os
from pathlib import Path

# Load .env from project root (before reading any env vars)
try:
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

# =============================================================================
# Cloud LLM Provider (OpenRouter)
# =============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# =============================================================================
# Local Inference (llama.cpp / LiteLLM)
# =============================================================================

# Primary: Model Orchestrator (direct, manages GPU lifecycle)
ORCHESTRATOR_BASE_URL = os.getenv("ORCHESTRATOR_BASE_URL", "http://localhost:18880/v1")
ORCHESTRATOR_API_KEY = os.getenv("ORCHESTRATOR_API_KEY", "sk-litellm-e2e-test-master-key")

# Fallback: LiteLLM proxy (for cloud models, Langfuse tracing)
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", ORCHESTRATOR_BASE_URL)
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", ORCHESTRATOR_API_KEY)

# Default models
DEFAULT_MODEL = os.getenv("MAIN_MODEL", "qwen35-35b-a3b-vision")
TOOL_MODEL = os.getenv("TOOL_MODEL", "qwen35-35b-a3b-vision")
# VLM defaults to the same vision model as INGESTION_MODEL (gemma4-e4b-vision-fast)
VISION_MODEL = os.getenv("VISION_MODEL", "gemma4-e4b-vision-fast")

# Ingestion-specific model (tiny model for NER, identity resolution)
# gemma4-e4b-vision-fast is ~6GB VRAM, no thinking mode — fast for text tasks
INGESTION_MODEL = os.getenv("INGESTION_MODEL", "gemma4-e4b-vision-fast")

# =============================================================================
# Provider Detection
# =============================================================================

# Prefixes that indicate cloud models → route to OpenRouter
_CLOUD_PREFIXES = ("deepseek/", "openai/", "anthropic/", "google/", "meta-llama/", "or-")


def is_cloud_model(model: str | None) -> bool:
    """Check if a model should route through OpenRouter."""
    if not model:
        return False
    return any(model.startswith(p) for p in _CLOUD_PREFIXES)


def get_llm_endpoint(model: str | None = None) -> tuple[str, str]:
    """Get (base_url, api_key) for the appropriate LLM provider.

    Returns:
        Tuple of (base_url, api_key) ready for OpenAI client.
    """
    m = model or DEFAULT_MODEL
    if is_cloud_model(m) and OPENROUTER_API_KEY:
        return (OPENROUTER_BASE_URL, OPENROUTER_API_KEY)
    return (LITELLM_BASE_URL, LITELLM_API_KEY)


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
    "INGESTION_MODEL",
    "LITELLM_API_KEY",
    "LITELLM_BASE_URL",
    "MODEL_MAPPING",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "ORCHESTRATOR_API_KEY",
    "ORCHESTRATOR_BASE_URL",
    "THINKING_TO_FAST",
    "TOOL_MODEL",
    "VISION_MODEL",
    "get_llm_endpoint",
    "is_cloud_model",
    "map_model_name",
]
