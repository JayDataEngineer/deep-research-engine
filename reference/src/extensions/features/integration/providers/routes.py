"""LLM Provider Discovery Routes.

Discovers available models from LiteLLM proxy.
All model routing is handled by LiteLLM configuration.

Endpoints:
- GET /api/v1/providers - List available providers (LiteLLM, local servers)
- GET /api/v1/providers/{provider_id}/models - Get models for a provider
- GET /api/v1/providers/validate/{model_id} - Validate a model ID
- GET /api/v1/providers/context/{model_id} - Get context window size
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional

import httpx
from fastapi import APIRouter, HTTPException

from src.shared.model_context import get_model_max_tokens
from src.shared.litellm_factory import LITELLM_BASE_URL, LITELLM_API_KEY
from src.services.api_keys import get_api_key_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/providers", tags=["providers"])


# Provider metadata for display
PROVIDER_METADATA = {
    "litellm": {
        "id": "litellm",
        "name": "LiteLLM Proxy",
        "description": "Unified LLM gateway (all configured models)",
        "icon": "🔌",
        "color": "bg-blue-600",
        "is_local": False,
        "is_running": True,  # Assumed running if we can reach it
    },
    "llamacpp": {
        "id": "llamacpp",
        "name": "llama.cpp",
        "description": "GGUF models (local)",
        "icon": "🦙",
        "color": "bg-green-500",
        "local": True,
        "default_port": 18734,
    },
    "ollama": {
        "id": "ollama",
        "name": "Ollama",
        "description": "Local LLM server",
        "icon": "🦙",
        "color": "bg-purple-500",
        "local": True,
        "default_port": 11434,
    },
}


async def check_local_server(base_url: str) -> bool:
    """Check if a local server is running."""
    from src.services.provider_checks import check_local_server as _check_local_server
    return await _check_local_server(base_url)


async def get_litellm_models() -> List[Dict[str, Any]]:
    """Fetch available models from LiteLLM proxy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{LITELLM_BASE_URL}/models",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
    except Exception as e:
        logger.debug(f"Failed to fetch LiteLLM models: {e}")
    return []


@router.get("")
async def list_providers() -> List[Dict[str, Any]]:
    """List all available LLM providers.

    Returns providers based on:
    1. LiteLLM proxy (primary - all configured models)
    2. Local servers (llama.cpp, Ollama) if detected

    Returns:
        List of provider configurations with metadata
    """
    providers: Dict[str, Dict[str, Any]] = {}

    # Check LiteLLM connectivity
    litellm_models = await get_litellm_models()
    if litellm_models:
        providers["litellm"] = {
            **PROVIDER_METADATA["litellm"],
            "model_count": len(litellm_models),
            "is_running": True,
        }

    # Check local llama.cpp (now using llama.local via shared infra)
    llamacpp_url = "http://llama.local/v1"
    is_running = await check_local_server(llamacpp_url)
    if is_running:
        providers["llamacpp"] = {
            **PROVIDER_METADATA["llamacpp"],
            "base_url": llamacpp_url,
            "is_local": True,
            "is_running": True,
            "has_api_key": True,
        }

    # Check Ollama
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    ollama_url = f"http://localhost:{ollama_port}/v1"
    is_running = await check_local_server(ollama_url)
    if is_running:
        providers["ollama"] = {
            **PROVIDER_METADATA["ollama"],
            "base_url": ollama_url,
            "is_local": True,
            "is_running": True,
            "has_api_key": True,
        }

    # Sort: local running first, then others
    provider_list = list(providers.values())

    def sort_key(p: Dict[str, Any]) -> tuple:
        if p.get("is_local") and p.get("is_running"):
            return (0, p.get("name", ""))
        else:
            return (1, p.get("name", ""))

    provider_list.sort(key=sort_key)
    return provider_list


@router.get("/{provider_id}/models")
async def get_provider_models(provider_id: str) -> Dict[str, Any]:
    """Get available models for a specific provider.

    Args:
        provider_id: The provider ID (litellm, llamacpp, ollama)

    Returns:
        Dict with models list and metadata
    """
    metadata = PROVIDER_METADATA.get(provider_id, {
        "id": provider_id,
        "name": provider_id.title(),
    })

    if provider_id == "litellm":
        # Fetch models from LiteLLM proxy
        models = await get_litellm_models()

        # Group by provider prefix (e.g., "zai-", "qwen-", etc.)
        model_list = []
        for model in models:
            model_id = model.get("id", "")
            model_list.append({
                "id": model_id,
                "name": model.get("name", model_id),
                "owned_by": model.get("owned_by", "unknown"),
            })

        return {
            "provider_id": provider_id,
            "provider_name": metadata.get("name", "LiteLLM"),
            "is_local": False,
            "is_running": True,
            "base_url": LITELLM_BASE_URL,
            "models": model_list,
        }

    elif provider_id == "llamacpp":
        # Fetch models from llama.cpp (via llama.local)
        llamacpp_url = "http://llama.local/v1"
        is_running = await check_local_server(llamacpp_url)
        models = []

        if is_running:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{llamacpp_url}/models")
                    if response.status_code == 200:
                        data = response.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "unknown")
                            models.append({
                                "id": model_id,
                                "name": model_id,
                            })
            except Exception as e:
                logger.debug(f"Failed to fetch llama.cpp models: {e}")

        return {
            "provider_id": provider_id,
            "provider_name": metadata.get("name", "llama.cpp"),
            "is_local": True,
            "is_running": is_running,
            "base_url": llamacpp_url,
            "models": models,
        }

    elif provider_id == "ollama":
        # Fetch models from Ollama
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        ollama_url = f"http://localhost:{ollama_port}"
        is_running = await check_local_server(f"{ollama_url}/v1")
        models = []

        if is_running:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{ollama_url}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        for model in data.get("models", []):
                            models.append({
                                "id": model.get("name"),
                                "name": model.get("name"),
                                "size": model.get("size", 0),
                            })
            except Exception as e:
                logger.debug(f"Failed to fetch Ollama models: {e}")

        return {
            "provider_id": provider_id,
            "provider_name": metadata.get("name", "Ollama"),
            "is_local": True,
            "is_running": is_running,
            "base_url": f"{ollama_url}/v1",
            "models": models,
        }

    else:
        raise HTTPException(
            status_code=404,
            detail=f"Provider {provider_id} not found"
        )


@router.get("/validate/{model_id}")
async def validate_model_endpoint(model_id: str) -> Dict[str, Any]:
    """Validate if a model ID is available.

    Checks if the model exists in LiteLLM configuration.

    Args:
        model_id: The model ID to validate

    Returns:
        Dict with validation result and metadata
    """
    if not model_id:
        return {
            "valid": False,
            "error": "Model ID is empty",
            "model_id": model_id,
        }

    # Check against LiteLLM models
    models = await get_litellm_models()
    model_ids = {m.get("id") for m in models}

    if model_id in model_ids:
        return {
            "valid": True,
            "model_id": model_id,
            "provider": "litellm",
            "base_url": LITELLM_BASE_URL,
        }

    # Check custom models stored in API keys database
    try:
        service = get_api_key_service()
        for key_type in ["openrouter", "anthropic", "openai", "zai", "google", "cohere"]:
            keys = await service.list_keys(key_type=key_type)
            for key in keys:
                # Skip auto-generated date-based names
                if re.match(r'^[a-z]+-\d{4}-\d{2}-\d{2}$', key.name):
                    continue
                if key.name == model_id:
                    return {
                        "valid": True,
                        "model_id": model_id,
                        "provider": key_type,
                        "custom": True,
                        "base_url": key.base_url,
                    }
    except Exception as e:
        logger.debug(f"Failed to check custom models: {e}")

    return {
        "valid": False,
        "error": f"Model '{model_id}' not found",
        "model_id": model_id,
    }


@router.get("/context/{model_id}")
async def get_model_context(model_id: str) -> Dict[str, Any]:
    """Get the context window size for a model.

    Args:
        model_id: The model ID (e.g., "qwen-35-27-fast", "zai-glm-4-7")

    Returns:
        Dict with max_tokens for the model
    """
    max_tokens = get_model_max_tokens(model_id, default=128000)
    return {
        "model_id": model_id,
        "max_tokens": max_tokens,
    }
