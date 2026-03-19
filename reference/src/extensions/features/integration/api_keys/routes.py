"""API Key Management Routes.

Provides REST endpoints for storing, retrieving, and managing encrypted API keys.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel, Field

from src.services.api_keys import (
    APIKeyService,
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyResponse,
    KeyType,
    get_api_key_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/keys", tags=["api-keys"])


@router.post("", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(key_data: APIKeyCreate) -> APIKeyResponse:
    """Store a new API key.

    The key is encrypted before storage.

    Args:
        key_data: The key data to store

    Returns:
        The created key metadata

    Raises:
        400: If key data is invalid
        409: If a key with the same type/name already exists
    """
    service = get_api_key_service()

    try:
        result = await service.create_key(key_data)
        return result
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"API key '{key_data.name}' of type '{key_data.key_type}' already exists"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create API key: {str(e)}"
        )


@router.get("", response_model=List[APIKeyResponse])
async def list_api_keys(key_type: Optional[str] = None) -> List[APIKeyResponse]:
    """List all stored API keys.

    Returns metadata only - actual keys are not included.

    Args:
        key_type: Optional filter by key type

    Returns:
        List of API key metadata
    """
    service = get_api_key_service()

    try:
        keys = await service.list_keys(key_type=key_type)
        return keys
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}"
        )


@router.get("/{key_type}/{name}", response_model=dict)
async def get_api_key(key_type: str, name: str) -> dict:
    """Retrieve and decrypt an API key.

    This endpoint returns the actual decrypted key value.

    Args:
        key_type: The type of key
        name: The name of the key

    Returns:
        Dictionary with the decrypted key

    Raises:
        404: If the key is not found
    """
    service = get_api_key_service()

    try:
        decrypted_key = await service.get_key(key_type, name)

        if decrypted_key is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key '{name}' of type '{key_type}' not found"
            )

        return {
            "key_type": key_type,
            "name": name,
            "api_key": decrypted_key
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve API key: {str(e)}"
        )


@router.put("/{key_type}/{name}", response_model=dict)
async def update_api_key(key_type: str, name: str, key_data: APIKeyUpdate) -> dict:
    """Update an existing API key.

    Args:
        key_type: The type of key
        name: The name of the key
        key_data: New key data

    Returns:
        Success message

    Raises:
        404: If the key is not found
    """
    service = get_api_key_service()

    try:
        updated = await service.update_key(key_type, name, key_data)

        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key '{name}' of type '{key_type}' not found"
            )

        return {
            "message": f"API key '{name}' of type '{key_type}' updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update API key: {str(e)}"
        )


@router.delete("/{key_type}/{name}", response_model=dict)
async def delete_api_key(key_type: str, name: str) -> dict:
    """Delete an API key.

    Args:
        key_type: The type of key
        name: The name of the key

    Returns:
        Success message

    Raises:
        404: If the key is not found
    """
    service = get_api_key_service()

    try:
        deleted = await service.delete_key(key_type, name)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key '{name}' of type '{key_type}' not found"
            )

        return {
            "message": f"API key '{name}' of type '{key_type}' deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete API key: {str(e)}"
        )


@router.post("/{key_type}/{name}/deactivate", response_model=dict)
async def deactivate_api_key(key_type: str, name: str) -> dict:
    """Deactivate an API key without deleting it.

    Args:
        key_type: The type of key
        name: The name of the key

    Returns:
        Success message

    Raises:
        404: If the key is not found
    """
    service = get_api_key_service()

    try:
        deactivated = await service.deactivate_key(key_type, name)

        if not deactivated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key '{name}' of type '{key_type}' not found"
            )

        return {
            "message": f"API key '{name}' of type '{key_type}' deactivated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deactivate API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate API key: {str(e)}"
        )


@router.get("/types", response_model=List[str])
async def get_supported_key_types() -> List[str]:
    """Get list of supported API key types.

    With LiteLLM proxy handling all provider routing, we only support:
    - litellm: Primary key for LiteLLM proxy
    - custom: Custom OpenAI-compatible endpoints

    Legacy provider-specific keys are deprecated but still supported
    for backward compatibility during migration.

    Returns:
        List of supported key types
    """
    return [
        KeyType.LITELLM,    # Primary: LiteLLM proxy
        KeyType.CUSTOM,     # Custom OpenAI-compatible endpoint
        # Deprecated but still supported:
        KeyType.OPENAI,
        KeyType.ANTHROPIC,
        KeyType.OPENROUTER,
        KeyType.ZAI,
        KeyType.GLM,
        KeyType.GOOGLE,
        KeyType.COHERE,
    ]


class TestApiKeyRequest(BaseModel):
    """Request model for testing an API key."""
    key_type: str = Field(..., description="Type of API key")
    api_key: str = Field(..., description="The API key to test")
    base_url: str = Field(default="", description="Base URL for the API")
    model_id: str = Field(default="", description="Model ID to test with")


class TestApiKeyResponse(BaseModel):
    """Response model for API key test results."""
    valid: bool = Field(..., description="Whether the key is valid")
    message: str = Field(..., description="Result message")


class FetchModelsRequest(BaseModel):
    """Request model for fetching available models.

    If api_key is not provided, will use the first stored active key for the key_type.
    """
    key_type: str = Field(..., description="Type of API key")
    api_key: str = Field(default="", description="The API key to use (if empty, uses stored key)")
    base_url: str = Field(default="", description="Base URL for the API")


class FetchModelsResponse(BaseModel):
    """Response model for fetched models."""
    models: list = Field(default_factory=list, description="List of available models")
    error: str | None = None


@router.post("/test-api-key", response_model=TestApiKeyResponse)
async def test_api_key(request: TestApiKeyRequest) -> TestApiKeyResponse:
    """Test an API key without saving it.

    With LiteLLM proxy handling all provider routing, we test connectivity
    to the LiteLLM proxy or custom OpenAI-compatible endpoint.

    Args:
        request: The test request with key type, API key, and optional base URL

    Returns:
        Validation result
    """
    import httpx

    logger.info(f"Testing API key for type: {request.key_type}")

    # For LiteLLM proxy, test the proxy endpoint
    if request.key_type == KeyType.LITELLM:
        base_url = request.base_url or "http://litellm.local/v1"
        api_key = request.api_key or "sk-litellm-master"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test with models endpoint
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if response.status_code == 200:
                    return TestApiKeyResponse(
                        valid=True,
                        message="LiteLLM proxy is accessible and key is valid"
                    )
                elif response.status_code == 401:
                    return TestApiKeyResponse(
                        valid=False,
                        message="Invalid API key for LiteLLM proxy"
                    )
                else:
                    return TestApiKeyResponse(
                        valid=False,
                        message=f"LiteLLM returned status {response.status_code}"
                    )
        except httpx.TimeoutException:
            return TestApiKeyResponse(
                valid=False,
                message="Request timed out - check LiteLLM proxy is running"
            )
        except httpx.ConnectError:
            return TestApiKeyResponse(
                valid=False,
                message="Could not connect to LiteLLM proxy"
            )
        except Exception as e:
            logger.error(f"Error testing LiteLLM key: {e}")
            return TestApiKeyResponse(
                valid=False,
                message=f"Error testing LiteLLM key: {str(e)}"
            )

    # For custom providers, test with generic OpenAI-compatible endpoint
    elif request.key_type == KeyType.CUSTOM:
        if not request.base_url:
            return TestApiKeyResponse(
                valid=False,
                message="Base URL is required for custom provider"
            )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{request.base_url}/models",
                    headers={"Authorization": f"Bearer {request.api_key}"}
                )
                if response.status_code == 200:
                    return TestApiKeyResponse(
                        valid=True,
                        message="Custom endpoint is accessible and key is valid"
                    )
                elif response.status_code == 401:
                    return TestApiKeyResponse(
                        valid=False,
                        message="Invalid API key for custom endpoint"
                    )
                else:
                    return TestApiKeyResponse(
                        valid=True,
                        message="Custom endpoint accepted request (limited validation)"
                    )
        except httpx.TimeoutException:
            return TestApiKeyResponse(
                valid=False,
                message="Request timed out - check your network connection"
            )
        except httpx.ConnectError:
            return TestApiKeyResponse(
                valid=False,
                message="Could not connect to custom endpoint"
            )
        except Exception as e:
            logger.error(f"Error testing custom key: {e}")
            return TestApiKeyResponse(
                valid=False,
                message=f"Error testing custom key: {str(e)}"
            )

    # Legacy provider-specific keys - redirect to use LiteLLM
    else:
        return TestApiKeyResponse(
            valid=True,
            message=f"Provider-specific keys ({request.key_type}) are deprecated. Use LiteLLM proxy instead."
        )


@router.post("/models", response_model=FetchModelsResponse)
async def fetch_available_models(request: FetchModelsRequest) -> FetchModelsResponse:
    """Fetch available models from the provider's API.

    Queries the provider's /models endpoint (or equivalent) to get a list of available models.

    If api_key is not provided in the request, will use the first stored active key for the key_type.

    Args:
        request: The fetch request with key type, optional API key, and optional base URL

    Returns:
        List of available models with their IDs and names
    """
    import httpx

    logger.info(f"Fetching models for provider: {request.key_type}")

    # Default base URLs for known providers
    default_base_urls = {
        KeyType.OPENAI: "https://api.openai.com/v1",
        KeyType.ANTHROPIC: "https://api.anthropic.com",
        KeyType.ZAI: "https://api.z.ai/api/coding/paas/v4",
        KeyType.GLM: "https://open.bigmodel.cn/api/paas/v4",
        KeyType.GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
        KeyType.COHERE: "https://api.cohere.ai/v1",
    }

    # If API key not provided, retrieve from storage
    api_key = request.api_key
    stored_base_url = None
    if not api_key:
        service = get_api_key_service()
        keys = await service.list_keys(key_type=request.key_type)

        if not keys:
            return FetchModelsResponse(
                models=[],
                error=f"No API key found for {request.key_type}. Please add an API key first."
            )

        # Use the first active key
        key_record = keys[0]
        api_key = await service.get_key(request.key_type, key_record.name)
        stored_base_url = key_record.base_url  # Get custom base_url from stored key

        if not api_key:
            return FetchModelsResponse(
                models=[],
                error=f"Failed to retrieve stored API key for {request.key_type}"
            )

        logger.info(f"Using stored key: {request.key_type}/{key_record.name}")

    # Determine base_url: request > stored key > default
    base_url = request.base_url or stored_base_url or default_base_urls.get(request.key_type, "")

    if not base_url:
        return FetchModelsResponse(
            models=[],
            error=f"Base URL is required for key type: {request.key_type}"
        )

    try:
        # Fetch models based on provider type
        if request.key_type == KeyType.OPENAI:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [
                        {
                            "id": m.get("id", ""),
                            "name": m.get("id", ""),
                            "displayName": _format_openai_model_name(m.get("id", ""))
                        }
                        for m in data.get("data", [])
                    ]
                    return FetchModelsResponse(models=models)
                elif response.status_code == 401:
                    return FetchModelsResponse(models=[], error="Invalid API key")
                else:
                    return FetchModelsResponse(models=[], error=f"API returned status {response.status_code}")

        elif request.key_type == KeyType.ANTHROPIC:
            # Anthropic doesn't have a /models endpoint, return known models
            models = [
                {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "displayName": "Claude 3.5 Sonnet"},
                {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "displayName": "Claude 3.5 Haiku"},
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "displayName": "Claude 3 Opus"},
                {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "displayName": "Claude 3 Sonnet"},
                {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "displayName": "Claude 3 Haiku"},
            ]
            # Verify API key by testing access
            async with httpx.AsyncClient(timeout=10.0) as client:
                test_response = await client.post(
                    f"{base_url}/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": []}
                )
                if test_response.status_code == 401:
                    return FetchModelsResponse(models=[], error="Invalid API key")
            return FetchModelsResponse(models=models)

        else:
            # For custom providers, try /models endpoint
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [
                        {
                            "id": m.get("id", ""),
                            "name": m.get("id", ""),
                            "displayName": m.get("id", "")
                        }
                        for m in data.get("data", []) if isinstance(m, dict)
                    ]
                    return FetchModelsResponse(models=models)
                else:
                    return FetchModelsResponse(models=[], error="Could not fetch models from custom provider")

    except httpx.TimeoutException:
        return FetchModelsResponse(models=[], error="Request timed out")
    except httpx.ConnectError:
        return FetchModelsResponse(models=[], error="Could not connect to API")
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return FetchModelsResponse(models=[], error=f"Error fetching models: {str(e)}")


def _format_openai_model_name(model_id: str) -> str:
    """Format OpenAI model ID for display."""
    # Known model name mappings
    display_names = {
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4-turbo": "GPT-4 Turbo",
        "gpt-4": "GPT-4",
        "gpt-3.5-turbo": "GPT-3.5 Turbo",
        "o1-preview": "o1 Preview",
        "o1-mini": "o1 Mini",
    }
    return display_names.get(model_id, model_id)
