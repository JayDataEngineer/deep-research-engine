"""API Key Management Extension.

Provides secure storage and retrieval of encrypted API keys.
Keys are encrypted at rest using Fernet symmetric encryption.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Dict

from fastapi import APIRouter
from src.extensions.core.base import LogicExtension, ExtensionStatus

logger = logging.getLogger(__name__)


class APIKeysExtension(LogicExtension):
    """API key management extension."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router: Optional[APIRouter] = None

    async def initialize(self, context):
        """Initialize the API keys extension."""
        await super().initialize(context)

        # Initialize the API key service
        from src.services.api_keys import get_api_key_service
        service = get_api_key_service()
        await service.initialize()

        logger.info("API keys extension initialized")

    def register_routes(self, app) -> None:
        """Register API routes with the FastAPI application."""
        app.include_router(self.router)
        logger.info("API keys routes registered at /api/v1/keys")

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the router this extension provides."""
        return {
            "prefix": "/api/v1/keys",
            "name": "api_keys",
            "tags": ["api-keys"],
            "description": "API key management endpoints"
        }

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(self, input_data: dict, **kwargs):
        """Process API key request."""
        action = input_data.get('action', 'list')

        if action == 'create':
            return await self.create_key(input_data)
        elif action == 'list':
            return await self.list_keys(input_data)
        elif action == 'get':
            return await self.get_key(input_data)
        elif action == 'update':
            return await self.update_key(input_data)
        elif action == 'delete':
            return await self.delete_key(input_data)
        elif action == 'deactivate':
            return await self.deactivate_key(input_data)
        elif action == 'test':
            return await self.test_key(input_data)
        elif action == 'fetch_models':
            return await self.fetch_models(input_data)
        else:
            return {"error": f"Unknown action: {action}"}

    @property
    def router(self) -> APIRouter:
        """Get the API router for this extension."""
        if self._router is None:
            from .routes import router
            self._router = router
        return self._router

    async def create_key(self, input_data: dict) -> dict:
        """Create a new API key."""
        from src.services.api_keys import get_api_key_service, APIKeyCreate

        try:
            service = get_api_key_service()
            key_data = APIKeyCreate(**input_data)
            result = await service.create_key(key_data)
            return result
        except Exception as e:
            logger.error(f"Create key error: {e}")
            return {"error": str(e)}

    async def list_keys(self, input_data: dict) -> dict:
        """List all stored API keys."""
        from src.services.api_keys import get_api_key_service

        try:
            service = get_api_key_service()
            keys = await service.list_keys(key_type=input_data.get("key_type"))
            return {"keys": keys}
        except Exception as e:
            logger.error(f"List keys error: {e}")
            return {"error": str(e)}

    async def get_key(self, input_data: dict) -> dict:
        """Get a decrypted API key."""
        from src.services.api_keys import get_api_key_service

        try:
            service = get_api_key_service()
            decrypted_key = await service.get_key(
                input_data.get("key_type"),
                input_data.get("name")
            )

            if decrypted_key is None:
                return {"error": "Key not found"}

            return {
                "key_type": input_data.get("key_type"),
                "name": input_data.get("name"),
                "api_key": decrypted_key
            }
        except Exception as e:
            logger.error(f"Get key error: {e}")
            return {"error": str(e)}

    async def update_key(self, input_data: dict) -> dict:
        """Update an existing API key."""
        from src.services.api_keys import get_api_key_service, APIKeyUpdate

        try:
            service = get_api_key_service()
            key_data = APIKeyUpdate(**input_data)
            updated = await service.update_key(
                input_data.get("key_type"),
                input_data.get("name"),
                key_data
            )

            if not updated:
                return {"error": "Key not found"}

            return {"message": "API key updated successfully"}
        except Exception as e:
            logger.error(f"Update key error: {e}")
            return {"error": str(e)}

    async def delete_key(self, input_data: dict) -> dict:
        """Delete an API key."""
        from src.services.api_keys import get_api_key_service

        try:
            service = get_api_key_service()
            deleted = await service.delete_key(
                input_data.get("key_type"),
                input_data.get("name")
            )

            if not deleted:
                return {"error": "Key not found"}

            return {"message": "API key deleted successfully"}
        except Exception as e:
            logger.error(f"Delete key error: {e}")
            return {"error": str(e)}

    async def deactivate_key(self, input_data: dict) -> dict:
        """Deactivate an API key without deleting it."""
        from src.services.api_keys import get_api_key_service

        try:
            service = get_api_key_service()
            deactivated = await service.deactivate_key(
                input_data.get("key_type"),
                input_data.get("name")
            )

            if not deactivated:
                return {"error": "Key not found"}

            return {"message": "API key deactivated successfully"}
        except Exception as e:
            logger.error(f"Deactivate key error: {e}")
            return {"error": str(e)}

    async def test_key(self, input_data: dict) -> dict:
        """Test an API key without saving it."""
        # This is handled by the routes module
        pass

    async def fetch_models(self, input_data: dict) -> dict:
        """Fetch available models from provider."""
        # This is handled by the routes module
        pass

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(APIKeysExtension)
