"""LLM Provider Discovery Extension.

Discovers available models from:
1. LiteLLM proxy (primary - all configured models)
2. Local servers (llama.cpp via llama.local, Ollama)

All model routing is handled by LiteLLM configuration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Dict

from fastapi import APIRouter
from src.extensions.core.base import LogicExtension, ExtensionStatus

logger = logging.getLogger(__name__)


class ProvidersExtension(LogicExtension):
    """LLM provider discovery extension."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router: APIRouter = None

    async def initialize(self, context):
        """Initialize the providers extension."""
        await super().initialize(context)
        logger.info("Providers extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(self, input_data: dict, **kwargs):
        """Process provider request."""
        action = input_data.get('action', 'list')

        if action == 'list':
            return await self.list_providers()
        elif action == 'get_models':
            return await self.get_provider_models(input_data.get("provider_id", ""))
        else:
            return {"error": f"Unknown action: {action}"}

    def register_routes(self, app) -> None:
        """Register API routes with the FastAPI application."""
        app.include_router(self.router)
        logger.info("Providers routes registered at /api/v1/providers")

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the router this extension provides."""
        return {
            "prefix": "/api/v1/providers",
            "name": "providers",
            "tags": ["providers"],
            "description": "LLM provider discovery endpoints"
        }

    @property
    def router(self) -> APIRouter:
        """Get the API router for this extension."""
        if self._router is None:
            from .routes import router
            self._router = router
        return self._router

    async def list_providers(self) -> dict:
        """List all available LLM providers."""
        # This is handled by the routes module
        pass

    async def get_provider_models(self, provider_id: str) -> dict:
        """Get available models for a specific provider."""
        # This is handled by the routes module
        pass


from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(ProvidersExtension)
