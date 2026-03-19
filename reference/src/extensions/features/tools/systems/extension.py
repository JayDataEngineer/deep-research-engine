"""System Health & Config Extension.

Provides health checks, configuration, and system-level endpoints.
"""

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from src.extensions.core.base import LogicExtension, ExtensionStatus

logger = logging.getLogger(__name__)


class SystemsExtension(LogicExtension):
    """System health and configuration extension."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router: APIRouter = None

    async def initialize(self, context):
        """Initialize the systems extension."""
        await super().initialize(context)
        logger.info("Systems extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(self, input_data: dict, **kwargs):
        """Process system request."""
        action = input_data.get('action', 'health')

        if action == 'health':
            return await self.get_health()
        elif action == 'config':
            return await self.get_config()
        elif action == 'extensions':
            return await self.list_extensions()
        else:
            return {"error": f"Unknown action: {action}"}

    @property
    def router(self) -> APIRouter:
        """Get the API router for this extension."""
        if self._router is None:
            from .routes import router
            self._router = router
        return self._router

    def register_routes(self, app) -> None:
        """Register the system health and config routes."""
        # Import and include the router
        from . import routes
        app.include_router(routes.router)

    def get_router_info(self) -> dict:
        """Get router information for frontend discovery."""
        return {
            "prefix": "/api/v1/system",
            "name": "system",
            "tags": ["system"],
            "description": "System health checks, configuration, and capabilities",
        }

    async def get_health(self) -> dict:
        """Get system health status."""
        from src.constants import ROOT_EXTENSIONS_DIR

        # Check if extensions directory exists
        extensions_dir = None
        extensions_count = 0

        if ROOT_EXTENSIONS_DIR.exists():
            extensions_dir = str(ROOT_EXTENSIONS_DIR)
            # Count extension folders
            if ROOT_EXTENSIONS_DIR.is_dir():
                extensions_count = len([
                    d for d in ROOT_EXTENSIONS_DIR.iterdir()
                    if d.is_dir() and not d.name.startswith("_")
                ])

        return {
            "status": "healthy",
            "version": "0.1.0",
            "extensions_dir": extensions_dir,
            "extensions_count": extensions_count,
        }

    async def get_config(self) -> dict:
        """Get public configuration."""
        from src.config import settings
        from src.constants import ROOT_EXTENSIONS_DIR
        import os

        # Detect available features based on extensions
        features = ["chat"]  # Always available

        # Check for TTS
        if os.environ.get("TTS_AVAILABLE") == "true":
            features.append("tts")

        # Check for deep research
        deep_research_path = ROOT_EXTENSIONS_DIR / "deep_research"
        if deep_research_path.exists():
            features.append("deep_research")

        # Check for vision
        features.append("vision")  # Assuming vision is always available

        return {
            "app_name": "Unified LLM Interface",
            "version": "0.1.0",
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url if settings.show_llm_config else None,
            "features": features,
        }

    async def list_extensions(self) -> dict:
        """List all loaded extensions."""
        # This requires access to the extension manager from context
        if not self.context or not self.context.manager:
            return {"extensions": [], "message": "Extension manager not available"}

        extensions = []
        for ext_id, ext in self.context.manager.extensions.items():
            extensions.append({
                "id": ext_id,
                "name": getattr(ext, "name", ext_id),
                "type": type(ext).__name__,
                "active": getattr(ext, "active", True),
            })

        return {"extensions": extensions}

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(SystemsExtension)
