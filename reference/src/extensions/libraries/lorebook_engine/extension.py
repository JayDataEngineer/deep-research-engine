"""
Lorebook Engine Extension

Provides keyword and semantic activation for world information.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from src.extensions.core.base import BaseExtension, ExtensionStatus


class LorebookEngineExtension(BaseExtension):
    """Lorebook engine extension for managing world information."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router = None
        self._status = ExtensionStatus.LOADED

    @property
    def status(self) -> ExtensionStatus:
        return self._status

    def register_routes(self, app) -> None:
        """Register lorebook API routes."""
        from .routes import router as lorebooks_router
        app.include_router(lorebooks_router, prefix="/api/v1/lorebooks", tags=["lorebooks"])

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/api/v1/lorebooks",
            "name": "lorebooks",
            "tags": ["lorebooks", "world-info", "activation"],
            "description": "Lorebook management API with keyword and semantic activation",
        }

    async def initialize(self, context) -> None:
        """Initialize the lorebook engine."""
        self._status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Shutdown the lorebook engine."""
        self._status = ExtensionStatus.UNLOADED
