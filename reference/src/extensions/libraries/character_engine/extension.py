"""
Character Engine Extension

Provides character card and book management with parsing and CRUD operations.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from src.extensions.core.base import BaseExtension, ExtensionStatus


class CharacterEngineExtension(BaseExtension):
    """Character engine extension for managing character cards."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, config.get("root_path", root_path))
        self._router = None
        self._status = ExtensionStatus.LOADED

    @property
    def status(self) -> ExtensionStatus:
        return self._status

    def register_routes(self, app) -> None:
        """Register character engine API routes."""
        from .routes import router as character_router
        app.include_router(character_router, prefix="/api/v1/characters", tags=["characters"])

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/api/v1/characters",
            "name": "characters",
            "tags": ["characters", "character-cards", "roleplay"],
            "description": "Character card management API with JSON, YAML, and PNG support",
        }

    async def initialize(self, context) -> None:
        """Initialize the character engine."""
        self._status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Shutdown the character engine."""
        self._status = ExtensionStatus.UNLOADED
