"""
Extension Management API Extension

Provides API endpoints for the extension system itself.
This is a meta-extension that manages the extension system.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from src.extensions.core.base import BaseExtension, ExtensionStatus


class ExtensionManagementExtension(BaseExtension):
    """
    Extension management API extension.

    This is a special extension that provides the API for managing
    other extensions. It's loaded by the core to enable extension
    capabilities query and lifecycle management.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._status = ExtensionStatus.LOADED

    @property
    def status(self) -> ExtensionStatus:
        return self._status

    def register_routes(self, app) -> None:
        """Register extension management API routes."""
        from src.api.core.extensions import router, set_extension_manager
        # The extension_manager will be set by main.py during boot
        app.include_router(router, tags=["extensions"])

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/api/v1/extensions",
            "name": "extensions",
            "tags": ["extensions", "system", "management"],
            "description": "Extension system API for capabilities and lifecycle management",
        }

    async def initialize(self, context) -> None:
        """Initialize the extension management API."""
        # Set the extension manager reference if available
        if context and hasattr(context, 'manager'):
            from src.api.core.extensions import set_extension_manager
            set_extension_manager(context.manager)
        self._status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Shutdown the extension management API."""
        self._status = ExtensionStatus.UNLOADED
