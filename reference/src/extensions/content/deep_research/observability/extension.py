"""
Observability API Extension

Provides API endpoints for run history and metrics tracking.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from src.extensions.core.base import BaseExtension, ExtensionStatus


class ObservabilityExtension(BaseExtension):
    """
    Observability API extension.

    Provides endpoints for accessing run history, metrics, and checkpoints
    from the LangGraph checkpointer database.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)

    def register_routes(self, app) -> None:
        """Register observability API routes."""
        from .routes import router as observability_router
        app.include_router(observability_router, tags=["observability"])

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/observability/api",
            "name": "observability",
            "tags": ["observability", "metrics", "runs", "checkpoints"],
            "description": "Observability API for run history and metrics tracking",
        }

    async def initialize(self, context) -> None:
        """Initialize the observability API."""
        await super().initialize(context)

    async def shutdown(self) -> None:
        """Shutdown the observability API."""
        await super().shutdown()


# Alias for extension loader compatibility
Extension = ObservabilityExtension
