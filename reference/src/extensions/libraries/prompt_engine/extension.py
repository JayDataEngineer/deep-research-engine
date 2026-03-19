"""
Prompt Engine Extension

Provides YAML-based prompt stack management with Jinja2 rendering.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from src.extensions.core.base import BaseExtension, ExtensionStatus


class PromptEngineExtension(BaseExtension):
    """Prompt engine extension for managing prompt stacks."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._status = ExtensionStatus.LOADED

    @property
    def status(self) -> ExtensionStatus:
        return self._status

    def register_routes(self, app) -> None:
        """Register prompt engine API routes."""
        from .prompt_stacks import router as prompt_stacks_router
        from .prompts import router as prompts_router
        app.include_router(prompt_stacks_router, prefix="/api/v1/prompts/stacks", tags=["prompts"])
        app.include_router(prompts_router, prefix="/api/v1/prompts", tags=["prompts"])

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/api/v1/prompts",
            "name": "prompts",
            "tags": ["prompts", "prompt-stacks", "templates"],
            "description": "Prompt stack management API with YAML and Jinja2 support",
        }

    async def initialize(self, context) -> None:
        """Initialize the prompt engine."""
        self._status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Shutdown the prompt engine."""
        # Shutdown stack manager
        try:
            from .manager import get_stack_manager
            manager = get_stack_manager()
            if manager:
                await manager.shutdown()
        except Exception:
            pass
        self._status = ExtensionStatus.UNLOADED
