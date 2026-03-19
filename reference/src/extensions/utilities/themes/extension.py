"""Themes Extension

Manages UI themes and appearance settings.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import ThemesManager


class Extension(LogicExtension):
    """Themes extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: ThemesManager = None

    async def initialize(self, context) -> None:
        """Initialize the themes manager."""
        self.context = context
        self.manager = ThemesManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process theme-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "create":
                result = self.manager.create_theme(
                    name=input_data.get("name", ""),
                    colors=input_data.get("colors", {}),
                    fonts=input_data.get("fonts", {})
                )
            elif action == "list":
                result = self.manager.list_themes()
            elif action == "apply":
                result = self.manager.apply_theme(
                    theme_id=input_data.get("theme_id", "")
                )
            elif action == "get_current":
                result = self.manager.get_current_theme()
            elif action == "delete":
                result = self.manager.delete_theme(
                    theme_id=input_data.get("theme_id", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
