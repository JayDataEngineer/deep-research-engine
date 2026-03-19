"""Shortcuts Extension

Manages keyboard shortcuts and quick actions.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import ShortcutsManager


class Extension(LogicExtension):
    """Shortcuts extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: ShortcutsManager = None

    async def initialize(self, context) -> None:
        """Initialize the shortcuts manager."""
        self.context = context
        self.manager = ShortcutsManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process shortcut-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "create":
                result = self.manager.create_shortcut(
                    name=input_data.get("name", ""),
                    keys=input_data.get("keys", []),
                    action=input_data.get("action", "")
                )
            elif action == "list":
                result = self.manager.list_shortcuts()
            elif action == "execute":
                result = self.manager.execute_shortcut(
                    name=input_data.get("name", "")
                )
            elif action == "delete":
                result = self.manager.delete_shortcut(
                    name=input_data.get("name", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
