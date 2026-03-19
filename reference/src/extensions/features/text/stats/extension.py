"""Stats Extension

Manages chat statistics and analytics.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import StatsManager


class Extension(LogicExtension):
    """Stats extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: StatsManager = None

    async def initialize(self, context) -> None:
        """Initialize the stats manager."""
        self.context = context
        self.manager = StatsManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.manager:
            self.manager.save_stats()
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process stats-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "get_chat_stats":
                result = self.manager.get_chat_stats(
                    chat_name=input_data.get("chat_name", "")
                )
            elif action == "get_character_stats":
                result = self.manager.get_character_stats(
                    character_name=input_data.get("character_name", "")
                )
            elif action == "get_summary":
                result = self.manager.get_summary_stats()
            elif action == "reset":
                result = self.manager.reset_stats(
                    chat_name=input_data.get("chat_name", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
