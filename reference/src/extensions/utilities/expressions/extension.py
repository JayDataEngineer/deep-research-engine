"""Expressions & Sprites Extension

Manages character expression images and sprites for visual chat.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import ExpressionsManager


class Extension(LogicExtension):
    """Expressions & Sprites extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: ExpressionsManager = None

    async def initialize(self, context) -> None:
        """Initialize the expressions manager."""
        self.context = context
        self.manager = ExpressionsManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process expression-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "add_sprite":
                result = self.manager.add_sprite(
                    character_name=input_data.get("character_name", ""),
                    emotion=input_data.get("emotion", ""),
                    image_path=input_data.get("image_path", ""),
                    name=input_data.get("name", "")
                )
            elif action == "get_sprites":
                result = self.manager.get_character_sprites(
                    character_name=input_data.get("character_name", "")
                )
            elif action == "get_by_emotion":
                result = self.manager.get_sprite_by_emotion(
                    character_name=input_data.get("character_name", ""),
                    emotion=input_data.get("emotion", "")
                )
            elif action == "delete":
                result = self.manager.delete_sprite(
                    sprite_id=input_data.get("sprite_id", "")
                )
            elif action == "list_emotions":
                result = self.manager.list_available_emotions()
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
