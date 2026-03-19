"""Gallery Extension

Manages image gallery and generated content.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import GalleryManager


class Extension(LogicExtension):
    """Gallery extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: GalleryManager = None

    async def initialize(self, context) -> None:
        """Initialize the gallery manager."""
        self.context = context
        self.manager = GalleryManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process gallery-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "add_image":
                result = self.manager.add_image(
                    path=input_data.get("path", ""),
                    metadata=input_data.get("metadata", {})
                )
            elif action == "list":
                result = self.manager.list_images(
                    category=input_data.get("category", "all")
                )
            elif action == "get":
                result = self.manager.get_image(
                    image_id=input_data.get("image_id", "")
                )
            elif action == "delete":
                result = self.manager.delete_image(
                    image_id=input_data.get("image_id", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
