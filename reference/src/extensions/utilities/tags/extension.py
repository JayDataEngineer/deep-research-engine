"""Tags Extension

Manages tags for characters, messages, and content.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import TagsManager


class Extension(LogicExtension):
    """Tags extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: TagsManager = None

    async def initialize(self, context) -> None:
        """Initialize the tags manager."""
        self.context = context
        self.manager = TagsManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process tag-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "create":
                result = self.manager.create_tag(
                    name=input_data.get("name", ""),
                    color=input_data.get("color", "#3b82f6"),
                    description=input_data.get("description", "")
                )
            elif action == "list":
                result = self.manager.list_tags()
            elif action == "add_to_item":
                result = self.manager.add_tag_to_item(
                    tag_id=input_data.get("tag_id", ""),
                    item_type=input_data.get("item_type", ""),
                    item_id=input_data.get("item_id", "")
                )
            elif action == "remove_from_item":
                result = self.manager.remove_tag_from_item(
                    tag_id=input_data.get("tag_id", ""),
                    item_type=input_data.get("item_type", ""),
                    item_id=input_data.get("item_id", "")
                )
            elif action == "delete":
                result = self.manager.delete_tag(
                    tag_id=input_data.get("tag_id", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
