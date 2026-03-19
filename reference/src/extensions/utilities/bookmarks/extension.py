"""Bookmarks Extension

Provides chat state bookmarks and snapshots for saving/restoring chat positions.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import BookmarkManager


class Extension(LogicExtension):
    """Bookmarks extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: BookmarkManager = None

    async def initialize(self, context) -> None:
        """Initialize the bookmarks manager."""
        self.context = context
        self.manager = BookmarkManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.manager:
            self.manager.save_all()
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process bookmark-related requests.

        Args:
            input_data: Request data with action and parameters
            **kwargs: Additional parameters

        Returns:
            Result of the bookmark operation
        """
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "create":
                result = self.manager.create_bookmark(
                    chat_name=input_data.get("chat_name", ""),
                    message_index=input_data.get("message_index", 0),
                    name=input_data.get("name", ""),
                    notes=input_data.get("notes", "")
                )
            elif action == "list":
                result = self.manager.list_bookmarks(
                    chat_name=input_data.get("chat_name", "")
                )
            elif action == "restore":
                result = self.manager.restore_bookmark(
                    bookmark_id=input_data.get("bookmark_id", "")
                )
            elif action == "delete":
                result = self.manager.delete_bookmark(
                    bookmark_id=input_data.get("bookmark_id", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
