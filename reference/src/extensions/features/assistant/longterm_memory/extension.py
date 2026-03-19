"""Longterm Memory Extension

Manages persistent long-term memory for characters and conversations.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import LongTermMemoryManager


class Extension(LogicExtension):
    """Longterm Memory extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: LongTermMemoryManager = None

    async def initialize(self, context) -> None:
        """Initialize the longterm memory manager."""
        self.context = context
        self.manager = LongTermMemoryManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.manager:
            # LongTermMemoryManager has _save_memories method, not save_all
            self.manager._save_memories()
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process memory-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "add_memory":
                result = self.manager.add_memory(
                    character_name=input_data.get("character_name", ""),
                    content=input_data.get("content", ""),
                    importance=input_data.get("importance", 0.5)
                )
            elif action == "recall":
                result = self.manager.recall_memories(
                    character_name=input_data.get("character_name", ""),
                    query=input_data.get("query", ""),
                    limit=input_data.get("limit", 10)
                )
            elif action == "list":
                result = self.manager.list_memories(
                    character_name=input_data.get("character_name", "")
                )
            elif action == "delete":
                result = self.manager.delete_memory(
                    memory_id=input_data.get("memory_id", "")
                )
            elif action == "update_importance":
                result = self.manager.update_importance(
                    memory_id=input_data.get("memory_id", ""),
                    importance=input_data.get("importance", 0.5)
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
