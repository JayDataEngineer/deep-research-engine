"""Connection Manager Extension

Manages external service connections and API keys.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import ConnectionManager


class Extension(LogicExtension):
    """Connection Manager extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: ConnectionManager = None

    async def initialize(self, context) -> None:
        """Initialize the connection manager."""
        self.context = context
        self.manager = ConnectionManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process connection-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "add_connection":
                result = self.manager.add_connection(
                    name=input_data.get("name", ""),
                    service_type=input_data.get("service_type", ""),
                    config=input_data.get("config", {})
                )
            elif action == "test_connection":
                result = self.manager.test_connection(
                    connection_id=input_data.get("connection_id", "")
                )
            elif action == "list":
                result = self.manager.list_connections()
            elif action == "delete":
                result = self.manager.delete_connection(
                    connection_id=input_data.get("connection_id", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
