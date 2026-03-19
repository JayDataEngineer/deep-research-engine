"""Chat Templates Extension

Manages message templates and preset responses.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import ChatTemplatesManager


class Extension(LogicExtension):
    """Chat Templates extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: ChatTemplatesManager = None

    async def initialize(self, context) -> None:
        """Initialize the chat templates manager."""
        self.context = context
        self.manager = ChatTemplatesManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process template-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "create":
                result = self.manager.create_template(
                    name=input_data.get("name", ""),
                    content=input_data.get("content", ""),
                    category=input_data.get("category", "user")
                )
            elif action == "list":
                result = self.manager.list_templates(
                    category=input_data.get("category", "all")
                )
            elif action == "apply":
                result = self.manager.apply_template(
                    template_id=input_data.get("template_id", ""),
                    variables=input_data.get("variables", {})
                )
            elif action == "delete":
                result = self.manager.delete_template(
                    template_id=input_data.get("template_id", "")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
