"""Reasoning Extension

Manages chain-of-thought reasoning and structured thinking.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import ReasoningManager


class Extension(LogicExtension):
    """Reasoning extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: ReasoningManager = None

    async def initialize(self, context) -> None:
        """Initialize the reasoning manager."""
        self.context = context
        self.manager = ReasoningManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process reasoning-related requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "create_chain":
                result = self.manager.create_reasoning_chain(
                    query=input_data.get("query", ""),
                    steps=input_data.get("steps", [])
                )
            elif action == "analyze":
                result = self.manager.analyze_problem(
                    problem=input_data.get("problem", "")
                )
            elif action == "validate":
                result = self.manager.validate_reasoning(
                    reasoning=input_data.get("reasoning", "")
                )
            elif action == "format_output":
                result = self.manager.format_reasoning_output(
                    chain=input_data.get("chain", []),
                    format_type=input_data.get("format_type", "markdown")
                )
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
