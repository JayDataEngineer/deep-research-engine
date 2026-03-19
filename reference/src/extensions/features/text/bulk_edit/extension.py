"""Bulk Edit Extension

Provides bulk operations for editing messages and chat content.
This is a Job extension (CLI tool) not a Mode extension.
"""

from src.extensions.core.base import JobExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import BulkEditManager


class Extension(JobExtension):
    """Bulk Edit job extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: BulkEditManager = None

    async def initialize(self, context) -> None:
        """Initialize the bulk edit manager."""
        self.context = context
        self.manager = BulkEditManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def run_job(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a bulk edit job.

        Args:
            input_data: Job parameters including operation, items, etc.

        Returns:
            Job execution results
        """
        if not self.manager:
            return {"error": "Manager not initialized"}

        operation = input_data.get("operation")
        items = input_data.get("items", [])

        try:
            if operation == "replace":
                result = self.manager.bulk_replace(
                    items,
                    input_data.get("target", ""),
                    input_data.get("replacement", "")
                )
            elif operation == "regex":
                result = self.manager.bulk_regex(
                    items,
                    input_data.get("pattern", ""),
                    input_data.get("replacement", "")
                )
            else:
                return {"error": f"Unknown operation: {operation}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
