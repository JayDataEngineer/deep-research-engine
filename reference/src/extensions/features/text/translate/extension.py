"""Translate Extension

Manages text translation and language detection.
"""

from src.extensions.core.base import LogicExtension, ExtensionStatus
from pathlib import Path
from typing import Dict, Any
from .manager import TranslateManager


class Extension(LogicExtension):
    """Translate extension."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.manager: TranslateManager = None

    async def initialize(self, context) -> None:
        """Initialize the translation manager."""
        self.context = context
        self.manager = TranslateManager()
        self.status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.manager = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process translation requests."""
        if not self.manager:
            return {"error": "Manager not initialized"}

        action = input_data.get("action")

        try:
            if action == "translate":
                result = self.manager.translate(
                    text=input_data.get("text", ""),
                    target_lang=input_data.get("target_lang", "en"),
                    source_lang=input_data.get("source_lang", "auto")
                )
            elif action == "detect":
                result = self.manager.detect_language(
                    text=input_data.get("text", "")
                )
            elif action == "batch_translate":
                result = self.manager.batch_translate(
                    texts=input_data.get("texts", []),
                    target_lang=input_data.get("target_lang", "en")
                )
            elif action == "list_languages":
                result = self.manager.list_supported_languages()
            else:
                return {"error": f"Unknown action: {action}"}

            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
