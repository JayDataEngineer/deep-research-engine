"""Content Formatter Extension.

Transforms content into various formats using deep agents with validation.

Formats are auto-discovered from formats/* directories, each containing:
- config.yaml: Prompts and overrides (Jinja2)
- schema.py: Pydantic model for validation

Usage:
    formatter = ContentFormatterExtension()
    result = await formatter.format(
        format="tweets",
        content="...",
        session_id="..."  # Optional Context Engine access
    )
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

# Add backend to path for absolute imports
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(backend_dir) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(backend_dir))

from src.extensions.core.base import LogicExtension
from src.extensions.core.context import ExtensionContext
from .utility import get_format_directories
from .base import format_content


class ContentFormatterExtension(LogicExtension):
    """Content Formatter Extension.

    Auto-discovers formats from formats/* directories.
    Each format has config.yaml (prompts) and schema.py (Pydantic model).
    """

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.formats_dir = root_path / "formats"
        self.available_formats = self._discover_formats()

    def _discover_formats(self) -> List[str]:
        """Auto-discover available formats.

        Returns:
            List of format names
        """
        format_dirs = get_format_directories(self.formats_dir)
        return [d.name for d in format_dirs]

    async def format(
        self,
        format_name: str,
        content: str,
        session_id: Optional[str] = None,
        character: Optional[str] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any
    ) -> BaseModel:
        """Format content using deep_agent with validation.

        Args:
            format_name: Format to use (tweets, podcast, substack, etc.)
            content: Input content to format
            session_id: Optional session ID for Context Engine access
            character: Optional character name for character-based formats
            max_retries: Optional retry count override
            **kwargs: Additional format-specific parameters

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If format not found
            RuntimeError: If formatting fails
        """
        if format_name not in self.available_formats:
            raise ValueError(
                f"Unknown format: {format_name}. "
                f"Available: {self.available_formats}"
            )

        return await format_content(
            format_name=format_name,
            content=content,
            base_config=self.config,
            formats_dir=self.formats_dir,
            session_id=session_id,
            character=character,
            **kwargs
        )

    def list_formats(self) -> List[str]:
        """List available formats.

        Returns:
            List of format names
        """
        return self.available_formats.copy()

    def get_emotion_detector(self) -> "EmotionDetector":
        """Get an EmotionDetector instance configured from this extension's config.

        Returns:
            EmotionDetector instance with configuration from config.yaml

        Example:
            detector = formatter.get_emotion_detector()
            lines_with_emotions = await detector.detect(dialogue_lines)
        """
        from .emotion import EmotionDetector
        emotion_config = self.config.get("emotion", {})
        return EmotionDetector(emotion_config)

    # Extension lifecycle methods
    async def on_start(self, context: ExtensionContext) -> None:
        """Initialize extension."""
        pass

    async def on_stop(self, context: ExtensionContext) -> None:
        """Cleanup extension."""
        pass


# CLI helper function
async def deep_formatter(
    config_selection: str,
    content: str,
    session_id: Optional[str] = None,
    retries: int = 10,
    **kwargs: Any
) -> BaseModel:
    """Simple function-based API for formatting.

    Args:
        config_selection: Format name to use
        content: Content to format
        session_id: Optional session ID for Context Engine
        retries: Number of retries (default 10)
        **kwargs: Additional parameters

    Returns:
        Formatted Pydantic model

    Raises:
        RuntimeError: If formatting fails or format not found
    """
    # Get extension
    try:
        from src.extensions.core.manager.loader import get_extension_manager
        manager = get_extension_manager()
        if not manager:
            raise RuntimeError("Extension manager not available")
    except ImportError:
        raise RuntimeError("Cannot get extension manager")

    # Get content_formatter extension
    formatter_ext = manager.get_extension("content_formatter")
    if not formatter_ext or not formatter_ext.enabled:
        raise RuntimeError("Content formatter extension not available")

    formatter = formatter_ext.instance

    # Format content
    return await formatter.format(
        format_name=config_selection,
        content=content,
        session_id=session_id,
        max_retries=retries,
        **kwargs
    )
