"""Attachments Manager Extension

Manages file attachments and web scraping history.
Tracks URLs that have been scraped, cached content, and provides CLI management.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class AttachmentsExtension(LogicExtension):
    """Manages attachments and web scraping history."""

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self.storage_path = Path(self.config.get("storage_path", "inputs/attachments"))
        self.max_history = self.config.get("max_history", 1000)
        self.history_file = self.storage_path / "history.json"
        self._history: List[Dict[str, Any]] = []

    async def initialize(self, context) -> None:
        """Initialize the attachments manager."""
        await super().initialize(context)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        await self._load_history()
        logger.info(f"Attachments Manager initialized with {len(self._history)} history entries")

    async def shutdown(self) -> None:
        """Shutdown and save history."""
        await self._save_history()

    async def _load_history(self) -> None:
        """Load attachment history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self._history = json.load(f)
                logger.debug(f"Loaded {len(self._history)} attachment history entries")
            except Exception as e:
                logger.error(f"Failed to load attachment history: {e}")
                self._history = []
        else:
            self._history = []

    async def _save_history(self) -> None:
        """Save attachment history to disk."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self._history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save attachment history: {e}")

    async def add_attachment(
        self,
        url: str,
        content: str,
        title: str = "",
        source_type: str = "web_scrape"
    ) -> Dict[str, Any]:
        """Add an attachment to history.

        Args:
            url: The URL or file path
            content: The content (scraped or file content)
            title: Optional title
            source_type: Type of attachment (web_scrape, file_upload, etc.)

        Returns:
            The attachment record
        """
        attachment = {
            "id": f"att_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "url": url,
            "title": title or url,
            "content": content[:5000],  # Store first 5000 chars as preview
            "content_length": len(content),
            "source_type": source_type,
            "created_at": datetime.now().isoformat(),
        }

        self._history.insert(0, attachment)  # Add to beginning

        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[:self.max_history]

        await self._save_history()
        logger.info(f"Added attachment: {url}")
        return attachment

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a URL and add to history.

        Args:
            url: The URL to scrape

        Returns:
            The attachment record with scraped content
        """
        # Import web scraping tool
        from src.shared.tools.registry import get_tool

        web_get_tool = get_tool("web_get")
        if not web_get_tool:
            raise RuntimeError("web_get tool not available")

        # Scrape the URL
        if hasattr(web_get_tool, 'ainvoke'):
            result = await web_get_tool.ainvoke({"url": url})
        elif asyncio.iscoroutinefunction(web_get_tool):
            result = await web_get_tool(url=url)
        else:
            result = web_get_tool(url=url)

        # Extract title and content
        if isinstance(result, dict):
            title = result.get("title", "")
            content = result.get("content", str(result))
        else:
            title = ""
            content = str(result)

        return await self.add_attachment(url, content, title, "web_scrape")

    def get_history(self, limit: int = 50, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get attachment history.

        Args:
            limit: Maximum number of entries to return
            source_type: Filter by source type (optional)

        Returns:
            List of attachment records
        """
        history = self._history

        if source_type:
            history = [a for a in history if a.get("source_type") == source_type]

        return history[:limit]

    def get_attachment(self, attachment_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific attachment by ID.

        Args:
            attachment_id: The attachment ID

        Returns:
            The attachment record or None
        """
        for attachment in self._history:
            if attachment.get("id") == attachment_id:
                return attachment
        return None

    async def clear_history(self, source_type: Optional[str] = None) -> int:
        """Clear attachment history.

        Args:
            source_type: Only clear this type (optional)

        Returns:
            Number of entries cleared
        """
        if source_type:
            original_count = len(self._history)
            self._history = [a for a in self._history if a.get("source_type") != source_type]
            cleared = original_count - len(self._history)
        else:
            cleared = len(self._history)
            self._history = []

        await self._save_history()
        return cleared

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process attachments in chat context.

        This is called during chat processing to handle any attachments
        in the user message.
        """
        # Check if user message contains URLs to scrape
        user_message = input_data.get("user_message", "")
        # This would be expanded to detect URLs and auto-scrape them
        return {}
