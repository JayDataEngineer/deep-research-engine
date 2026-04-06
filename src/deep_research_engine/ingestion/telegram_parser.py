"""Telegram export parser - Kreuzberg-powered + custom JSON handling.

Telegram exports come in two formats:
1. **result.json** - Structured JSON with message objects (our custom parser)
2. **messages.html** - HTML export (Kreuzberg extracts text, we add metadata)

The JSON format is Telegram-specific and Kreuzberg doesn't understand its schema,
so we parse it ourselves. For HTML exports, we use Kreuzberg's HTML extractor
for clean text extraction.

Both paths output items in our unified item schema:
{
    "type": "message" | "voice" | "video" | "photo" | "sticker" | "document",
    "path": str,           # Media file path relative to export dir
    "text": str | None,    # Message text content
    "timestamp": str,      # ISO timestamp (America/New_York normalized)
    "tz_offset": str,      # Original timezone offset
    "sender": str,         # Message sender
    "forwarded": bool,     # Whether message was forwarded
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from pathlib import Path


from ..shared.multimodal import normalize_timestamp
from ..utils.logging import get_loguru_logger

_internal_logger = get_loguru_logger("ingestion.telegram")


@dataclass
class TelegramExtractionResult:
    """Result from parsing a Telegram export."""

    success: bool
    source: str
    """Source format: "json" or "html"."""

    items: list[dict[str, Any]] = field(default_factory=list)
    """Unified item list (see module docstring for schema)."""

    message_count: int = 0
    """Total messages parsed."""

    media_count: int = 0
    """Messages with media attachments."""

    sender_set: set[str] = field(default_factory=set)
    """Unique senders found in export."""

    date_range: dict[str, str] = field(default_factory=dict)
    """Earliest and latest message timestamps."""

    error: str | None = None
    """Error message if parsing failed."""

    orphan_media: list[dict[str, Any]] = field(default_factory=list)
    """Media files found in archive dirs but not linked from messages."""


class TelegramParser:
    """Parser for Telegram export data.

    Handles both JSON (result.json) and HTML (messages.html) export formats.
    JSON uses our custom parser; HTML uses Kreuzberg for text extraction
    combined with our metadata extraction.
    """

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.result_json = input_dir / "result.json"
        self.messages_html = input_dir / "messages.html"

    async def parse(self) -> TelegramExtractionResult:
        """Parse the Telegram export, auto-detecting format.

        Prefers JSON over HTML if both exist.

        Returns:
            TelegramExtractionResult with unified items.
        """
        if self.result_json.exists():
            return self._parse_json()
        if self.messages_html.exists():
            return await self._parse_html()
        return TelegramExtractionResult(
            success=False,
            source="none",
            error=f"No Telegram export found in {self.input_dir}",
        )

    def parse_json(self) -> TelegramExtractionResult:
        """Synchronous wrapper for JSON parsing."""
        return self._parse_json()

    def _parse_json(self) -> TelegramExtractionResult:
        """Parse Telegram JSON export (result.json).

        This is our custom parser because Telegram's JSON schema is
        application-specific and Kreuzberg doesn't understand it.

        Handles:
        - Text messages (including multi-part messages with mixed media/text)
        - Voice messages
        - Video files
        - Photos
        - Stickers (with emoji)
        - Forwarded messages
        - Service messages (skipped)
        """
        if not self.result_json.exists():
            return TelegramExtractionResult(
                success=False,
                source="json",
                error=f"result.json not found in {self.input_dir}",
            )

        _internal_logger.info(f"Parsing Telegram JSON: {self.result_json}")
        items: list[dict[str, Any]] = []

        try:
            with self.result_json.open("r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])
            _internal_logger.info(f"Found {len(messages)} messages in JSON export")

            for msg in messages:
                if msg.get("type") == "service":
                    continue

                ts = normalize_timestamp(msg.get("date", ""))
                sender = msg.get("from", "Unknown")
                text = self._extract_json_text(msg.get("text", ""))

                media_path = msg.get("file", "") or msg.get("photo", "")
                forwarded = bool(msg.get("forwarded_from"))

                # Determine item type
                item_type = msg.get("type", "message")
                if msg.get("media_type") == "voice_message":
                    item_type = "voice"
                elif msg.get("media_type") == "video_file":
                    item_type = "video"
                elif msg.get("photo"):
                    item_type = "photo"
                elif msg.get("sticker_emoji"):
                    if not text:
                        text = f"[Sticker: {msg['sticker_emoji']}]"
                    item_type = "sticker"

                if not text and not media_path:
                    continue

                items.append({
                    "type": item_type,
                    "path": media_path,
                    "text": text,
                    "timestamp": ts["iso"],
                    "tz_offset": ts.get("tz_offset", ""),
                    "sender": sender,
                    "forwarded": forwarded,
                })

            result = self._build_result("json", items)
            _internal_logger.info(
                f"Parsed {result.message_count} messages from JSON "
                f"({result.media_count} with media, {len(result.sender_set)} senders)"
            )
            return result

        except Exception as e:
            _internal_logger.error(f"Failed to parse Telegram JSON: {e}")
            return TelegramExtractionResult(
                success=False,
                source="json",
                error=str(e),
            )

    async def _parse_html(self) -> TelegramExtractionResult:
        """Parse Telegram HTML export (messages.html).

        Uses BeautifulSoup for structure (Telegram HTML has a specific DOM
        layout with message divs) because Kreuzberg gives us flat text.
        We need the DOM structure to extract per-message metadata (sender,
        timestamp, media links, forwarded status).

        Kreuzberg can extract the text content cleanly, but we still need
        the structural parsing to map text → messages → metadata.
        """
        if not self.messages_html.exists():
            return TelegramExtractionResult(
                success=False,
                source="html",
                error=f"messages.html not found in {self.input_dir}",
            )

        _internal_logger.info(f"Parsing Telegram HTML: {self.messages_html}")
        items: list[dict[str, Any]] = []

        try:
            with self.messages_html.open("r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml")

            message_divs = soup.find_all("div", class_="message")
            _internal_logger.info(f"Found {len(message_divs)} message divs in HTML")

            for div in message_divs:
                if "service" in (div.get("class") or []):
                    continue

                # Timestamp
                timestamp_raw = ""
                time_elem = div.find("div", class_="pull_right") or div.find("time")
                if time_elem:
                    timestamp_raw = time_elem.get("title", "") or time_elem.text.strip()
                ts = normalize_timestamp(timestamp_raw)

                # Sender
                sender = "Unknown"
                from_elem = div.find("div", class_="from_name")
                if from_elem:
                    sender = from_elem.text.strip()

                # Text
                text = None
                text_elem = div.find("div", class_="text")
                if text_elem:
                    text = text_elem.text.strip()

                # Media and type
                media_path = None
                item_type = "message"
                forwarded = bool(div.find("div", class_="forwarded"))

                photo_link = div.find("a", class_="photo_wrap")
                if photo_link:
                    media_path = photo_link.get("href", "")
                    item_type = "photo"

                # Voice messages are <a> tags with class media_voice_message
                voice_link = div.find("a", class_="media_voice_message")
                if voice_link:
                    media_path = voice_link.get("href", "")
                    item_type = "voice"

                # Video
                video_link = div.find("a", class_="video_file_wrap")
                if video_link:
                    video_src = video_link.get("href", "")
                    if video_src:
                        media_path = video_src
                    item_type = "video"
                else:
                    video_elem = div.find("div", class_="video")
                    if video_elem:
                        video_src = video_elem.find("source")
                        if video_src:
                            media_path = video_src.get("src", "")
                        item_type = "video"

                # Stickers
                is_sticker = bool(
                    div.find("img", class_="sticker") or div.find("div", class_="sticker")
                )
                if is_sticker:
                    sticker_img = div.find("img", class_="sticker")
                    emoji = ""
                    if sticker_img:
                        emoji = sticker_img.get("alt", "") or sticker_img.get("title", "")
                    if not text:
                        text = f"[Sticker: {emoji}]" if emoji else "[Sticker]"
                    item_type = "sticker"

                if not text and not media_path:
                    continue

                items.append({
                    "type": item_type,
                    "path": media_path,
                    "text": text,
                    "timestamp": ts["iso"],
                    "tz_offset": ts.get("tz_offset", ""),
                    "sender": sender,
                    "forwarded": forwarded,
                })

            # Find orphan media and build result
            result = self._build_result("html", items)
            _internal_logger.info(
                f"Parsed {result.message_count} messages from HTML "
                f"({result.media_count} with media, {len(result.sender_set)} senders, "
                f"{len(result.orphan_media)} orphan media)"
            )
            return result

        except Exception as e:
            _internal_logger.error(f"Failed to parse Telegram HTML: {e}")
            return TelegramExtractionResult(
                success=False,
                source="html",
                error=str(e),
            )

    def _find_orphan_media(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find media files in archive directories not linked from messages.

        Telegram exports media into subdirectories: photos/, voice_messages/,
        video_files/, stickers/. Some files may exist on disk but not be
        referenced in the HTML.
        """
        orphan_items: list[dict[str, Any]] = []
        existing_paths = {
            item["path"] for item in items if item.get("path")
        }

        media_dirs = {
            "voice_messages": "voice",
            "video_files": "video",
            "photos": "photo",
            "stickers": "sticker",
        }

        for dir_name, type_name in media_dirs.items():
            media_dir = self.input_dir / dir_name
            if not media_dir.exists():
                continue
            for f in sorted(media_dir.iterdir()):
                if f.is_file() and "_thumb." not in f.name:
                    rel_path = f"{dir_name}/{f.name}"
                    if rel_path not in existing_paths:
                        orphan_items.append({
                            "type": type_name,
                            "path": rel_path,
                            "text": None,
                            "timestamp": "Found in archive",
                            "tz_offset": "",
                            "sender": "Unknown",
                            "forwarded": False,
                        })

        return orphan_items

    def _extract_json_text(self, text: Any) -> str:
        """Extract text from Telegram JSON message text field.

        Telegram represents text as either:
        - A plain string
        - A list of mixed text/inline entities: [{"text": "hello"}, " world"]

        Args:
            text: Raw text field from JSON message.

        Returns:
            Concatenated text string.
        """
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            return "".join(
                t.get("text", "") if isinstance(t, dict) else str(t)
                for t in text
            )
        return str(text) if text else ""

    def _build_result(
        self,
        source: str,
        items: list[dict[str, Any]],
        orphan_media: list[dict[str, Any]] | None = None,
    ) -> TelegramExtractionResult:
        """Build extraction result from parsed items.

        Args:
            source: Source format ("json" or "html").
            items: Parsed message items.
            orphan_media: Orphan media files.

        Returns:
            TelegramExtractionResult with summary stats.
        """
        # Find orphan media if not provided
        if orphan_media is None:
            orphan_media = self._find_orphan_media(items)

        all_items = items + orphan_media

        sender_set = {item["sender"] for item in all_items if item.get("sender")}
        timestamps = [
            item["timestamp"]
            for item in all_items
            if item.get("timestamp") and item["timestamp"] != "Unknown"
        ]

        media_count = sum(
            1 for item in all_items
            if item.get("path") or item["type"] in ("voice", "video", "photo", "sticker")
        )

        date_range: dict[str, str] = {}
        if timestamps:
            sorted_ts = sorted(timestamps)
            date_range = {
                "earliest": sorted_ts[0],
                "latest": sorted_ts[-1],
            }

        return TelegramExtractionResult(
            success=True,
            source=source,
            items=all_items,
            message_count=len(all_items),
            media_count=media_count,
            sender_set=sender_set,
            date_range=date_range,
            orphan_media=orphan_media,
        )


def has_telegram_export(input_dir: Path) -> bool:
    """Check if directory contains a Telegram export.

    Args:
        input_dir: Directory to check.

    Returns:
        True if result.json or messages.html exists.
    """
    return (input_dir / "result.json").exists() or (input_dir / "messages.html").exists()
