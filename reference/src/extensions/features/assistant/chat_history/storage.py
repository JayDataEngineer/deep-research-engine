"""Chat History Storage Service

Manages chat history with JSONL format (SillyTavern compatible).
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

logger = logging.getLogger(__name__)


def _makedirs_safe(path: Path) -> None:
    """
    Thread-safe directory creation.

    Uses os.makedirs with exist_ok=True which is atomic on most systems.
    Falls back to try/except for edge cases.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        # Handle race condition: directory was created by another thread
        if not path.is_dir():
            raise


# =============================================================================
# Data Models - SillyTavern Chat Format
# =============================================================================

@dataclass
class ChatMessage:
    """Chat message model compatible with SillyTavern format."""
    name: str  # Character name or "user"
    is_user: bool
    is_name: bool = False  # Whether this is a system/narrator message
    send_date: Optional[int] = None  # Unix timestamp
    extra: Dict[str, Any] = field(default_factory=dict)

    # Message content (one of these should be set)
    mes: str = ""  # Message text

    # Optional fields
    title: str = ""  # For title generation
    gen_started: Optional[int] = None  # Generation start time
    gen_finished: Optional[int] = None  # Generation finish time
    swipe_id: int = 0  # For swipe/regeneration tracking
    swipe_count: int = 1  # Total number of swipes
    generation_id: Optional[str] = None  # Unique generation ID
    api_id: Optional[int] = None  # Message format ID

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
        if self.send_date is None:
            self.send_date = int(datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL export."""
        d = asdict(self)
        # Remove empty optional fields
        return {k: v for k, v in d.items() if v or v is False or v == 0}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create from dictionary (JSONL import)."""
        # Handle both old and new SillyTavern formats
        return cls(
            name=data.get("name", ""),
            is_user=data.get("is_user", False),
            is_name=data.get("is_name", False),
            send_date=data.get("send_date") or data.get("send_date") or int(datetime.now().timestamp()),
            mes=data.get("mes", data.get("content", data.get("text", ""))),
            title=data.get("title", ""),
            gen_started=data.get("gen_started"),
            gen_finished=data.get("gen_finished"),
            swipe_id=data.get("swipe_id", 0),
            swipe_count=data.get("swipe_count", 1),
            generation_id=data.get("generation_id"),
            api_id=data.get("api_id"),
            extra=data.get("extra", {}),
        )


@dataclass
class ChatMetadata:
    """Chat metadata for header info."""
    chat_name: str = ""
    character_name: str = ""
    create_date: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    last_update: int = field(default_factory=lambda: int(datetime.now().timestamp()))

    # Optional SillyTavern fields
    chat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_group_chat: bool = False
    members: List[str] = field(default_factory=list)

    # Stats
    message_count: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMetadata":
        """Create from dictionary."""
        return cls(
            chat_name=data.get("chat_name", ""),
            character_name=data.get("character_name", ""),
            create_date=data.get("create_date") or data.get("create_date") or int(datetime.now().timestamp()),
            last_update=data.get("last_update") or data.get("last_update") or int(datetime.now().timestamp()),
            chat_id=data.get("chat_id", str(uuid.uuid4())),
            is_group_chat=data.get("is_group_chat", False),
            members=data.get("members", []),
            message_count=data.get("message_count", 0),
            user_message_count=data.get("user_message_count", 0),
            assistant_message_count=data.get("assistant_message_count", 0),
        )


# =============================================================================
# Storage Service
# =============================================================================

class ChatHistoryStorage:
    """Service for managing chat history storage."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize chat history storage.

        Args:
            base_dir: Base directory for chat data.
                     Defaults to inputs/chats/
        """
        if base_dir is None:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            base_dir = get_inputs_root_dir()

        self.base_dir = Path(base_dir)
        self.chats_dir = self.base_dir / "chats"
        _makedirs_safe(self.chats_dir)

        logger.info(f"Chat History storage initialized at: {self.chats_dir}")

    # ========================================================================
    # Chat CRUD Operations
    # ========================================================================

    def save_chat(
        self,
        messages: List[ChatMessage],
        metadata: Optional[ChatMetadata] = None,
        format: str = "jsonl"
    ) -> Path:
        """Save a chat to JSONL format.

        Args:
            messages: List of chat messages
            metadata: Optional chat metadata
            format: File format (jsonl or json)

        Returns:
            Path to saved file
        """
        if metadata is None:
            metadata = ChatMetadata(
                chat_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

        # Update metadata stats
        metadata.message_count = len(messages)
        metadata.user_message_count = sum(1 for m in messages if m.is_user)
        metadata.assistant_message_count = sum(1 for m in messages if not m.is_user)
        metadata.last_update = int(datetime.now().timestamp())

        # Create safe filename
        safe_name = self._sanitize_filename(metadata.chat_name or metadata.character_name or "chat")
        character_dir = self.chats_dir / safe_name
        _makedirs_safe(character_dir)

        filename = f"{safe_name}.jsonl"
        filepath = character_dir / filename

        # Write JSONL
        with open(filepath, "w", encoding="utf-8") as f:
            for message in messages:
                json.dump(message.to_dict(), f, ensure_ascii=False)
                f.write("\n")

        # Save metadata as separate file
        metadata_path = character_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved chat: {filepath}")
        return filepath

    def load_chat(
        self,
        chat_name: str,
        format: str = "jsonl"
    ) -> Optional[tuple[List[ChatMessage], ChatMetadata]]:
        """Load a chat from JSONL format.

        Args:
            chat_name: Chat name (character name usually)
            format: File format

        Returns:
            Tuple of (messages, metadata) or None if not found
        """
        safe_name = self._sanitize_filename(chat_name)
        character_dir = self.chats_dir / safe_name

        if not character_dir.exists():
            return None

        # Load JSONL
        filepath = character_dir / f"{safe_name}.jsonl"
        if not filepath.exists():
            filepath = character_dir / f"{safe_name}.json"

        if not filepath.exists():
            return None

        messages = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    messages.append(ChatMessage.from_dict(data))

        # Load metadata
        metadata_path = character_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_data = json.load(f)
                metadata = ChatMetadata.from_dict(metadata_data)
        else:
            # Create metadata from messages
            metadata = ChatMetadata(
                chat_name=chat_name,
                message_count=len(messages),
                user_message_count=sum(1 for m in messages if m.is_user),
                assistant_message_count=sum(1 for m in messages if not m.is_user),
            )

        return messages, metadata

    def list_chats(self) -> List[Dict[str, Any]]:
        """List all available chats.

        Returns:
            List of chat info dictionaries
        """
        chats = []
        for character_dir in self.chats_dir.iterdir():
            if not character_dir.is_dir():
                continue

            # Try to load metadata
            metadata_path = character_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    chats.append({
                        "chat_name": metadata.get("chat_name", character_dir.name),
                        "character_name": metadata.get("character_name", ""),
                        "message_count": metadata.get("message_count", 0),
                        "create_date": metadata.get("create_date"),
                        "last_update": metadata.get("last_update"),
                        "directory": str(character_dir),
                    })
            else:
                # Count messages in JSONL
                jsonl_files = list(character_dir.glob("*.jsonl")) + list(character_dir.glob("*.json"))
                if jsonl_files:
                    message_count = 0
                    with open(jsonl_files[0], "r") as f:
                        for _ in f:
                            message_count += 1
                    chats.append({
                        "chat_name": character_dir.name,
                        "character_name": character_dir.name,
                        "message_count": message_count,
                        "directory": str(character_dir),
                    })

        return sorted(chats, key=lambda x: x.get("last_update", 0), reverse=True)

    def delete_chat(self, chat_name: str) -> bool:
        """Delete a chat.

        Args:
            chat_name: Chat name

        Returns:
            True if deleted, False if not found
        """
        safe_name = self._sanitize_filename(chat_name)
        character_dir = self.chats_dir / safe_name

        if not character_dir.exists():
            return False

        # Remove directory and all files
        import shutil
        shutil.rmtree(character_dir)
        logger.info(f"Deleted chat: {character_dir}")
        return True

    # ========================================================================
    # Import/Export
    # ========================================================================

    def import_chat(
        self,
        source: Union[str, Path],
        format: str = "auto"
    ) -> Optional[tuple[List[ChatMessage], ChatMetadata]]:
        """Import chat from file.

        Args:
            source: Path to chat file (JSONL, JSON)
            format: Format or 'auto' to detect

        Returns:
            Tuple of (messages, metadata) or None
        """
        source_path = Path(source)

        if not source_path.exists():
            source_path = Path.cwd() / source
            if not source_path.exists():
                logger.error(f"Source file not found: {source}")
                return None

        # Detect format
        if format == "auto":
            if source_path.suffix == ".jsonl":
                format = "jsonl"
            elif source_path.suffix == ".json":
                format = "json"
            else:
                format = "jsonl"  # Default

        # Load messages
        messages = []
        if format == "jsonl":
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        messages.append(ChatMessage.from_dict(data))
        else:  # JSON
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for msg_data in data:
                        messages.append(ChatMessage.from_dict(msg_data))
                elif isinstance(data, dict) and "messages" in data:
                    for msg_data in data["messages"]:
                        messages.append(ChatMessage.from_dict(msg_data))

        if not messages:
            logger.error(f"No messages found in file: {source_path}")
            return None

        # Create metadata
        metadata = ChatMetadata(
            chat_name=source_path.stem,
            message_count=len(messages),
            user_message_count=sum(1 for m in messages if m.is_user),
            assistant_message_count=sum(1 for m in messages if not m.is_user),
        )

        # Save to storage
        self.save_chat(messages, metadata)

        return messages, metadata

    def export_chat(
        self,
        chat_name: str,
        output_path: Optional[Path] = None,
        format: str = "jsonl",
        include_metadata: bool = True
    ) -> Optional[Path]:
        """Export chat to file.

        Args:
            chat_name: Chat name
            output_path: Optional output path (default: chats_dir/export/)
            format: Export format (jsonl, json)
            include_metadata: Whether to include metadata file

        Returns:
            Path to exported file or None
        """
        result = self.load_chat(chat_name)
        if not result:
            logger.error(f"Chat not found: {chat_name}")
            return None

        messages, metadata = result

        # Determine output path
        if output_path is None:
            output_path = Path.cwd() / f"{chat_name}_export.{format}"
        else:
            output_path = Path(output_path)

        # Create parent directory (thread-safe)
        _makedirs_safe(output_path.parent)

        # Write messages
        with open(output_path, "w", encoding="utf-8") as f:
            if format == "jsonl":
                for message in messages:
                    json.dump(message.to_dict(), f, ensure_ascii=False)
                    f.write("\n")
            else:  # JSON
                json.dump([m.to_dict() for m in messages], f, indent=2, ensure_ascii=False)

        # Write metadata if requested
        if include_metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Exported chat to: {output_path}")
        return output_path

    # ========================================================================
    # Utilities
    # ========================================================================

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a name for safe filename usage."""
        unsafe = '<>:"/\\|?*'
        for char in unsafe:
            name = name.replace(char, "_")
        return name.strip()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored chats."""
        chats = self.list_chats()
        total_messages = sum(c.get("message_count", 0) for c in chats)

        return {
            "directory": str(self.chats_dir),
            "chat_count": len(chats),
            "total_messages": total_messages,
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_chat_storage: Optional[ChatHistoryStorage] = None


def get_chat_storage(base_dir: Optional[Path] = None) -> ChatHistoryStorage:
    """Get the singleton chat storage instance."""
    global _chat_storage
    if _chat_storage is None:
        _chat_storage = ChatHistoryStorage(base_dir=base_dir)
    return _chat_storage
