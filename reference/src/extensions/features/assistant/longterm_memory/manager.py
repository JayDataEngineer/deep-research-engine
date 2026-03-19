"""Long-Term Memory Manager

Manages long-term memory storage for important information across conversations.
Distinct from lorebooks (which are character/world-specific).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """A long-term memory item."""
    id: str
    content: str
    category: str  # fact, preference, event, relationship, etc.
    source_chat: str = ""
    character_name: str = ""
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(**data)


class LongTermMemoryManager:
    """Manages long-term memory storage."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            storage_path = get_inputs_root_dir() / "memory"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.storage_path / "memories.json"

        self._memories: Dict[str, Memory] = {}
        self._load_memories()

        logger.info(f"Long-Term Memory Manager initialized at: {self.storage_path}")

    def _load_memories(self) -> None:
        """Load memories from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    for mem_data in data:
                        memory = Memory.from_dict(mem_data)
                        self._memories[memory.id] = memory
                logger.debug(f"Loaded {len(self._memories)} memories")
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
                self._memories = {}

    def _save_memories(self) -> None:
        """Save memories to disk."""
        try:
            data = [m.to_dict() for m in self._memories.values()]
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

    def add_memory(
        self,
        content: str,
        category: str = "fact",
        source_chat: str = "",
        character_name: str = "",
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> Memory:
        """Add a memory.

        Args:
            content: The memory content
            category: Memory category (fact, preference, event, relationship)
            source_chat: Optional source chat ID
            character_name: Optional associated character
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags

        Returns:
            The created Memory
        """
        memory = Memory(
            id=str(uuid.uuid4())[:8],
            content=content,
            category=category,
            source_chat=source_chat,
            character_name=character_name,
            importance=max(0.0, min(1.0, importance)),
            tags=tags or []
        )

        self._memories[memory.id] = memory
        self._save_memories()
        logger.info(f"Added memory: {memory.id}")
        return memory

    def list_memories(
        self,
        category: Optional[str] = None,
        character: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 50
    ) -> List[Memory]:
        """List memories.

        Args:
            category: Filter by category
            character: Filter by character name
            tags: Filter by tags (any match)
            min_importance: Minimum importance score
            limit: Max memories to return

        Returns:
            List of Memories
        """
        memories = list(self._memories.values())

        if category:
            memories = [m for m in memories if m.category == category]
        if character:
            memories = [m for m in memories if m.character_name == character]
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        memories = [m for m in memories if m.importance >= min_importance]

        # Sort by importance and created_at
        memories.sort(key=lambda x: (x.importance, x.created_at), reverse=True)

        return memories[:limit]

    def search_memories(self, query: str, limit: int = 20) -> List[Memory]:
        """Search memories by content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching Memories
        """
        query_lower = query.lower()
        results = []

        for memory in self._memories.values():
            # Search in content and tags
            if (query_lower in memory.content.lower() or
                any(query_lower in tag.lower() for tag in memory.tags)):
                results.append(memory)

        # Sort by importance
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:limit]

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        return self._memories.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save_memories()
            return True
        return False

    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return sorted(set(m.category for m in self._memories.values()))

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        memories = list(self._memories.values())
        categories = self.get_categories()

        # Count by category
        by_category = {}
        for memory in memories:
            by_category[memory.category] = by_category.get(memory.category, 0) + 1

        return {
            "total_memories": len(memories),
            "categories": categories,
            "by_category": by_category,
            "storage_path": str(self.storage_path),
        }
