"""Memory Manager - High-level interface for Companion AI long-term memory.

Provides convenient methods for storing, retrieving, and searching memories
using the SQLiteStore backend.

Namespace Structure:
- memories/{user_id}/episodes/{memory_id}
- memories/{user_id}/facts/{memory_id}
- memories/{user_id}/preferences/{memory_id}
- memories/{user_id}/relationships/{memory_id}
- memories/{user_id}/goals/{memory_id}
- memories/{user_id}/events/{memory_id}
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .memory_schema import (
    Memory,
    MemoryType,
    MemoryImportance,
    MemorySearchQuery,
    MemorySearchResult,
    EpisodeMemory,
    FactMemory,
    PreferenceMemory,
    RelationshipMemory,
    GoalMemory,
    EventMemory,
    MemorySummary,
)
from .sqlite_store import SQLiteStore, get_memory_store

logger = logging.getLogger(__name__)


class CompanionMemoryManager:
    """High-level memory manager for Companion AI.

    Provides a clean API for the companion brain to:
    - Store new memories from conversations
    - Retrieve relevant memories for context
    - Search memories by semantic similarity
    - Manage memory lifecycle

    Example:
        ```python
        manager = CompanionMemoryManager(user_id="alice")
        # Store a memory
        manager.store_memory(
            memory_type=MemoryType.PREFERENCE,
            content="User loves coffee, especially lattes",
            importance=MemoryImportance.MEDIUM,
        )
        # Search memories
        results = manager.search_memories("what does user like to drink")
        ```
    """

    def __init__(
        self,
        user_id: str = "default",
        store: Optional[SQLiteStore] = None,
    ):
        """Initialize the memory manager.

        Args:
            user_id: User identifier for namespacing memories
            store: Optional custom store (uses global default if not provided)
        """
        self.user_id = user_id
        self.store = store or get_memory_store()
        self._namespace_prefix = ("memories", user_id)

    def _make_namespace(self, memory_type: MemoryType) -> Tuple[str, ...]:
        """Create full namespace tuple for a memory type."""
        return (*self._namespace_prefix, memory_type.value)

    def _generate_memory_id(self) -> str:
        """Generate a unique memory ID."""
        return f"mem_{uuid.uuid4().hex[:12]}_{int(time.time())}"

    # =========================================================================
    # Store Operations
    # =========================================================================

    def store_memory(
        self,
        memory_type: MemoryType,
        content: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        extra_data: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> Memory:
        """Store a new memory.

        Args:
            memory_type: Type of memory to store
            content: Main content/text of the memory
            importance: How important this memory is
            extra_data: Additional type-specific data
            keywords: Keywords for matching
            thread_id: Thread where this memory was created

        Returns:
            The created Memory object
        """
        memory_id = self._generate_memory_id()
        now = datetime.utcnow().isoformat()

        # Base memory data
        memory_data = {
            "memory_id": memory_id,
            "memory_type": memory_type.value,
            "content": content,
            "importance": importance.value,
            "created_at": now,
            "updated_at": now,
            "keywords": keywords or [],
            "user_id": self.user_id,
            "thread_id": thread_id,
            "access_count": 0,
            "extra_data": extra_data or {},
        }

        # Create type-specific memory model
        memory = self._create_memory_model(memory_type, memory_data)

        # Store in database
        namespace = self._make_namespace(memory_type)
        self.store.put(namespace, memory_id, memory.dict())

        logger.info(
            f"[MEMORY] Stored {memory_type.value}: {memory_id} - {content[:50]}..."
        )
        return memory

    def _create_memory_model(self, memory_type: MemoryType, data: Dict[str, Any]) -> Memory:
        """Create the appropriate memory model class."""
        models = {
            MemoryType.EPISODE: EpisodeMemory,
            MemoryType.FACT: FactMemory,
            MemoryType.PREFERENCE: PreferenceMemory,
            MemoryType.RELATIONSHIP: RelationshipMemory,
            MemoryType.GOAL: GoalMemory,
            MemoryType.EVENT: EventMemory,
        }
        model_class = models.get(memory_type, Memory)
        return model_class(**data)

    # =========================================================================
    # Retrieve Operations
    # =========================================================================

    def get_memory(self, memory_type: MemoryType, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID.

        Args:
            memory_type: Type of memory
            memory_id: Memory ID

        Returns:
            Memory object or None if not found
        """
        namespace = self._make_namespace(memory_type)
        data = self.store.get(namespace, memory_id)

        if data:
            return self._create_memory_model(memory_type, data)
        return None

    def get_recent_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Get recent memories.

        Args:
            memory_type: Optional type filter
            limit: Max memories to return

        Returns:
            List of memories ordered by recency
        """
        if memory_type:
            namespace = self._make_namespace(memory_type)
            results = list(self.store.list(namespace, limit=limit))
        else:
            # Search across all types
            results = []
            for mtype in MemoryType:
                namespace = self._make_namespace(mtype)
                results.extend(list(self.store.list(namespace, limit=limit)))

        # Convert to Memory objects
        memories = []
        for namespace_tuple, key, value in results:
            mtype = MemoryType(namespace_tuple[-1])  # Last element is type
            memories.append(self._create_memory_model(mtype, value))

        # Sort by created_at
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def get_important_memories(
        self,
        min_importance: MemoryImportance = MemoryImportance.HIGH,
        limit: int = 20,
    ) -> List[Memory]:
        """Get important memories.

        Args:
            min_importance: Minimum importance level
            limit: Max memories to return

        Returns:
            List of important memories
        """
        memories = []
        for mtype in MemoryType:
            namespace = self._make_namespace(mtype)
            for _, _, value in self.store.list(namespace, limit=100):
                memory = self._create_memory_model(mtype, value)
                if memory.importance >= min_importance:
                    memories.append(memory)

        # Sort by importance and access frequency
        memories.sort(
            key=lambda m: (m.importance, m.access_count),
            reverse=True,
        )
        return memories[:limit]

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
    ) -> List[MemorySearchResult]:
        """Search memories by semantic similarity.

        Args:
            query: Natural language search query
            memory_types: Optional list of types to search
            limit: Max results to return

        Returns:
            List of search results with relevance scores
        """
        results = []

        if memory_types:
            # Search specific types
            for mtype in memory_types:
                namespace = self._make_namespace(mtype)
                results.extend(self._search_namespace(namespace, query, limit))
        else:
            # Search all types
            for mtype in MemoryType:
                namespace = self._make_namespace(mtype)
                results.extend(self._search_namespace(namespace, query, limit))

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def _search_namespace(
        self,
        namespace: Tuple[str, ...],
        query: str,
        limit: int,
    ) -> List[MemorySearchResult]:
        """Search a specific namespace."""
        results = []
        for ns_tuple, key, value in self.store.search(
            prefix=namespace,
            query=query,
            limit=limit,
        ):
            mtype = MemoryType(ns_tuple[-1])
            memory = self._create_memory_model(mtype, value)

            # Extract relevance score if present
            relevance = value.get("_relevance", 0.5)

            results.append(
                MemorySearchResult(
                    memory=memory,
                    relevance_score=relevance,
                    match_reason=self._explain_match(memory, query),
                )
            )
        return results

    def _explain_match(self, memory: Memory, query: str) -> str:
        """Generate explanation for why a memory matched."""
        reasons = []

        # Check keyword match
        query_lower = query.lower()
        if query_lower in memory.content.lower():
            reasons.append("content contains query")

        # Check keyword match
        for keyword in memory.keywords:
            if keyword.lower() in query_lower:
                reasons.append(f"keyword '{keyword}' matches")
                break

        return "; ".join(reasons) if reasons else "semantic similarity"

    # =========================================================================
    # Memory Summary
    # =========================================================================

    def get_memory_summary(self) -> MemorySummary:
        """Get a summary of the user's memory state.

        Returns:
            MemorySummary with stats and key memories
        """
        # Count by type
        memories_by_type: Dict[str, int] = {}
        memory_density: Dict[str, int] = {}

        for mtype in MemoryType:
            namespace = self._make_namespace(mtype)
            count = 0
            for _, _, value in self.store.list(namespace, limit=1000):
                count += 1
                # Extract keywords for density
                memory = self._create_memory_model(mtype, value)
                for kw in memory.keywords:
                    memory_density[kw] = memory_density.get(kw, 0) + 1

            memories_by_type[mtype.value] = count

        total = sum(memories_by_type.values())

        return MemorySummary(
            total_memories=total,
            memories_by_type=memories_by_type,
            most_recent_memories=self.get_recent_memories(limit=5),
            important_memories=self.get_important_memories(limit=5),
            memory_density=dict(sorted(memory_density.items(), key=lambda x: -x[1])[:20]),
        )

    # =========================================================================
    # Context Building for Companion
    # =========================================================================

    def build_context(
        self,
        current_input: str,
        max_memories: int = 5,
    ) -> str:
        """Build memory context for the companion brain.

        Searches for relevant memories and formats them as context
        for the LLM.

        Args:
            current_input: Current user input for relevance
            max_memories: Maximum memories to include

        Returns:
            Formatted memory context string
        """
        # Get relevant memories
        search_results = self.search_memories(
            query=current_input,
            limit=max_memories,
        )

        # Also get recent important memories
        important = self.get_important_memories(
            min_importance=MemoryImportance.HIGH,
            limit=3,
        )

        # Combine results
        context_parts = []
        context_parts.append("## Relevant Memories\n")

        for result in search_results:
            mem = result.memory
            context_parts.append(
                f"- [{mem.memory_type.value}] {mem.content} "
                f"(relevance: {result.relevance_score:.2f})"
            )

        for mem in important:
            if mem.content not in [r.memory.content for r in search_results]:
                context_parts.append(
                    f"- [{mem.memory_type.value}] {mem.content} (important)"
                )

        return "\n".join(context_parts) if len(context_parts) > 1 else ""

    # =========================================================================
    # Update and Delete
    # =========================================================================

    def update_memory_importance(
        self,
        memory_type: MemoryType,
        memory_id: str,
        new_importance: MemoryImportance,
    ) -> bool:
        """Update the importance level of a memory.

        Args:
            memory_type: Type of memory
            memory_id: Memory ID
            new_importance: New importance level

        Returns:
            True if updated, False if not found
        """
        memory = self.get_memory(memory_type, memory_id)
        if not memory:
            return False

        memory.importance = new_importance
        memory.updated_at = datetime.utcnow().isoformat()

        namespace = self._make_namespace(memory_type)
        self.store.put(namespace, memory_id, memory.dict())
        return True

    def delete_memory(self, memory_type: MemoryType, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_type: Type of memory
            memory_id: Memory ID

        Returns:
            True if deleted, False if not found
        """
        namespace = self._make_namespace(memory_type)
        existing = self.store.get(namespace, memory_id)

        if existing is None:
            return False

        self.store.delete(namespace, memory_id)
        logger.info(f"[MEMORY] Deleted {memory_type.value}/{memory_id}")
        return True

    def clear_all_memories(self) -> int:
        """Clear all memories for this user.

        WARNING: This is irreversible!

        Returns:
            Number of memories deleted
        """
        return self.store.clear_namespace(self._namespace_prefix)


# =============================================================================
# Factory Functions
# =============================================================================

_managers: Dict[str, CompanionMemoryManager] = {}
_manager_lock = threading.Lock()


def get_memory_manager(user_id: str = "default") -> CompanionMemoryManager:
    """Get or create a memory manager for a user.

    Args:
        user_id: User identifier

    Returns:
        CompanionMemoryManager instance
    """
    import threading

    with _manager_lock:
        if user_id not in _managers:
            _managers[user_id] = CompanionMemoryManager(user_id=user_id)
        return _managers[user_id]
