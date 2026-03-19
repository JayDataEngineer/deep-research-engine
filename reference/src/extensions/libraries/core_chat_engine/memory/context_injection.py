"""Memory Context Injection - Integrates relevant memories into conversation context.

This module provides the bridge between long-term memory storage and the LLM context.
It searches for relevant memories and formats them for injection into the conversation.

Key Features:
- Semantic memory search based on current input
- Automatic memory prioritization (importance + relevance)
- Token-efficient memory formatting
- Memory-aware system prompt injection
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ..memory.memory_manager import get_memory_manager
from ..memory.memory_schema import MemoryType, MemoryImportance
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Context Builder
# =============================================================================

class MemoryContextBuilder:
    """Builds memory-aware context for companion conversations.

    This class handles the integration of long-term memories into the LLM context,
    ensuring the companion can recall relevant information from past conversations.

    Example:
        ```python
        builder = MemoryContextBuilder(user_id="alice")

        # Get memory context for current input
        memory_context = builder.build_context(
            current_input="What should I make for dinner?",
            max_tokens=500,
        )

        # Inject into conversation
        messages = [SystemMessage(content=memory_context), ...]
        ```
    """

    def __init__(
        self,
        user_id: str,
        max_context_memories: int = 5,
        min_importance: MemoryImportance = MemoryImportance.MEDIUM,
    ):
        """Initialize the memory context builder.

        Args:
            user_id: User ID for memory namespacing
            max_context_memories: Maximum number of memories to include in context
            min_importance: Minimum importance level for automatic injection
        """
        self.user_id = user_id
        self.max_context_memories = max_context_memories
        self.min_importance = min_importance

        # Get memory manager
        self.memory_manager = get_memory_manager(user_id=user_id)

        logger.info(f"[MEMORY CONTEXT] Initialized for user={user_id}, max_memories={max_context_memories}")

    def build_context(
        self,
        current_input: str,
        max_tokens: int = 500,
        include_recent: bool = True,
    ) -> str:
        """Build memory context for the current input.

        Searches for relevant memories and formats them for LLM injection.

        Args:
            current_input: Current user message/input for relevance matching
            max_tokens: Approximate token budget for memory context
            include_recent: Whether to include recent important memories

        Returns:
            Formatted memory context string (empty if no relevant memories)
        """
        try:
            # Step 1: Search for relevant memories
            relevant_memories = self.memory_manager.search_memories(
                query=current_input,
                limit=self.max_context_memories,
            )

            # Step 2: Get recent important memories (if enabled)
            recent_memories = []
            if include_recent:
                recent_memories = self.memory_manager.get_important_memories(
                    min_importance=self.min_importance,
                    limit=3,
                )

            # Step 3: Combine and deduplicate
            all_memories = self._deduplicate_memories(
                relevant_memories + recent_memories
            )

            if not all_memories:
                logger.debug(f"[MEMORY CONTEXT] No memories found for input: {current_input[:50]}...")
                return ""

            # Step 4: Format memories efficiently
            memory_context = self._format_memories(all_memories, max_tokens)

            logger.info(f"[MEMORY CONTEXT] Built context with {len(all_memories)} memories ({len(memory_context)} chars)")
            return memory_context

        except Exception as e:
            logger.error(f"[MEMORY CONTEXT] Failed to build context: {e}")
            return ""

    def build_system_message(
        self,
        current_input: str,
        max_tokens: int = 500,
    ) -> Optional[SystemMessage]:
        """Build a SystemMessage with memory context.

        This is the preferred way to inject memories into LangChain conversations.

        Args:
            current_input: Current user input for relevance matching
            max_tokens: Token budget for memory context

        Returns:
            SystemMessage with memory context, or None if no relevant memories
        """
        context = self.build_context(
            current_input=current_input,
            max_tokens=max_tokens,
        )

        if not context:
            return None

        return SystemMessage(content=context)

    def _deduplicate_memories(
        self,
        memories_with_results: List[Any],
    ) -> List[Any]:
        """Remove duplicate memories from the list.

        Args:
            memories_with_results: List of MemorySearchResult or Memory objects

        Returns:
            Deduplicated list of Memory objects
        """
        seen_ids = set()
        unique_memories = []

        for item in memories_with_results:
            # Extract memory object (handle both Memory and MemorySearchResult)
            if hasattr(item, 'memory'):
                memory = item.memory
            else:
                memory = item

            # Deduplicate by ID
            if memory.memory_id not in seen_ids:
                seen_ids.add(memory.memory_id)
                unique_memories.append(memory)

        return unique_memories

    def _format_memories(
        self,
        memories: List[Any],
        max_tokens: int,
    ) -> str:
        """Format memories efficiently for LLM context.

        Uses compact formatting to minimize token usage while maintaining
        readability for the LLM.

        Args:
            memories: List of Memory objects
            max_tokens: Token budget (approximate, based on character count)

        Returns:
            Formatted memory context string
        """
        if not memories:
            return ""

        # Sort by importance and relevance
        memories.sort(
            key=lambda m: (m.importance, m.access_count),
            reverse=True,
        )

        # Build context sections
        context_parts = ["## Relevant Information About the User\n"]

        # Group by type for better organization
        by_type: Dict[str, List[Any]] = {}
        for memory in memories[:self.max_context_memories]:
            mem_type = memory.memory_type.value
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(memory)

        # Format each type
        type_order = ["preference", "fact", "goal", "relationship", "event", "episode"]

        for mem_type in type_order:
            if mem_type not in by_type:
                continue

            type_label = mem_type.capitalize() + "s"
            context_parts.append(f"### {type_label}\n")

            for memory in by_type[mem_type]:
                # Compact format: "• Content [importance]"
                importance_mark = self._get_importance_mark(memory.importance)
                context_parts.append(f"• {memory.content} {importance_mark}\n")

            context_parts.append("\n")

        # Check token budget (rough estimate: 1 token ≈ 4 characters)
        full_context = "".join(context_parts)
        estimated_tokens = len(full_context) // 4

        if estimated_tokens > max_tokens:
            # Trim to fit budget
            ratio = max_tokens / estimated_tokens
            target_chars = int(len(full_context) * ratio * 0.9)
            full_context = full_context[:target_chars] + "\n..."

            logger.info(f"[MEMORY CONTEXT] Trimmed context from {estimated_tokens} to ~{max_tokens} tokens")

        return full_context

    def _get_importance_mark(self, importance: MemoryImportance) -> str:
        """Get a visual mark for importance level."""
        marks = {
            MemoryImportance.CRITICAL: "‼️",
            MemoryImportance.HIGH: "❗",
            MemoryImportance.MEDIUM: "❕",
            MemoryImportance.LOW: "",
        }
        return marks.get(importance, "")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the user's memory state.

        Useful for debugging and memory management UIs.

        Returns:
            Dict with memory statistics and key memories
        """
        try:
            summary = self.memory_manager.get_memory_summary()
            return {
                "total_memories": summary.total_memories,
                "by_type": summary.memories_by_type,
                "recent": [m.content for m in summary.most_recent_memories],
                "important": [m.content for m in summary.important_memories],
                "density": summary.memory_density,
            }
        except Exception as e:
            logger.error(f"[MEMORY CONTEXT] Failed to get summary: {e}")
            return {"error": str(e)}


# =============================================================================
# Convenience Functions
# =============================================================================

_builders: Dict[str, MemoryContextBuilder] = {}


def get_memory_context_builder(user_id: str = "default") -> MemoryContextBuilder:
    """Get or create a memory context builder for a user.

    Args:
        user_id: User identifier

    Returns:
        MemoryContextBuilder instance
    """
    if user_id not in _builders:
        _builders[user_id] = MemoryContextBuilder(user_id=user_id)
    return _builders[user_id]


def build_memory_context(
    current_input: str,
    user_id: str = "default",
    max_tokens: int = 500,
) -> str:
    """Convenience function to build memory context.

    This is the main entry point for memory context injection.

    Args:
        current_input: Current user input for relevance matching
        user_id: User ID
        max_tokens: Token budget for memory context

    Returns:
        Formatted memory context string (empty if no relevant memories)
    """
    builder = get_memory_context_builder(user_id=user_id)
    return builder.build_context(
        current_input=current_input,
        max_tokens=max_tokens,
    )


def build_memory_system_message(
    current_input: str,
    user_id: str = "default",
    max_tokens: int = 500,
) -> Optional[SystemMessage]:
    """Convenience function to build memory SystemMessage.

    This is the preferred way to inject memories into LangChain conversations.

    Args:
        current_input: Current user input for relevance matching
        user_id: User ID
        max_tokens: Token budget for memory context

    Returns:
        SystemMessage with memory context, or None if no relevant memories
    """
    builder = get_memory_context_builder(user_id=user_id)
    return builder.build_system_message(
        current_input=current_input,
        max_tokens=max_tokens,
    )
