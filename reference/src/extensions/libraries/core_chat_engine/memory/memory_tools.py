"""Memory Tools for Companion AI - LangChain StructuredTools.

These tools allow the AI to:
- save_memory: Store important information in long-term memory
- search_memory: Search through stored memories
- reflect_on_conversation: Extract and store memories from conversation
"""

import json
import logging
from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from .memory_schema import MemoryType, MemoryImportance
from .memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


# =============================================================================
# Input Schemas
# =============================================================================


class SaveMemoryInput(BaseModel):
    """Input for saving a memory."""

    content: str = Field(
        ...,
        description="The memory content to store (what should be remembered)",
    )
    memory_type: str = Field(
        ...,
        description="Type of memory: 'fact' (information), 'preference' (likes/dislikes), "
        "'episode' (events), 'relationship' (people), 'goal' (aspirations), or 'event' (life events)",
    )
    importance: str = Field(
        default="medium",
        description="How important: 'low', 'medium', 'high', or 'critical'",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for later retrieval (e.g., ['coffee', 'morning'])",
    )


class SearchMemoryInput(BaseModel):
    """Input for searching memories."""

    query: str = Field(
        ...,
        description="Natural language search query (e.g., 'what does user like to drink')",
    )
    memory_types: Optional[List[str]] = Field(
        default=None,
        description="Optional filter by memory types: ['fact', 'preference', 'episode', etc.]",
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return",
    )


class ReflectInput(BaseModel):
    """Input for reflection and memory extraction."""

    conversation_summary: str = Field(
        ...,
        description="Summary of the recent conversation to extract memories from",
    )
    extract_count: int = Field(
        default=3,
        description="Number of important memories to extract",
    )


# =============================================================================
# Tool Functions
# =============================================================================


def save_memory(
    content: str,
    memory_type: str = "fact",
    importance: str = "medium",
    keywords: Optional[List[str]] = None,
    user_id: str = "default",
) -> str:
    """Save an important piece of information to long-term memory.

    Use this tool when the user shares something important that should be
    remembered for future conversations. Examples:
    - User preferences (likes/dislikes)
    - Important facts about the user
    - Significant events or experiences
    - Goals or aspirations
    - Important relationships

    Args:
        content: What to remember
        memory_type: Type of memory (fact, preference, episode, relationship, goal, event)
        importance: How important (low, medium, high, critical)
        keywords: Keywords for finding this memory later
        user_id: User to store memory for

    Returns:
        Confirmation message
    """
    try:
        manager = get_memory_manager(user_id=user_id)

        # Parse memory type
        try:
            mem_type = MemoryType(memory_type.lower())
        except ValueError:
            valid_types = [t.value for t in MemoryType]
            return f"Invalid memory type '{memory_type}'. Valid types: {', '.join(valid_types)}"

        # Parse importance
        try:
            imp = MemoryImportance[importance.upper()]
        except KeyError:
            imp = MemoryImportance.MEDIUM

        # Store the memory
        memory = manager.store_memory(
            memory_type=mem_type,
            content=content,
            importance=imp,
            keywords=keywords or [],
        )

        logger.info(f"[TOOL] Saved memory: {memory.memory_id} - {content[:50]}...")

        return (
            f"Saved to memory: {content[:100]}{'...' if len(content) > 100 else ''}\n"
            f"Type: {mem_type.value}, Importance: {imp.value}"
        )

    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return f"Failed to save memory: {str(e)}"


def search_memory(
    query: str,
    memory_types: Optional[List[str]] = None,
    limit: int = 10,
    user_id: str = "default",
) -> str:
    """Search through long-term memories.

    Use this tool to recall previously stored information about the user.
    This helps maintain context across conversations.

    Args:
        query: What to search for (natural language)
        memory_types: Optional filter by memory type
        limit: Max results to return
        user_id: User to search memories for

    Returns:
        Search results formatted as text
    """
    try:
        manager = get_memory_manager(user_id=user_id)

        # Parse memory types if provided
        types = None
        if memory_types:
            types = []
            for mt in memory_types:
                try:
                    types.append(MemoryType(mt.lower()))
                except ValueError:
                    continue

        # Search memories
        results = manager.search_memories(
            query=query,
            memory_types=types,
            limit=limit,
        )

        if not results:
            return f"No memories found matching: {query}"

        # Format results
        output_parts = [f"Found {len(results)} memories matching '{query}':\n"]

        for i, result in enumerate(results, 1):
            mem = result.memory
            relevance_pct = int(result.relevance_score * 100)

            output_parts.append(
                f"{i}. [{mem.memory_type.value}] {mem.content}\n"
                f"   Importance: {mem.importance.name} | Relevance: {relevance_pct}%"
            )

        logger.info(f"[TOOL] Searched memories: '{query}' -> {len(results)} results")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        return f"Failed to search memories: {str(e)}"


def reflect_on_conversation(
    conversation_summary: str,
    extract_count: int = 3,
    user_id: str = "default",
) -> str:
    """Reflect on a conversation and extract important memories.

    This tool analyzes a conversation summary and automatically extracts
    and stores important information as memories.

    Args:
        conversation_summary: Summary of the conversation
        extract_count: Number of memories to extract
        user_id: User to store memories for

    Returns:
        Summary of extracted memories
    """
    try:
        # Use LLM to extract memories
        # This is a placeholder - actual implementation would call the LLM
        # For now, return a simple response

        manager = get_memory_manager(user_id=user_id)

        # Simple heuristic extraction (replace with LLM-based extraction)
        extracted = []

        # Look for preference indicators
        if "likes" in conversation_summary.lower() or "loves" in conversation_summary.lower():
            extracted.append(
                manager.store_memory(
                    memory_type=MemoryType.PREFERENCE,
                    content="Extracted from: " + conversation_summary[:200],
                    importance=MemoryImportance.MEDIUM,
                )
            )

        if not extracted:
            return "No significant memories identified in this conversation."

        logger.info(f"[TOOL] Reflected and extracted {len(extracted)} memories")

        return f"Extracted {len(extracted)} memories from conversation:\n" + "\n".join(
            f"- {m.memory_type.value}: {m.content[:80]}..." for m in extracted
        )

    except Exception as e:
        logger.error(f"Failed to reflect on conversation: {e}")
        return f"Failed to extract memories: {str(e)}"


# =============================================================================
# LangChain Tool Definitions
# =============================================================================

# Create the tools
save_memory_tool = StructuredTool.from_function(
    func=save_memory,
    name="save_memory",
    description=(
        "Save important information to long-term memory. "
        "Use this when the user shares something significant that should be remembered "
        "for future conversations (preferences, facts, events, goals, relationships). "
        "This enables the AI to maintain context across sessions and build a deeper "
        "understanding of the user over time."
    ),
    args_schema=SaveMemoryInput,
)

search_memory_tool = StructuredTool.from_function(
    func=search_memory,
    name="search_memory",
    description=(
        "Search through stored long-term memories. "
        "Use this to recall previously learned information about the user "
        "to provide more personalized responses."
    ),
    args_schema=SearchMemoryInput,
)

reflect_tool = StructuredTool.from_function(
    func=reflect_on_conversation,
    name="reflect_on_conversation",
    description=(
        "Reflect on a conversation and automatically extract important memories. "
        "This analyzes the conversation and identifies significant information "
        "that should be stored for future reference."
    ),
    args_schema=ReflectInput,
)


# =============================================================================
# Tool Export
# =============================================================================

MEMORY_TOOLS = [
    save_memory_tool,
    search_memory_tool,
    # reflect_tool  # Optional - can be enabled for auto-reflection
]


def get_memory_tools() -> List[StructuredTool]:
    """Get all memory tools.

    Returns:
        List of LangChain StructuredTool instances
    """
    return MEMORY_TOOLS
