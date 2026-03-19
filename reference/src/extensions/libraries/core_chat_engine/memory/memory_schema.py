"""Long-term Memory Schema for Companion AI.

Defines the data models for storing and retrieving memories across sessions.
Memories are organized into types: episodes, facts, preferences, relationships.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories the companion can store."""

    EPISODE = "episode"  # Conversation events/moments to remember
    FACT = "fact"  # Learned information about the user
    PREFERENCE = "preference"  # User likes/dislikes
    RELATIONSHIP = "relationship"  # Important people in user's life
    GOAL = "goal"  # User's goals and aspirations
    EVENT = "event"  # Important life events


class MemoryImportance(int, Enum):
    """Importance level of a memory (affects retrieval priority)."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Memory(BaseModel):
    """A single memory stored in long-term memory.

    Stored as JSON in the BaseStore key-value system.
    """

    # Core fields
    memory_id: str = Field(..., description="Unique identifier for this memory")
    memory_type: MemoryType = Field(..., description="Type of memory")
    content: str = Field(..., description="Main content/text of the memory")

    # Metadata
    importance: MemoryImportance = Field(
        default=MemoryImportance.MEDIUM, description="How important this memory is"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="When this memory was created",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="When this memory was last updated",
    )

    # Semantic indexing
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding for semantic search"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Keywords for matching"
    )

    # User/Session context
    user_id: str = Field(default="default", description="User this memory belongs to")
    thread_id: Optional[str] = Field(
        default=None, description="Thread where this memory was created"
    )

    # Related memories
    related_memory_ids: List[str] = Field(
        default_factory=list, description="IDs of related memories"
    )

    # Access pattern
    access_count: int = Field(default=0, description="How often this memory is accessed")
    last_accessed_at: Optional[str] = Field(
        default=None, description="When this memory was last accessed"
    )

    # Additional data (type-specific)
    extra_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional data for specific memory types"
    )


class EpisodeMemory(Memory):
    """A memory of a specific conversation or event.

    Example: "User was really happy when they got promoted last week"
    """

    memory_type: MemoryType = MemoryType.EPISODE

    # Episode-specific fields
    timestamp: Optional[str] = Field(
        default=None, description="When the episode occurred (if different from created_at)"
    )
    participants: List[str] = Field(
        default_factory=list, description="Who was involved in this episode"
    )
    emotion: Optional[str] = Field(
        default=None, description="Emotion associated with this episode"
    )


class FactMemory(Memory):
    """A factual memory about the user.

    Example: "User is a software engineer who loves Python"
    """

    memory_type: MemoryType = MemoryType.FACT

    # Fact-specific fields
    category: Optional[str] = Field(
        default=None, description="Category of fact (e.g., 'work', 'hobby')"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this fact"
    )
    verified: bool = Field(default=False, description="Whether this fact was confirmed")


class PreferenceMemory(Memory):
    """A memory of user preferences.

    Example: "User prefers coffee over tea, especially lattes"
    """

    memory_type: MemoryType = MemoryType.PREFERENCE

    # Preference-specific fields
    sentiment: str = Field(
        default="neutral",
        description="User's sentiment: 'positive', 'negative', or 'neutral'",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How strong this preference is (0-1)",
    )


class RelationshipMemory(Memory):
    """A memory about someone important in the user's life.

    Example: "User's spouse is named Alex, they've been married 5 years"
    """

    memory_type: MemoryType = MemoryType.RELATIONSHIP

    # Relationship-specific fields
    relationship_type: str = Field(
        ..., description="Type of relationship (e.g., 'spouse', 'friend', 'parent')"
    )
    person_name: str = Field(..., description="Name of the person")
    connection_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How close this relationship is (0-1)",
    )


class GoalMemory(Memory):
    """A memory about user's goals or aspirations.

    Example: "User wants to learn Spanish before their trip to Mexico"
    """

    memory_type: MemoryType = MemoryType.GOAL

    # Goal-specific fields
    target_date: Optional[str] = Field(
        default=None, description="Target date for achieving this goal"
    )
    progress: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Progress toward this goal (0-1)"
    )
    status: str = Field(
        default="active",
        description="Goal status: 'active', 'completed', 'paused', 'abandoned'",
    )


class EventMemory(Memory):
    """A memory of an important life event.

    Example: "User got married on June 15, 2023"
    """

    memory_type: MemoryType = MemoryType.EVENT

    # Event-specific fields
    event_date: Optional[str] = Field(
        default=None, description="When the event occurred"
    )
    event_type: str = Field(
        ..., description="Type of event (e.g., 'wedding', 'graduation', 'move')"
    )
    impact_level: str = Field(
        default="medium",
        description="Life impact: 'low', 'medium', 'high'",
    )


class MemorySearchQuery(BaseModel):
    """Query for searching memories."""

    query_text: str = Field(..., description="Natural language search query")
    memory_types: Optional[List[MemoryType]] = Field(
        default=None, description="Filter by memory types"
    )
    min_importance: Optional[MemoryImportance] = Field(
        default=None, description="Filter by minimum importance"
    )
    user_id: str = Field(default="default", description="User to search memories for")
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")
    keywords: Optional[List[str]] = Field(
        default=None, description="Filter by keywords"
    )


class MemorySearchResult(BaseModel):
    """Result from a memory search."""

    memory: Memory
    relevance_score: float = Field(..., description="Semantic similarity score (0-1)")
    match_reason: str = Field(..., description="Why this memory matched")


class MemorySummary(BaseModel):
    """Summary of a user's memory state."""

    total_memories: int
    memories_by_type: Dict[str, int]
    most_recent_memories: List[Memory]
    important_memories: List[Memory]
    memory_density: Dict[str, int]  # Keywords with frequency
