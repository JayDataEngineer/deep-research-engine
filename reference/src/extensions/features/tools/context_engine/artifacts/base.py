"""
Base artifact schema for Context Engine.

All artifacts inherit from BaseArtifact which includes session isolation fields.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum


class ArtifactType(str, Enum):
    """Types of artifacts in the system."""
    BASE = "base"
    MEMORY = "memory"
    FINDING = "finding"
    CLAIM = "claim"
    CITATION = "citation"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    DECISION = "decision"
    POSITION = "position"


class BaseArtifact(BaseModel):
    """Base artifact schema with session isolation.

    SESSION ISOLATION:
    - Research mode: Each research query gets a unique session_id
    - Companion mode: Multiple conversations share a user_id session
    - Artifacts are NEVER shared across unrelated sessions
    """
    id: str = Field(description="Unique artifact identifier")
    content: str = Field(description="Primary content/text")
    artifact_type: ArtifactType = Field(default=ArtifactType.BASE)

    # Session isolation (CRITICAL for research mode)
    session_id: str = Field(
        description="Session/research ID that scopes this artifact. "
                      "Artifacts from different sessions are NOT mixed during retrieval."
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for companion mode (shared across sessions). "
                     "Null for isolated research sessions."
    )

    # Metadata
    source_ids: List[str] = Field(
        default_factory=list,
        description="IDs of source documents/URLs"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this artifact was created (UTC timezone)"
    )

    # Scoring
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="0-1 importance score"
    )
    decay_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="How fast relevance decays (0=permanent, 1=fast decay)"
    )

    # Extended metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional artifact-specific metadata"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dictionary with all fields for database storage
        """
        return {
            "id": self.id,
            "content": self.content,
            "artifact_type": self.artifact_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "source_ids": self.source_ids,
            "created_at": self.created_at,
            "relevance_score": self.relevance_score,
            "decay_rate": self.decay_rate,
            "metadata": self.metadata
        }

    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> "BaseArtifact":
        """Create artifact from storage dictionary.

        Args:
            data: Dictionary from database storage

        Returns:
            BaseArtifact instance
        """
        return cls(
            id=data["id"],
            content=data["content"],
            artifact_type=ArtifactType(data["artifact_type"]),
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            source_ids=data.get("source_ids", []),
            created_at=data.get("created_at"),
            relevance_score=data.get("relevance_score", 1.0),
            decay_rate=data.get("decay_rate", 0.1),
            metadata=data.get("metadata", {})
        )
