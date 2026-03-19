"""
OSINT artifact schemas.

Artifacts for OSINT investigations: EntityArtifact, RelationshipArtifact
"""
from typing import List, Dict, Literal
from pydantic import Field

from .base import BaseArtifact, ArtifactType


class EntityArtifact(BaseArtifact):
    """OSINT entity artifact.

    Represents a person, organization, or location discovered during investigation.
    """
    artifact_type: ArtifactType = ArtifactType.ENTITY

    # Entity-specific fields
    entity_name: str = Field(..., description="Name of the entity")
    entity_type: Literal["person", "organization", "location", "concept"] = (
        "concept"
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names for this entity"
    )
    attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted attributes (role, title, etc.)"
    )
    connections: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Known relationships to other entities"
    )


class RelationshipArtifact(BaseArtifact):
    """OSINT relationship artifact.

    Represents a connection between two entities.
    """
    artifact_type: ArtifactType = ArtifactType.RELATIONSHIP

    # Relationship-specific fields
    from_entity: str = Field(..., description="Source entity name")
    to_entity: str = Field(..., description="Target entity name")
    relationship_type: Literal[
        "works_for",
        "owns",
        "invested_in",
        "board_member_of",
        "founded",
        "acquired",
        "partner_of",
        "family_of",
        "knows",
        "enemy_of",
        "located_at",
        "connected_to"
    ] = "connected_to"
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Sources that confirm this relationship"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship"
    )


__all__ = ["EntityArtifact", "RelationshipArtifact"]
