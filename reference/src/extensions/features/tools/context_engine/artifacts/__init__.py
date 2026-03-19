"""
Artifact schemas for Context Engine.

Exports:
    - BaseArtifact: Base artifact class with session isolation
    - FindingArtifact: Research finding artifact
    - ClaimArtifact: Atomic claim/fact artifact
    - CitationArtifact: Source citation artifact
    - EntityArtifact: OSINT entity artifact
    - RelationshipArtifact: OSINT relationship artifact
"""

from .base import BaseArtifact, ArtifactType
from .research import FindingArtifact, ClaimArtifact, CitationArtifact
from .osint import EntityArtifact, RelationshipArtifact

__all__ = [
    "BaseArtifact",
    "ArtifactType",
    "FindingArtifact",
    "ClaimArtifact",
    "CitationArtifact",
    "EntityArtifact",
    "RelationshipArtifact",
]
