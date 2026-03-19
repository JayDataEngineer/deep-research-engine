"""
Research artifact schemas.

Artifacts for research mode: FindingArtifact, ClaimArtifact, CitationArtifact
"""
from typing import List, Literal, Optional
from pydantic import Field
from datetime import datetime, timezone

from .base import BaseArtifact, ArtifactType


class FindingArtifact(BaseArtifact):
    """Research finding artifact.

    Represents a verified piece of information discovered during research.
    """
    artifact_type: ArtifactType = ArtifactType.FINDING

    # Research-specific fields
    topic_tags: List[str] = Field(
        default_factory=list,
        description="Topic labels for clustering"
    )
    verification_status: Literal["verified", "disputed", "unconfirmed"] = (
        "unconfirmed"
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this finding"
    )
    contradiction_ids: List[str] = Field(
        default_factory=list,
        description="IDs of artifacts that contradict this finding"
    )


class ClaimArtifact(BaseArtifact):
    """Atomic claim/fact extracted from research.

    Smaller than a finding - a single verifiable statement.
    """
    artifact_type: ArtifactType = ArtifactType.CLAIM

    # Claim-specific fields
    claim_type: Literal["fact", "opinion", "statistic", "quote"] = "fact"
    entity_mentions: List[str] = Field(
        default_factory=list,
        description="Entities mentioned in this claim"
    )
    temporal_scope: Optional[str] = Field(
        None,
        description="Time period this claim applies to"
    )
    support_count: int = Field(
        default=1,
        ge=1,
        description="How many sources support this claim"
    )


class CitationArtifact(BaseArtifact):
    """Citation for a source used in research."""
    artifact_type: ArtifactType = ArtifactType.CITATION

    url: str = ""
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    access_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    journal_ref: Optional[str] = None
    doi: Optional[str] = None

    # Citation-specific metadata
    tier: Optional[str] = Field(
        None,
        description="Source quality tier (A, B, C)"
    )
    snippet: Optional[str] = Field(
        None,
        description="Relevant snippet from source"
    )


__all__ = ["FindingArtifact", "ClaimArtifact", "CitationArtifact"]
