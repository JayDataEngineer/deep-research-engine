"""
Context Engine Extension - Artifact-centric context management.

Provides a tiered storage architecture with:
- Postgres + pgvector: Semantic search and artifact persistence
- MinIO/S3: Raw data storage for large files
- Neo4j: Optional graph database for entity relationships
- Health check monitoring
- Session isolation (Research vs Companion modes)
- Deduplication for concurrent workers

Usage:
    from src.extensions.features.tools.context_engine import ContextEngineExtension

    extension = ContextEngineExtension(config={
        "enabled": True,
        "mode": "research",
        "postgres": {...},
        "minio": {...},
        "neo4j": {...}
    })

    # Get tools based on healthy backends
    tools = await extension.get_available_tools(state)

    # Add artifacts
    await extension.add_artifact(artifact)

    # Search
    results = await extension.similarity_search("query", session_id="...")
"""

from .extension import ContextEngineExtension
from .health_check import (
    HealthCheckRegistry,
    HealthCheckResult,
    HealthStatus,
    get_health_registry
)
from .storage import (
    StorageBackend,
    StorageCapability,
    PostgresVectorStore,
    S3Backend,
    Neo4jStore,
    compute_content_hash
)
from .artifacts import (
    BaseArtifact,
    ArtifactType,
    FindingArtifact,
    ClaimArtifact,
    CitationArtifact,
    EntityArtifact,
    RelationshipArtifact
)

__all__ = [
    "ContextEngineExtension",
    "HealthCheckRegistry",
    "HealthCheckResult",
    "HealthStatus",
    "get_health_registry",
    "StorageBackend",
    "StorageCapability",
    "PostgresVectorStore",
    "S3Backend",
    "Neo4jStore",
    "compute_content_hash",
    "BaseArtifact",
    "ArtifactType",
    "FindingArtifact",
    "ClaimArtifact",
    "CitationArtifact",
    "EntityArtifact",
    "RelationshipArtifact",
]
