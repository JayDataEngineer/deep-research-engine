"""
Storage backends for Context Engine.

Exports:
    - base: Base classes and mixins
    - postgres_store: Postgres + pgvector implementation
    - langgraph_store: LangGraph-compatible Postgres Store with auto-embeddings
    - s3_backend: MinIO/S3 implementation
    - neo4j_store: Neo4j graph implementation
    - minio_backend: DeepAgents BackendProtocol implementation for MinIO
    - unified_memory_router: CompositeBackend routing all 4 entry points
"""

from .base import (
    StorageBackend,
    StorageCapability,
    VectorStorageMixin,
    SessionScopedMixin,
    compute_content_hash
)
from .postgres_store import PostgresVectorStore
from .langgraph_store import LangGraphPostgresStore, get_langgraph_store
from .s3_backend import S3Backend
from .neo4j_store import Neo4jStore
from .minio_backend import (
    MinioBackend,
    FileInfo,
    WriteResult,
    EditResult,
    GrepMatch,
    make_minio_composite_backend,
    minio_backend_factory,
    get_minio_backend_for_session,
)
from .unified_memory_router import (
    PostgresBackend,
    Neo4jBackend,
    create_unified_memory_router,
    unified_memory_router_factory,
)

__all__ = [
    "StorageBackend",
    "StorageCapability",
    "VectorStorageMixin",
    "SessionScopedMixin",
    "compute_content_hash",
    "PostgresVectorStore",
    # LangGraph Store
    "LangGraphPostgresStore",
    "get_langgraph_store",
    # Other backends
    "S3Backend",
    "Neo4jStore",
    # DeepAgents BackendProtocol implementation
    "MinioBackend",
    "FileInfo",
    "WriteResult",
    "EditResult",
    "GrepMatch",
    "make_minio_composite_backend",
    "minio_backend_factory",
    "get_minio_backend_for_session",
    # Unified Memory Router (CompositeBackend for all 4 entry points)
    "PostgresBackend",
    "Neo4jBackend",
    "create_unified_memory_router",
    "unified_memory_router_factory",
]
