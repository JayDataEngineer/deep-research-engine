"""
Postgres + pgvector storage backend for Context Engine.

Provides semantic search and artifact persistence with deduplication.
"""
import asyncpg
from pgvector.asyncpg import register_vector
from typing import Optional, List, Tuple
import logging
import json

from langchain_core.embeddings import Embeddings

from .base import (
    StorageBackend,
    StorageCapability,
    VectorStorageMixin,
    SessionScopedMixin,
    compute_content_hash
)
from ..artifacts.base import BaseArtifact, ArtifactType
from ..health_check import HealthCheckResult, HealthStatus

logger = logging.getLogger(__name__)


class PostgresVectorStore(
    StorageBackend,
    VectorStorageMixin,
    SessionScopedMixin
):
    """Postgres + pgvector for semantic search with deduplication.

    PERFORMANCE: Uses connection pooling to minimize connection overhead.
    DEDUPLICATION: Uses content_hash and unique constraints to prevent
    duplicate artifacts from concurrent workers.
    """

    def __init__(
        self,
        name: str = "postgres",
        connection_string: str = "postgresql://ulli:ulli_password@localhost:5432/ulli_db",
        embedding_model: Embeddings = None,
        artifact_table: str = "artifacts",
        embedding_table: str = "artifact_embeddings",
        pool_min_size: int = 5,
        pool_max_size: int = 20,
        config: dict = None
    ):
        super().__init__(name, config or {})

        self.connection_string = connection_string
        self.embedding_model = embedding_model
        self.artifact_table = artifact_table
        self.embedding_table = embedding_table
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool = None

    @property
    def capabilities(self) -> set[StorageCapability]:
        """Postgres supports add, get, search, vector search."""
        return {
            StorageCapability.ADD,
            StorageCapability.GET,
            StorageCapability.SEARCH,
            StorageCapability.VECTOR_SEARCH
        }

    async def _add_internal(
        self,
        artifact: BaseArtifact
    ) -> Tuple[str, bool]:
        """Add artifact with embedding to Postgres with deduplication.

        Returns:
            Tuple of (artifact_id, is_new)
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await register_vector(conn)

            # Compute content hash for deduplication
            content_hash = compute_content_hash(artifact.content)

            # Check if artifact with same content already exists in this session
            existing = await conn.fetchrow(f"""
                SELECT id FROM {self.artifact_table}
                WHERE session_id = $1 AND content_hash = $2
            """, artifact.session_id, content_hash)

            if existing:
                logger.info(f"Duplicate artifact detected by content_hash: {existing['id']}")
                return (existing['id'], False)

            # Check for duplicate by source (for artifacts with source_ids)
            if artifact.source_ids:
                source_array = list(artifact.source_ids)
                existing = await conn.fetchrow(f"""
                    SELECT id FROM {self.artifact_table}
                    WHERE session_id = $1
                      AND artifact_type = $2
                      AND source_ids = $3
                """, artifact.session_id, artifact.artifact_type.value, source_array)

                if existing:
                    logger.info(f"Duplicate artifact detected by source_ids: {existing['id']}")
                    return (existing['id'], False)

            # For entities, check by entity name
            if artifact.artifact_type == ArtifactType.ENTITY:
                entity_name = artifact.metadata.get("entity_name")
                if entity_name:
                    existing = await conn.fetchrow(f"""
                        SELECT id FROM {self.artifact_table}
                        WHERE session_id = $1
                          AND artifact_type = 'entity'
                          AND metadata->>'entity_name' = $2
                    """, artifact.session_id, entity_name)

                    if existing:
                        logger.info(f"Duplicate entity detected by name: {existing['id']}")
                        return (existing['id'], False)

            # No duplicate found - insert new artifact
            # Generate embedding
            embedding = await self.embedding_model.aembed_query(artifact.content)

            # Convert metadata to plain dict for JSONB storage
            # Use model_dump() for Pydantic v2 compatibility
            if hasattr(artifact, 'model_dump'):
                # Pydantic v2
                artifact_dict = artifact.model_dump()
            else:
                # Pydantic v1
                artifact_dict = artifact.dict()

            metadata_dict = artifact_dict.get('metadata', {})

            # Use transaction for atomicity
            async with conn.transaction():
                # Insert artifact with content_hash
                # NOTE: Must use json.dumps() for asyncpg - it doesn't auto-encode dict to JSONB
                await conn.execute(f"""
                    INSERT INTO {self.artifact_table}
                    (id, content, artifact_type, session_id, user_id,
                     source_ids, content_hash, relevance_score, decay_rate, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        relevance_score = EXCLUDED.relevance_score
                """,
                    artifact.id, artifact.content, artifact.artifact_type.value,
                    artifact.session_id, artifact.user_id,
                    list(artifact.source_ids), content_hash,
                    artifact.relevance_score, artifact.decay_rate, json.dumps(metadata_dict)
                )

                # Insert embedding
                await conn.execute(f"""
                    INSERT INTO {self.embedding_table} (artifact_id, embedding)
                    VALUES ($1, $2)
                    ON CONFLICT (artifact_id) DO UPDATE SET embedding = $2
                """, artifact.id, embedding)

            return (artifact.id, True)

    async def _get_internal(
        self,
        artifact_id: str
    ) -> Optional[BaseArtifact]:
        """Get artifact by ID."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT * FROM {self.artifact_table} WHERE id = $1
            """, artifact_id)

            if not row:
                return None

            return BaseArtifact(
                id=row["id"],
                content=row["content"],
                artifact_type=ArtifactType(row["artifact_type"]),
                session_id=row["session_id"],
                user_id=row["user_id"],
                source_ids=list(row["source_ids"]) if row["source_ids"] else [],
                created_at=row["created_at"],
                relevance_score=row["relevance_score"],
                decay_rate=row["decay_rate"],
                metadata=dict(row["metadata"])
            )

    async def _search_internal(
        self,
        query: str,
        k: int,
        session_id: Optional[str],
        artifact_type: Optional[str],
        **filters
    ) -> List[BaseArtifact]:
        """Search by semantic similarity with SQL filtering.

        CRITICAL: session_id is REQUIRED for research mode to enforce
        session isolation and prevent cross-contamination.
        """
        # Enforce session_id for research mode
        mode = filters.get("mode", "research")
        if mode == "research" and session_id is None:
            raise ValueError(
                "session_id is REQUIRED for research mode to prevent "
                "cross-session contamination. Use mode='companion' for "
                "user-scoped searches without session_id."
            )

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await register_vector(conn)

            # Generate query embedding
            query_embedding = await self.embedding_model.aembed_query(query)

            # Build SQL query with filters
            where_conditions = ["1=1"]
            params = []
            param_count = 1

            if session_id:
                where_conditions.append(f"a.session_id = ${param_count}")
                params.append(session_id)
                param_count += 1

            if artifact_type:
                where_conditions.append(f"a.artifact_type = ${param_count}")
                params.append(artifact_type)
                param_count += 1

            where_clause = " AND ".join(where_conditions)

            # Execute similarity search
            rows = await conn.fetch(f"""
                SELECT a.id, a.content, a.artifact_type, a.session_id,
                       a.relevance_score, a.metadata, a.source_ids
                FROM {self.artifact_table} a
                JOIN {self.embedding_table} e ON a.id = e.artifact_id
                WHERE {where_clause}
                ORDER BY e.embedding <=> $1
                LIMIT ${param_count}
            """, query_embedding, *params, k)

            # Convert to artifacts
            artifacts = []
            for row in rows:
                artifacts.append(BaseArtifact(
                    id=row["id"],
                    content=row["content"],
                    artifact_type=ArtifactType(row["artifact_type"]),
                    session_id=row["session_id"],
                    relevance_score=row["relevance_score"],
                    metadata=dict(row["metadata"]),
                    source_ids=list(row["source_ids"]) if row["source_ids"] else []
                ))

            return artifacts

    async def vector_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        session_id: Optional[str] = None,
        artifact_type: Optional[str] = None
    ) -> List[Tuple[BaseArtifact, float]]:
        """Search by pre-computed embedding vector.

        Returns:
            List of (artifact, similarity_score) tuples
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await register_vector(conn)

            where_conditions = ["1=1"]
            params = [query_embedding]
            param_count = 2

            if session_id:
                where_conditions.append(f"a.session_id = ${param_count}")
                params.append(session_id)
                param_count += 1

            if artifact_type:
                where_conditions.append(f"a.artifact_type = ${param_count}")
                params.append(artifact_type)
                param_count += 1

            where_clause = " AND ".join(where_conditions)

            rows = await conn.fetch(f"""
                SELECT a.*, e.embedding <=> $1 as distance
                FROM {self.artifact_table} a
                JOIN {self.embedding_table} e ON a.id = e.artifact_id
                WHERE {where_clause}
                ORDER BY e.embedding <=> $1
                LIMIT ${param_count}
            """, *params, k)

            return [
                (
                    BaseArtifact(
                        id=row["id"],
                        content=row["content"],
                        artifact_type=ArtifactType(row["artifact_type"]),
                        session_id=row["session_id"],
                        relevance_score=row["relevance_score"],
                        metadata=dict(row["metadata"]),
                        source_ids=list(row["source_ids"]) if row["source_ids"] else []
                    ),
                    1 - row["distance"]  # Convert distance to similarity
                )
                for row in rows
            ]

    async def health_check(self) -> HealthCheckResult:
        """Postgres-specific health check."""
        import time
        start = time.time()

        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                # Test query
                version = await conn.fetchval("SELECT version()")

            latency_ms = (time.time() - start) * 1000

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"Postgres connected, latency: {latency_ms:.1f}ms",
                latency_ms=latency_ms,
                metadata={"version": version[:50] if version else None}
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Postgres health check failed: {str(e)}"
            )

    async def _get_pool(self):
        """Get or create connection pool.

        PERFORMANCE: Connection pool reused across queries to minimize
        connection overhead. Pool is created on first use.
        """
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=60
            )
        return self._pool

    async def close(self):
        """Close the connection pool.

        Call this when shutting down the extension to release connections.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
