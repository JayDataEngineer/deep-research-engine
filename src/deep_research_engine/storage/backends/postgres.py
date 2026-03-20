"""PostgreSQL+pgvector storage backend."""

import uuid
from typing import Any

from ..base import StorageBackend, StorageResult
from ..errors import StorageInitializationError
from ..registry import StorageRegistry


@StorageRegistry.register("postgres")
class PostgresStorage(StorageBackend):
    """PostgreSQL storage backend with pgvector support.

    Features:
    - Vector similarity search via pgvector
    - Structured data storage
    - Automatic table creation
    - Connection pooling via asyncpg

    Configuration:
    ```yaml
    storage:
      postgres:
        url: "postgresql://user:pass@host:port/dbname"
        pool_size: 10
        embedding_dimension: 768
    ```
    """

    backend_name = "postgres"
    description = "PostgreSQL with pgvector for vectors and structured data"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", "postgresql://postgres:postgres@localhost:5432/shared_db")
        self.pool_size = config.get("pool_size", 10)
        self.embedding_dimension = config.get("embedding_dimension", 768)
        self.pool = None
        self._lock = None

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool and create tables."""
        try:
            import asyncpg
            from pgvector.asyncpg import register_vector
        except ImportError as e:
            raise StorageInitializationError(
                f"Missing dependencies for PostgreSQL: {e}. "
                "Install with: pip install asyncpg pgvector"
            )

        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.url,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=60,
            )

            # Register pgvector types
            async with self.pool.acquire() as conn:
                await register_vector(conn)

            # Create tables
            await self._create_tables()

            self.initialized = True
            self.health_status = "healthy"
            self._log(f"Initialized PostgreSQL storage at {self.url}")

        except Exception as e:
            self.health_status = "unhealthy"
            raise StorageInitializationError(f"Failed to initialize PostgreSQL: {e}")

    async def cleanup(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.initialized = False
            self._log("PostgreSQL connection pool closed")

    async def health_check(self) -> bool:
        """Check PostgreSQL connection health."""
        if not self.pool or not self.initialized:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            self.health_status = "unhealthy"
            return False

    async def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create main storage table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS storage_items (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    data JSONB NOT NULL,
                    embedding vector({self.embedding_dimension}),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_storage_namespace
                ON storage_items(namespace)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_storage_embedding
                ON storage_items USING ivfflat (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_storage_data
                ON storage_items USING GIN (data)
            """)

            self._log("Tables and indexes created successfully")

    async def store(
        self,
        data: dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        """Store data in PostgreSQL.

        Supports optional embedding for vector search.
        If data contains 'embedding' key, it will be stored for similarity search.
        """
        if not self.initialized:
            return StorageResult(
                success=False,
                error="PostgreSQL not initialized",
            )

        try:
            resource_id = data.get("id", str(uuid.uuid4()))
            embedding = data.pop("embedding", None)
            metadata = data.get("metadata", {})

            async with self.pool.acquire() as conn:
                # Upsert logic
                await conn.execute(
                    """
                    INSERT INTO storage_items (id, namespace, data, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET
                        data = EXCLUDED.data,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """,
                    resource_id,
                    namespace,
                    data,
                    embedding,
                    metadata,
                )

            self._log(f"Stored item {resource_id} in namespace {namespace}")

            return StorageResult(
                success=True,
                id=resource_id,
                metadata={"namespace": namespace, "has_embedding": embedding is not None},
            )

        except Exception as e:
            self._log(f"Store failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve data by ID."""
        if not self.initialized:
            return None

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT data, embedding, metadata, created_at
                    FROM storage_items
                    WHERE id = $1 AND namespace = $2
                """,
                    id,
                    namespace,
                )

                if not row:
                    return None

                result = dict(row["data"])
                if row["embedding"]:
                    result["embedding"] = list(row["embedding"])
                if row["metadata"]:
                    result["metadata"] = dict(row["metadata"])
                result["created_at"] = row["created_at"].isoformat()

                return result

        except Exception as e:
            self._log(f"Retrieve failed: {e}", level="error")
            return None

    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        """Delete data by ID."""
        if not self.initialized:
            return False

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM storage_items
                    WHERE id = $1 AND namespace = $2
                """,
                    id,
                    namespace,
                )

                deleted = result == "DELETE 1"
                if deleted:
                    self._log(f"Deleted item {id} from namespace {namespace}")
                return deleted

        except Exception as e:
            self._log(f"Delete failed: {e}", level="error")
            return False

    async def query(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Query data with optional vector similarity search.

        Args:
            query: Can be either:
                - A JSON query string for metadata filtering
                - A comma-separated vector for similarity search (if embedding_query=True)
            namespace: Namespace to query
            limit: Maximum results to return
            **kwargs: Additional options:
                - embedding_query: If True, treat query as vector for similarity search
                - similarity_threshold: Minimum similarity score (0-1)
        """
        if not self.initialized:
            return []

        embedding_query = kwargs.get("embedding_query", False)
        similarity_threshold = kwargs.get("similarity_threshold", 0.0)

        try:
            async with self.pool.acquire() as conn:
                if embedding_query:
                    # Vector similarity search
                    # Parse query as vector
                    try:
                        if isinstance(query, str):
                            query_vector = [float(x) for x in query.split(",")]
                        else:
                            query_vector = list(query)
                    except (ValueError, TypeError):
                        self._log(
                            "Invalid query vector, falling back to text query", level="warning"
                        )
                        return await self._text_query(conn, query, namespace, limit)

                    rows = await conn.fetch(
                        """
                        SELECT id, data, embedding, metadata, created_at,
                               1 - (embedding <=> $3::vector) as similarity
                        FROM storage_items
                        WHERE namespace = $1
                          AND embedding IS NOT NULL
                          AND 1 - (embedding <=> $3::vector) >= $4
                        ORDER BY embedding <=> $3::vector
                        LIMIT $2
                    """,
                        namespace,
                        limit,
                        query_vector,
                        similarity_threshold,
                    )

                    return [self._row_to_dict(row, include_similarity=True) for row in rows]

                else:
                    # Text/metadata query
                    return await self._text_query(conn, query, namespace, limit)

        except Exception as e:
            self._log(f"Query failed: {e}", level="error")
            return []

    async def _text_query(
        self,
        conn,
        query: str,
        namespace: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Perform text/metadata query."""
        # Simple text search in JSONB data
        rows = await conn.fetch(
            """
            SELECT id, data, embedding, metadata, created_at
            FROM storage_items
            WHERE namespace = $1
              AND (data::text LIKE $2 OR metadata::text LIKE $2)
            ORDER BY created_at DESC
            LIMIT $3
        """,
            namespace,
            f"%{query}%",
            limit,
        )

        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row, include_similarity: bool = False) -> dict[str, Any]:
        """Convert database row to dictionary."""
        result = dict(row["data"])
        if row["embedding"]:
            result["embedding"] = list(row["embedding"])
        if row["metadata"]:
            result["metadata"] = dict(row["metadata"])
        result["id"] = row["id"]
        result["created_at"] = row["created_at"].isoformat()
        if include_similarity and "similarity" in row:
            result["similarity"] = float(row["similarity"])
        return result

    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List items in namespace."""
        if not self.initialized:
            return []

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, namespace, created_at, metadata
                    FROM storage_items
                    WHERE namespace = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """,
                    namespace,
                    limit,
                )

                return [
                    {
                        "id": row["id"],
                        "namespace": row["namespace"],
                        "created_at": row["created_at"].isoformat(),
                        "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    }
                    for row in rows
                ]

        except Exception as e:
            self._log(f"List failed: {e}", level="error")
            return []

    async def store_batch(
        self,
        items: list[dict[str, Any]],
        namespace: str = "default",
    ) -> list[StorageResult]:
        """Store multiple items in a batch transaction."""
        if not self.initialized:
            return [StorageResult(success=False, error="PostgreSQL not initialized") for _ in items]

        results = []
        try:
            query_args = []
            for item in items:
                resource_id = item.get("id", str(uuid.uuid4()))
                embedding = item.pop("embedding", None)
                metadata = item.get("metadata", {})

                query_args.append((
                    resource_id,
                    namespace,
                    item,
                    embedding,
                    metadata,
                ))

                results.append(
                    StorageResult(
                        success=True,
                        id=resource_id,
                        metadata={"namespace": namespace},
                    )
                )

            async with self.pool.acquire() as conn, conn.transaction():
                if query_args:
                    await conn.executemany(
                        """
                            INSERT INTO storage_items (id, namespace, data, embedding, metadata)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (id) DO UPDATE SET
                                data = EXCLUDED.data,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                updated_at = NOW()
                        """,
                        query_args
                    )

            self._log(f"Batch stored {len(items)} items in namespace {namespace}")
            return results

        except Exception as e:
            self._log(f"Batch store failed: {e}", level="error")
            return [StorageResult(success=False, error=str(e)) for _ in items]
