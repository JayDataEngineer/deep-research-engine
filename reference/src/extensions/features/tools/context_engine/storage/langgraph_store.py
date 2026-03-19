"""
LangGraph-compatible Postgres Store for Context Engine.

Implements LangGraph's BaseStore interface using Postgres + pgvector
for persistent, searchable storage.

This replaces the manual embedding generation in PostgresVectorStore
with LangGraph's Store pattern (auto-embeddings via index config).
"""

import os
import asyncpg
from pgvector.asyncpg import register_vector
from typing import Optional, List, Dict, Any, Tuple
import logging
import json
import hashlib
from datetime import datetime, timezone

from langgraph.store.base import BaseStore, Item, SearchItem
from langchain_core.embeddings import Embeddings

from src.shared.logging_config import get_logger

logger = get_logger(__name__)


class LangGraphPostgresStore(BaseStore):
    """Postgres + pgvector implementation of LangGraph's BaseStore.

    Provides persistent storage with automatic embedding generation
    for semantic search. Compatible with LangGraph's Store ecosystem.

    Usage:
        store = LangGraphPostgresStore(
            connection_string="postgresql://...",
            embeddings=my_embeddings,
            index_fields=["content"],  # Auto-embed these fields
        )

        # Put with auto-embedding
        await store.aput(
            namespace=("session_123", "memories"),
            key="doc1",
            value={"content": "Important research findings..."}
        )

        # Semantic search
        results = await store.asearch(
            ("session_123", "memories"),
            query="research about AI"
        )
    """

    def __init__(
        self,
        connection_string: str,
        embeddings: Embeddings,
        index_fields: List[str] = None,
        table_name: str = "langgraph_store",
        pool_min_size: int = 5,
        pool_max_size: int = 20,
    ):
        """Initialize the LangGraph Postgres Store.

        Args:
            connection_string: Postgres connection string
            embeddings: LangChain Embeddings instance for auto-vectorization
            index_fields: Fields to auto-embed for semantic search (default: ["content"])
            table_name: Table name for storage (default: "langgraph_store")
            pool_min_size: Connection pool min size
            pool_max_size: Connection pool max size
        """
        self.connection_string = connection_string
        self.embeddings = embeddings
        self.index_fields = index_fields or ["content"]
        self.table_name = table_name
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool = None

        # Embedding dimensions (will be detected on first use)
        self._embedding_dims = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=60
            )
            # Ensure table exists
            await self._ensure_table()
        return self._pool

    async def _ensure_table(self):
        """Create table if not exists."""
        pool = self._pool
        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (namespace, key)
                )
            """)

            # Create index on embedding for fast similarity search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Create index on namespace for prefix queries
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_namespace_idx
                ON {self.table_name} (namespace)
            """)

    def _namespace_to_str(self, namespace: Tuple[str, ...]) -> str:
        """Convert namespace tuple to string for storage."""
        return "/".join(namespace)

    def _str_to_namespace(self, namespace_str: str) -> Tuple[str, ...]:
        """Convert string back to namespace tuple."""
        return tuple(namespace_str.split("/"))

    async def _get_embedding(self, value: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for indexed fields in value.

        Concatenates all indexed field values for embedding.
        """
        texts_to_embed = []
        for field in self.index_fields:
            if field in value:
                field_value = value[field]
                if isinstance(field_value, str):
                    texts_to_embed.append(field_value)
                elif isinstance(field_value, list):
                    texts_to_embed.append(" ".join(str(v) for v in field_value))
                else:
                    texts_to_embed.append(str(field_value))

        if not texts_to_embed:
            return None

        combined_text = " ".join(texts_to_embed)
        embedding = await self.embeddings.aembed_query(combined_text)

        # Detect dimensions on first use
        if self._embedding_dims is None:
            self._embedding_dims = len(embedding)
            logger.info(f"[LangGraphStore] Detected embedding dimensions: {self._embedding_dims}")

        return embedding

    def _compute_content_hash(self, value: Dict[str, Any], content_fields: List[str]) -> str:
        """Compute hash of content fields for deduplication.

        Normalizes text (lowercase, strip whitespace) before hashing.
        """
        texts = []
        for field in content_fields:
            if field in value:
                field_value = value[field]
                if isinstance(field_value, str):
                    texts.append(field_value)
                elif isinstance(field_value, list):
                    texts.append(" ".join(str(v) for v in field_value))
                else:
                    texts.append(str(field_value))

        combined = " ".join(texts)
        # Normalize: lowercase, collapse whitespace
        normalized = " ".join(combined.split()).lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _is_duplicate(
        self,
        namespace: Tuple[str, ...],
        value: Dict[str, Any],
        content_fields: List[str],
    ) -> bool:
        """Check if content already exists in namespace.

        Uses exact hash matching on normalized content.

        Args:
            namespace: Namespace to search
            value: Value dict to check
            content_fields: Fields to use for content hashing

        Returns:
            True if duplicate found, False otherwise
        """
        content_hash = self._compute_content_hash(value, content_fields)
        namespace_str = self._namespace_to_str(namespace)
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Get all items in namespace
            rows = await conn.fetch(f"""
                SELECT value FROM {self.table_name}
                WHERE namespace = $1
            """, namespace_str)

            for row in rows:
                existing_value = row["value"]  # Already a dict from asyncpg JSONB
                if isinstance(existing_value, str):
                    existing_value = json.loads(existing_value)
                existing_hash = self._compute_content_hash(existing_value, content_fields)
                if existing_hash == content_hash:
                    return True

        return False

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index = None,
        *,
        ttl = None,
        check_duplicate: bool = False,
    ) -> bool:
        """Store a value with auto-embedding.

        Args:
            namespace: Namespace tuple (e.g., ("session_123", "memories"))
            key: Unique key within namespace
            value: Dict value to store
            index: Override index fields (None = use self.index_fields)
            ttl: Time-to-live (not implemented yet)
            check_duplicate: If True, check for duplicate content before storing

        Returns:
            True if stored, False if duplicate and check_duplicate=True
        """
        pool = await self._get_pool()

        # Optional duplicate check
        if check_duplicate:
            content_fields = index if isinstance(index, list) else self.index_fields
            if await self._is_duplicate(namespace, value, content_fields):
                logger.debug(f"[LangGraphStore] Skipping duplicate content in {namespace}")
                return False

        # Determine which fields to index
        fields_to_index = index if index is not None else self.index_fields

        # Generate embedding for indexed fields
        embedding = None
        temp_fields = fields_to_index if isinstance(fields_to_index, list) else self.index_fields

        # Temporarily override index_fields if custom index provided
        original_fields = self.index_fields
        if isinstance(index, list):
            self.index_fields = index

        try:
            embedding = await self._get_embedding(value)
        finally:
            self.index_fields = original_fields

        namespace_str = self._namespace_to_str(namespace)

        async with pool.acquire() as conn:
            await register_vector(conn)

            # Upsert the value
            await conn.execute(f"""
                INSERT INTO {self.table_name} (namespace, key, value, embedding, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (namespace, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
            """, namespace_str, key, json.dumps(value), embedding)

        logger.debug(f"[LangGraphStore] Put {namespace_str}/{key}")
        return True

    async def aget(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl = None,
    ) -> Optional[Item]:
        """Get a value by namespace and key.

        Args:
            namespace: Namespace tuple
            key: Unique key
            refresh_ttl: Refresh TTL (not implemented)

        Returns:
            Item if found, None otherwise
        """
        pool = await self._get_pool()
        namespace_str = self._namespace_to_str(namespace)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT key, value, created_at, updated_at
                FROM {self.table_name}
                WHERE namespace = $1 AND key = $2
            """, namespace_str, key)

            if not row:
                return None

            return Item(
                value=dict(row["value"]),
                key=row["key"],
                namespace=namespace,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl = None,
    ) -> List[SearchItem]:
        """Search within namespace prefix.

        Args:
            namespace_prefix: Namespace prefix to search within
            query: Natural language query for semantic search
            filter: Metadata filters (not yet implemented)
            limit: Max results
            offset: Offset for pagination
            refresh_ttl: Refresh TTL (not implemented)

        Returns:
            List of SearchItem with relevance scores
        """
        pool = await self._get_pool()
        namespace_prefix_str = self._namespace_to_str(namespace_prefix)

        async with pool.acquire() as conn:
            await register_vector(conn)

            if query:
                # Semantic search with query embedding
                query_embedding = await self.embeddings.aembed_query(query)

                rows = await conn.fetch(f"""
                    SELECT key, value, namespace, created_at, updated_at,
                           1 - (embedding <=> $1) as score
                    FROM {self.table_name}
                    WHERE namespace LIKE $2 || '%'
                    ORDER BY embedding <=> $1
                    LIMIT $3 OFFSET $4
                """, query_embedding, namespace_prefix_str, limit, offset)
            else:
                # No query - just list items
                rows = await conn.fetch(f"""
                    SELECT key, value, namespace, created_at, updated_at,
                           0.0 as score
                    FROM {self.table_name}
                    WHERE namespace LIKE $1 || '%'
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                """, namespace_prefix_str + "/", limit, offset)

            results = []
            for row in rows:
                results.append(SearchItem(
                    value=dict(row["value"]),
                    key=row["key"],
                    namespace=self._str_to_namespace(row["namespace"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    score=row["score"],
                ))

            return results

    async def adelete(
        self,
        namespace: Tuple[str, ...],
        key: str,
    ) -> None:
        """Delete a value.

        Args:
            namespace: Namespace tuple
            key: Key to delete
        """
        pool = await self._get_pool()
        namespace_str = self._namespace_to_str(namespace)

        async with pool.acquire() as conn:
            await conn.execute(f"""
                DELETE FROM {self.table_name}
                WHERE namespace = $1 AND key = $2
            """, namespace_str, key)

        logger.debug(f"[LangGraphStore] Deleted {namespace_str}/{key}")

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[Tuple[str, ...]] = None,
        limit: int = 100,
    ) -> List[Tuple[str, ...]]:
        """List namespaces with optional prefix filter.

        Args:
            prefix: Namespace prefix to filter
            limit: Max namespaces to return

        Returns:
            List of namespace tuples
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if prefix:
                prefix_str = self._namespace_to_str(prefix)
                rows = await conn.fetch(f"""
                    SELECT DISTINCT namespace
                    FROM {self.table_name}
                    WHERE namespace LIKE $1 || '%'
                    LIMIT $2
                """, prefix_str, limit)
            else:
                rows = await conn.fetch(f"""
                    SELECT DISTINCT namespace
                    FROM {self.table_name}
                    LIMIT $1
                """, limit)

            return [self._str_to_namespace(row["namespace"]) for row in rows]

    async def abatch(
        self,
        ops: List[Dict[str, Any]],
        *,
        check_duplicate: bool = False,
        deduplicate_first: bool = False,
    ) -> List[Any]:
        """Execute multiple operations in a single batch.

        Args:
            ops: List of operations, each with 'namespace', 'key', 'value' for put
                 or 'namespace', 'key' for get/delete
            check_duplicate: Check for duplicates on each put operation
            deduplicate_first: Deduplicate ops list before processing (removes dup ops)

        Returns:
            List of results (None for put/delete, Item for get, bool for put with check_duplicate)
        """
        if deduplicate_first:
            # Remove duplicate operations based on content hash
            ops = self._deduplicate_ops(ops)

        results = []
        for op in ops:
            op_type = op.get("op", "put")
            namespace = op.get("namespace")
            key = op.get("key")

            if op_type == "put":
                value = op.get("value", {})
                result = await self.aput(namespace, key, value, check_duplicate=check_duplicate)
                results.append(result)
            elif op_type == "get":
                item = await self.aget(namespace, key)
                results.append(item)
            elif op_type == "delete":
                await self.adelete(namespace, key)
                results.append(None)
            else:
                results.append(None)

        return results

    def _deduplicate_ops(self, ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate a list of put operations by content hash.

        Only affects 'put' operations - keeps first occurrence of each unique content.
        """
        seen_hashes: set = set()
        unique_ops = []

        for op in ops:
            if op.get("op", "put") != "put":
                unique_ops.append(op)
                continue

            value = op.get("value", {})
            content_hash = self._compute_content_hash(value, self.index_fields)

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_ops.append(op)

        if len(unique_ops) < len(ops):
            logger.info(
                f"[LangGraphStore] Deduplicated batch: {len(ops)} -> {len(unique_ops)} ops"
            )

        return unique_ops

    def batch(self, ops: List[Dict[str, Any]]) -> List[Any]:
        """Synchronous batch operations (calls async version).

        Note: This is a convenience wrapper for sync code.
        For best performance, use abatch() directly in async code.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.abatch(ops)
                )
                return future.result()
        else:
            # No event loop, create one
            return asyncio.run(self.abatch(ops))

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


async def get_langgraph_store(
    connection_string: str = None,
    embeddings: Embeddings = None,
    index_fields: List[str] = None,
) -> LangGraphPostgresStore:
    """Factory function to create a LangGraph Postgres Store.

    Args:
        connection_string: Postgres connection string (default: from config)
        embeddings: Embeddings instance (default: from factory)
        index_fields: Fields to index (default: ["content"])

    Returns:
        Configured LangGraphPostgresStore
    """
    from src.config import settings

    # Default connection string from settings
    if connection_string is None:
        connection_string = getattr(
            settings,
            "CONTEXT_ENGINE_POSTGRES_URL",
            None
        )
        # Fall back to POSTGRES_URL if CONTEXT_ENGINE_POSTGRES_URL not set
        if connection_string is None:
            connection_string = os.getenv("POSTGRES_URL", "postgresql://ulli:ulli_password@localhost:5432/ulli_db")

    # Default embeddings from factory
    if embeddings is None:
        from src.shared.embeddings.factory import get_embeddings
        embeddings = get_embeddings(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

    return LangGraphPostgresStore(
        connection_string=connection_string,
        embeddings=embeddings,
        index_fields=index_fields or ["content"],
    )
