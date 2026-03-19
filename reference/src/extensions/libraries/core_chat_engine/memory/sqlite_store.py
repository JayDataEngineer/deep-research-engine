"""SQLite-based BaseStore implementation for long-term memory.

This implements LangGraph's BaseStore API for persistent key-value storage
with optional semantic search via embeddings.

Based on LangGraph's BaseStore interface:
- put(namespace, key, value): Store or update a value
- get(namespace, key): Retrieve a value
- delete(namespace, key): Remove a value
- search(namespace_prefix, query, filter): Search with optional semantic query
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from pydantic import BaseModel

# Try to import LangGraph's BaseStore - fall back to abstract base if not available
try:
    from langgraph.store.base import BaseStore

    HAS_LANGGRAPH_STORE = True
except ImportError:
    logger.warning(
        "LangGraph BaseStore not available - using abstract base implementation"
    )
    HAS_LANGGRAPH_STORE = False
    # Define abstract base for compatibility
    from abc import ABC, abstractmethod

    class BaseStore(ABC):
        """Abstract base for Store compatibility when LangGraph is not installed."""

        @abstractmethod
        def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Any]:
            pass

        @abstractmethod
        def put(
            self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any]
        ) -> None:
            pass

        @abstractmethod
        def delete(self, namespace: Tuple[str, ...], key: str) -> None:
            pass

        @abstractmethod
        def list(
        self,
        prefix: Tuple[str, ...],
        limit: int = 10,
    ) -> Iterator[Tuple[Tuple[str, ...], str, Dict[str, Any]]]:
            pass


logger = logging.getLogger(__name__)


def _ensure_string(value: Union[str, Tuple[str, ...]]) -> str:
    """Convert a namespace tuple to a string."""
    if isinstance(value, str):
        return value
    return "/".join(value)


def _ensure_tuple(value: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """Convert a namespace string to a tuple."""
    if isinstance(value, tuple):
        return value
    return tuple(value.split("/"))


class SQLiteStore(BaseStore):
    """SQLite-based key-value store for long-term memory.

    Implements LangGraph's BaseStore API with:
    - Persistent SQLite storage
    - JSON serialization for complex values
    - Optional embedding-based semantic search
    - Thread-safe operations

    Example:
        ```python
        store = SQLiteStore(db_path="memories.db")
        # Store a memory
        store.put(("user", "alice"), "memory_1", {"content": "Likes coffee"})
        # Retrieve
        memory = store.get(("user", "alice"), "memory_1")
        # Search
        results = list(store.search(("user", "alice"), query="coffee"))
        ```
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        embeddings_client: Optional[Any] = None,
        use_cache: bool = True,
    ):
        """Initialize the SQLite store.

        Args:
            db_path: Path to SQLite database file
            embeddings_client: Optional embeddings client for semantic search
            use_cache: Whether to wrap embeddings in CacheBackedEmbeddings (default: True)
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

        # Wrap embeddings with cache if provided and caching enabled
        self.embeddings_client = self._setup_cached_embeddings(
            embeddings_client, use_cache
        )

        # Create database and tables
        self._init_db()

    def _setup_cached_embeddings(
        self,
        embeddings_client: Optional[Any],
        use_cache: bool,
    ) -> Optional[Any]:
        """Set up CacheBackedEmbeddings if embeddings client provided.

        Uses LangChain's CacheBackedEmbeddings to avoid recomputing embeddings
        for the same text. Cache is stored in memory for fast lookups.

        IMPORTANT: Cache is only beneficial for API-based embeddings (OpenAI, etc.)
        where each call has network latency and costs money. For local embeddings
        (HuggingFace), the cache overhead may exceed the benefit.

        Args:
            embeddings_client: The underlying embeddings model
            use_cache: Whether to enable caching

        Returns:
            CacheBackedEmbeddings wrapper or original client
        """
        if not embeddings_client or not use_cache:
            return embeddings_client

        # Check if this is an API-based provider (benefits from caching)
        # Local providers like HuggingFace are already fast enough
        client_type = type(embeddings_client).__name__
        api_based_providers = {'OpenAIEmbeddings', 'AzureOpenAIEmbeddings', 'AnthropicEmbeddings'}

        if client_type not in api_based_providers:
            # Local embeddings - cache overhead may not be worth it
            logger.debug(f"[SQLiteStore] Skipping cache for local embeddings: {client_type}")
            return embeddings_client

        try:
            from langchain_classic.embeddings.cache import CacheBackedEmbeddings
            from langchain_core.stores import InMemoryStore

            # Create a namespace based on the model to avoid collisions
            model_name = getattr(embeddings_client, 'model_name',
                       getattr(embeddings_client, 'model', 'default'))
            namespace = f"memory-embeddings-{model_name}"

            # Use in-memory store for caching
            # For production with multiple workers, use PostgresByteStore or RedisByteStore
            cache_store = InMemoryStore()

            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                embeddings_client,
                cache_store,
                namespace=namespace,
            )

            logger.info(f"[SQLiteStore] Enabled embedding cache for {client_type} with namespace: {namespace}")
            return cached_embeddings

        except ImportError as e:
            logger.warning(f"[SQLiteStore] CacheBackedEmbeddings not available: {e}")
            return embeddings_client
        except Exception as e:
            logger.warning(f"[SQLiteStore] Failed to setup embedding cache: {e}")
            return embeddings_client

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = self._get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                PRIMARY KEY (namespace, key)
            )
        """
        )

        # Index for search performance
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_namespace
            ON memories(namespace)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON memories(created_at DESC)
        """
        )

        conn.commit()
        logger.info(f"SQLiteStore initialized at {self.db_path}")

    # =========================================================================
    # BaseStore API Implementation
    # =========================================================================

    def get(
        self, namespace: Tuple[str, ...], key: str
    ) -> Optional[Dict[str, Any]]:
        """Get a value from the store.

        Args:
            namespace: Hierarchical namespace tuple (e.g., ("user", "alice"))
            key: Key to retrieve

        Returns:
            The stored value dict, or None if not found
        """
        namespace_str = _ensure_string(namespace)
        conn = self._get_connection()

        with self._lock:
            # Update access count and timestamp
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE namespace = ? AND key = ?
            """,
                (datetime.utcnow().isoformat(), namespace_str, key),
            )

            row = conn.execute(
                "SELECT value FROM memories WHERE namespace = ? AND key = ?",
                (namespace_str, key),
            ).fetchone()

            if row is None:
                return None

            try:
                return json.loads(row["value"])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode value for {namespace}/{key}: {e}")
                return None

    def put(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        check_duplicate: bool = False,
    ) -> bool:
        """Store or update a value in the store.

        Args:
            namespace: Hierarchical namespace tuple
            key: Key to store under
            value: Value dict to store (will be JSON serialized)
            check_duplicate: If True, check for duplicate content before storing

        Returns:
            True if stored, False if duplicate and check_duplicate=True
        """
        namespace_str = _ensure_string(namespace)
        conn = self._get_connection()

        # Optional duplicate check
        if check_duplicate:
            content = value.get("content") or value.get("text", "")
            if content and self._is_duplicate_content(namespace_str, content):
                logger.debug(f"[SQLiteStore] Skipping duplicate content in {namespace_str}")
                return False

        # Serialize value
        try:
            value_json = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize value: {e}")
            raise

        # Generate embedding if client available and value has text content
        embedding = None
        if self.embeddings_client:
            content = value.get("content") or value.get("text", "")
            if content:
                embedding = self._generate_embedding(content)

        with self._lock:
            conn.execute(
                """
                INSERT INTO memories (namespace, key, value, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    value = excluded.value,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
            """,
                (
                    namespace_str,
                    key,
                    value_json,
                    sqlite3.Binary(embedding) if embedding else None,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

        logger.debug(f"Stored {namespace}/{key}")
        return True

    def _is_duplicate_content(self, namespace: str, content: str) -> bool:
        """Check if content already exists in namespace (exact hash match).

        Args:
            namespace: Namespace string
            content: Content to check

        Returns:
            True if duplicate found, False otherwise
        """
        import hashlib

        # Normalize content for consistent hashing
        normalized = " ".join(content.split()).lower().strip()
        content_hash = hashlib.md5(normalized.encode()).hexdigest()

        conn = self._get_connection()

        with self._lock:
            # Check all entries in namespace for matching content
            rows = conn.execute(
                "SELECT value FROM memories WHERE namespace = ?",
                (namespace,),
            ).fetchall()

            for row in rows:
                try:
                    existing = json.loads(row["value"])
                    existing_content = existing.get("content") or existing.get("text", "")
                    if existing_content:
                        existing_normalized = " ".join(existing_content.split()).lower().strip()
                        existing_hash = hashlib.md5(existing_normalized.encode()).hexdigest()
                        if existing_hash == content_hash:
                            return True
                except (json.JSONDecodeError, KeyError):
                    continue

        return False

    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a value from the store.

        Args:
            namespace: Hierarchical namespace tuple
            key: Key to delete
        """
        namespace_str = _ensure_string(namespace)
        conn = self._get_connection()

        with self._lock:
            conn.execute(
                "DELETE FROM memories WHERE namespace = ? AND key = ?",
                (namespace_str, key),
            )
            conn.commit()

        logger.debug(f"Deleted {namespace}/{key}")

    def list(
        self,
        prefix: Tuple[str, ...],
        limit: int = 10,
    ) -> Iterator[Tuple[Tuple[str, ...], str, Dict[str, Any]]]:
        """List items in a given namespace prefix.

        Args:
            prefix: Namespace prefix to list under
            limit: Maximum number of items to return

        Yields:
            Tuples of (namespace, key, value)
        """
        prefix_str = _ensure_string(prefix) + "%"
        conn = self._get_connection()

        with self._lock:
            rows = conn.execute(
                """
                SELECT namespace, key, value
                FROM memories
                WHERE namespace LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
            """,
                (prefix_str, limit),
            ).fetchall()

        for row in rows:
            namespace_tuple = _ensure_tuple(row["namespace"])
            try:
                value = json.loads(row["value"])
                yield (namespace_tuple, row["key"], value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode value for {row['namespace']}/{row['key']}")

    # =========================================================================
    # Search Implementation
    # =========================================================================

    def search(
        self,
        prefix: Tuple[str, ...],
        query: Optional[str] = None,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[Tuple[str, ...], str, Dict[str, Any]]]:
        """Search for items with optional semantic query.

        Args:
            prefix: Namespace prefix to search under
            query: Optional natural language query for semantic search
            limit: Maximum results to return
            filter: Optional dict of field filters (not implemented yet)

        Yields:
            Tuples of (namespace, key, value) ordered by relevance
        """
        prefix_str = _ensure_string(prefix) + "%"
        conn = self._get_connection()

        if query and self.embeddings_client:
            # Semantic search using embeddings
            query_embedding = self._generate_embedding(query)

            with self._lock:
                # Get all items in namespace with embeddings
                rows = conn.execute(
                    """
                    SELECT namespace, key, value, embedding
                    FROM memories
                    WHERE namespace LIKE ? AND embedding IS NOT NULL
                    LIMIT 100
                """,
                    (prefix_str,),
                ).fetchall()

            # Calculate similarity scores
            results = []
            for row in rows:
                if row["embedding"]:
                    stored_embedding = bytes(row["embedding"])
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    results.append((similarity, row))

            # Sort by similarity and return top results
            results.sort(key=lambda x: x[0], reverse=True)
            for similarity, row in results[:limit]:
                namespace_tuple = _ensure_tuple(row["namespace"])
                try:
                    value = json.loads(row["value"])
                    value["_relevance"] = float(similarity)
                    yield (namespace_tuple, row["key"], value)
                except json.JSONDecodeError:
                    pass

        else:
            # Fallback to simple listing by recency
            yield from self.list(prefix, limit=limit)

    def _generate_embedding(self, text: str) -> bytes:
        """Generate embedding for text.

        Uses the embeddings client if available, otherwise returns None.
        """
        if not self.embeddings_client:
            return None

        try:
            # Use the embeddings client's embed_query method
            # This works with LangChain Embeddings interface
            embedding_list = self.embeddings_client.embed_query(text)

            # Convert list to bytes
            import struct

            return struct.pack(f"{len(embedding_list)}f", *embedding_list)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _cosine_similarity(self, embedding1: bytes, embedding2: bytes) -> float:
        """Calculate cosine similarity between two embedding bytes."""
        import struct

        # Unpack bytes to float arrays
        size1 = len(embedding1) // 4
        size2 = len(embedding2) // 4

        if size1 != size2:
            return 0.0

        vec1 = struct.unpack(f"{size1}f", embedding1)
        vec2 = struct.unpack(f"{size2}f", embedding2)

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self, namespace_prefix: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
        """Get statistics about stored memories.

        Args:
            namespace_prefix: Optional namespace to scope stats

        Returns:
            Dict with count, size, and other stats
        """
        conn = self._get_connection()

        if namespace_prefix:
            prefix_str = _ensure_string(namespace_prefix) + "%"
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE namespace LIKE ?",
                (prefix_str,),
            ).fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        return {
            "total_memories": count,
            "db_path": str(self.db_path),
        }

    # =========================================================================
    # Batch Operations (Required by BaseStore)
    # =========================================================================

    def batch(
        self,
        operations: List[Tuple[str, Tuple[str, ...], str]],
    ) -> List[Any]:
        """Execute multiple store operations in a single transaction.

        Args:
            operations: List of tuples (operation, namespace, key) where operation is:
                - "get": Retrieve value
                - "put": Store/update value (value must be provided separately)
                - "delete": Remove value

        Returns:
            List of results for each operation

        Note: For "put" operations, use individual put() calls instead.
        """
        results = []
        conn = self._get_connection()

        try:
            for operation in operations:
                if len(operation) < 3:
                    results.append(None)
                    continue

                op_type, namespace, key = operation[0], operation[1], operation[2]

                if op_type == "get":
                    results.append(self.get(namespace, key))
                elif op_type == "delete":
                    self.delete(namespace, key)
                    results.append(None)
                else:
                    results.append(None)

            return results

        except Exception as e:
            logger.error(f"Batch operation failed: {e}")
            return results

    async def abatch(
        self,
        operations: List[Tuple[str, Tuple[str, ...], str]],
    ) -> List[Any]:
        """Async version of batch operations.

        Args:
            operations: List of tuples (operation, namespace, key)

        Returns:
            List of results for each operation
        """
        # Since SQLite operations are fast, run batch synchronously
        return self.batch(operations)

    # =========================================================================
    # Namespace Listing (Required by BaseStore)
    # =========================================================================

    def list_namespaces(self, prefix: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, ...]]:
        """List all namespaces in the store.

        Args:
            prefix: Optional namespace prefix to filter

        Returns:
            List of namespace tuples
        """
        conn = self._get_connection()

        if prefix:
            prefix_str = _ensure_string(prefix) + "%"
            rows = conn.execute(
                "SELECT DISTINCT namespace FROM memories WHERE namespace LIKE ?",
                (prefix_str,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT DISTINCT namespace FROM memories").fetchall()

        # Convert string namespaces back to tuples
        namespaces = []
        for (namespace_str,) in rows:
            # Parse namespace string back to tuple
            # Format: "memories/alice/preferences"
            parts = namespace_str.split("/")
            namespaces.append(tuple(parts))

        return namespaces

    async def alist_namespaces(
        self,
        prefix: Optional[Tuple[str, ...]] = None,
    ) -> List[Tuple[str, ...]]:
        """Async version of list_namespaces.

        Args:
            prefix: Optional namespace prefix to filter

        Returns:
            List of namespace tuples
        """
        return self.list_namespaces(prefix=prefix)

    # =========================================================================
    # TTL Support (Required by BaseStore)
    # =========================================================================

    @property
    def supports_ttl(self) -> bool:
        """Whether this store supports TTL (Time To Live)."""
        return False  # SQLiteStore doesn't support TTL currently

    @property
    def ttl_config(self) -> Optional[Dict[str, Any]]:
        """TTL configuration if supported."""
        return None

    def clear_namespace(self, namespace_prefix: Tuple[str, ...]) -> int:
        """Clear all memories under a namespace prefix.

        Args:
            namespace_prefix: Namespace prefix to clear

        Returns:
            Number of items deleted
        """
        prefix_str = _ensure_string(namespace_prefix) + "%"
        conn = self._get_connection()

        with self._lock:
            cursor = conn.execute(
                "DELETE FROM memories WHERE namespace LIKE ?",
                (prefix_str,),
            )
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Cleared {deleted} memories under {namespace_prefix}")
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("SQLiteStore connection closed")


# =============================================================================
# Factory Functions
# =============================================================================

_global_store: Optional[SQLiteStore] = None
_store_lock = threading.Lock()


def get_memory_store(
    db_path: Optional[Union[str, Path]] = None,
    embeddings_client: Optional[Any] = None,
    use_embeddings: bool = True,
    use_cache: bool = True,
) -> SQLiteStore:
    """Get or create the global memory store instance.

    Args:
        db_path: Optional path to database (default: data/outputs/memories.db)
        embeddings_client: Optional embeddings client for semantic search.
                          If None and use_embeddings=True, creates one from factory.
        use_embeddings: Whether to enable semantic search (default: True)
        use_cache: Whether to cache embeddings with CacheBackedEmbeddings (default: True)

    Returns:
        SQLiteStore instance with optional semantic search capability

    Example:
        >>> store = get_memory_store()
        >>> store.put(("user", "alice"), "mem1", {"content": "likes coffee"})
        >>> # Semantic search with cached embeddings
        >>> results = store.search(("user",), query="caffeine drinks")
    """
    global _global_store

    with _store_lock:
        if _global_store is None:
            if db_path is None:
                from src.constants import CHECKPOINTS_DIR

                db_path = CHECKPOINTS_DIR / "memories.db"

            # Initialize embeddings client if not provided and embeddings are enabled
            if embeddings_client is None and use_embeddings:
                try:
                    from src.shared.embeddings import get_embeddings
                    embeddings_client = get_embeddings()
                    logger.info("[MemoryStore] Enabled semantic search with embeddings")
                except Exception as e:
                    logger.warning(f"[MemoryStore] Could not initialize embeddings: {e}")
                    logger.warning("[MemoryStore] Falling back to keyword-based search")

            _global_store = SQLiteStore(
                db_path=db_path,
                embeddings_client=embeddings_client,
                use_cache=use_cache,
            )
            logger.info(f"Created global memory store at {db_path}")

        return _global_store
