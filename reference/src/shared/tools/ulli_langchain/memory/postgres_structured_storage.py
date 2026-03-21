"""
PostgreSQL Structured Storage Tool

A PostgreSQL-based storage for user information that mirrors the SQLite
StructuredStorage functionality. Used for production deployments with
shared database access.

Tables:
- user_facts: User facts (key-value pairs)
- events: Time-stamped occurrences
- relationships: Entity triples
- documents: Full text searchable documents
"""

import asyncpg
import json
from .logging_config import get_logger
import os
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone

logger = get_logger(__name__)


def get_postgres_url() -> Optional[str]:
    """Get PostgreSQL connection string from environment."""
    return os.getenv("POSTGRES_URL") or os.getenv("CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING")


class PostgresStructuredStorage:
    """
    PostgreSQL-based structured storage for user information.

    Stores facts in a searchable SQL database with:
    - User facts (key-value pairs)
    - Events (time-stamped occurrences)
    - Relationships (entity triples)
    - Documents (full text searchable)

    This class is production-ready and supports:
    - Connection pooling
    - Concurrent access
    - Multi-instance deployments
    """

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ):
        """
        Initialize PostgreSQL structured storage.

        Args:
            postgres_url: PostgreSQL connection string (defaults to env var)
            pool_min_size: Minimum connection pool size
            pool_max_size: Maximum connection pool size
        """
        self.postgres_url = postgres_url or get_postgres_url()

        if not self.postgres_url:
            raise ValueError(
                "PostgreSQL URL is required for PostgresStructuredStorage. "
                "Set POSTGRES_URL environment variable or pass postgres_url parameter."
            )

        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool: Optional[asyncpg.Pool] = None

        # Initialize tables on first pool access
        self._initialized = False

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=60
            )

        # Initialize tables on first use
        if not self._initialized:
            await self._create_tables(self._pool)
            self._initialized = True

        return self._pool

    async def _create_tables(self, pool: asyncpg.Pool):
        """Create storage tables."""

        async with pool.acquire() as conn:
            # User facts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    value_type TEXT DEFAULT 'string',
                    metadata JSONB,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, key)
                )
            """)

            # Events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    event_date TEXT,
                    metadata JSONB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Relationships table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    metadata JSONB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, subject, predicate, object)
                )
            """)

            # Documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    doc_type TEXT DEFAULT 'note',
                    tags TEXT[],
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON user_facts(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(user_id, event_date)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id)")

        logger.info("[PostgresStructuredStorage] Tables created")

    async def store_fact(
        self,
        user_id: str,
        key: str,
        value: Any,
        value_type: str = "string",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a fact about a user."""
        try:
            value_str = str(value)
            metadata_json = json.dumps(metadata) if metadata else None

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO user_facts
                    (user_id, key, value, value_type, metadata, confidence, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (user_id, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        value_type = EXCLUDED.value_type,
                        metadata = EXCLUDED.metadata,
                        confidence = EXCLUDED.confidence,
                        updated_at = EXCLUDED.updated_at
                    """,
                    user_id, key, value_str, value_type, metadata_json, confidence, datetime.now(timezone.utc).isoformat()
                )
            return True
        except Exception as e:
            logger.warning(f"[PostgresStructuredStorage] Failed to store fact: {e}")
            return False

    async def get_fact(
        self,
        user_id: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a fact about a user."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_facts WHERE user_id = $1 AND key = $2",
                user_id, key
            )

        if row:
            return {
                "key": row["key"],
                "value": row["value"],
                "value_type": row["value_type"],
                "metadata": dict(row["metadata"]) if row["metadata"] else None,
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        return None

    async def store_event(
        self,
        user_id: str,
        event_type: str,
        description: str,
        event_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store an event."""
        try:
            metadata_json = json.dumps(metadata) if metadata else None

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO events
                    (user_id, event_type, description, event_date, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    user_id, event_type, description, event_date, metadata_json
                )
            return True
        except Exception as e:
            logger.warning(f"[PostgresStructuredStorage] Failed to store event: {e}")
            return False

    async def store_relationship(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a relationship."""
        try:
            metadata_json = json.dumps(metadata) if metadata else None

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO relationships
                    (user_id, subject, predicate, object, confidence, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (user_id, subject, predicate, object) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        metadata = EXCLUDED.metadata
                    """,
                    user_id, subject, predicate, obj, confidence, metadata_json
                )
            return True
        except Exception as e:
            logger.warning(f"[PostgresStructuredStorage] Failed to store relationship: {e}")
            return False

    async def search(
        self,
        user_id: str,
        query: str,
        search_in: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search across all stored data.

        Args:
            user_id: User identifier
            query: Search query
            search_in: List of areas to search ("facts", "events", "relationships", "documents")

        Returns:
            List of matching results
        """
        results = []
        search_in = search_in or ["facts", "events", "relationships", "documents"]
        pattern = f"%{query}%"

        pool = await self._get_pool()

        if "facts" in search_in:
            rows = await pool.fetch(
                "SELECT 'fact' as type, key, value, created_at FROM user_facts WHERE user_id = $1 AND (key LIKE $2 OR value LIKE $2)",
                user_id, pattern
            )
            for row in rows:
                results.append({
                    "type": "fact",
                    "key": row["key"],
                    "value": row["value"],
                    "created_at": row["created_at"],
                })

        if "events" in search_in:
            rows = await pool.fetch(
                "SELECT 'event' as type, event_type, description, event_date FROM events WHERE user_id = $1 AND description LIKE $2",
                user_id, pattern
            )
            for row in rows:
                results.append({
                    "type": "event",
                    "event_type": row["event_type"],
                    "description": row["description"],
                    "event_date": row["event_date"],
                })

        if "relationships" in search_in:
            rows = await pool.fetch(
                "SELECT 'relationship' as type, subject, predicate, object FROM relationships WHERE user_id = $1 AND (subject LIKE $2 OR object LIKE $2)",
                user_id, pattern
            )
            for row in rows:
                results.append({
                    "type": "relationship",
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                })

        return results

    async def get_all_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all facts for a user."""
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM user_facts WHERE user_id = $1 ORDER BY created_at DESC",
            user_id
        )

        return [
            {
                "key": row["key"],
                "value": row["value"],
                "value_type": row["value_type"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    async def get_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of all stored data for a user."""
        facts = await self.get_all_facts(user_id)

        summary = {
            "fact_count": len(facts),
            "facts": {f["key"]: f["value"] for f in facts},
        }

        return summary

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("[PostgresStructuredStorage] Connection pool closed")


def create_postgres_structured_storage(
    postgres_url: Optional[str] = None,
    pool_min_size: int = 2,
    pool_max_size: int = 10,
) -> PostgresStructuredStorage:
    """Create a PostgreSQL structured storage instance."""
    return PostgresStructuredStorage(
        postgres_url=postgres_url,
        pool_min_size=pool_min_size,
        pool_max_size=pool_max_size
    )


def get_structured_storage(
    backend: Literal["auto", "sqlite", "postgres"] = "auto",
    postgres_url: Optional[str] = None,
):
    """
    Factory function to get the appropriate structured storage.

    Args:
        backend: Which backend to use ("auto" detects from POSTGRES_URL env var)
        postgres_url: Optional PostgreSQL connection string

    Returns:
        StructuredStorage instance (SQLite or PostgreSQL)
    """
    from .structured_storage import StructuredStorage

    if backend == "auto":
        # Detect from environment
        if get_postgres_url() or postgres_url:
            backend = "postgres"
        else:
            backend = "sqlite"

    if backend == "postgres":
        return create_postgres_structured_storage(postgres_url=postgres_url)
    else:
        return StructuredStorage(backend="sqlite")
