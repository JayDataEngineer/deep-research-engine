"""Research Sessions - DEPRECATED - Use context_engine session tools instead.

!!! DEPRECATED !!!
===============

This module is DEPRECATED. All session management has been migrated to:
- Session tools in: src/extensions/features/tools/context_engine/tools/session.py
- Used via: from src.shared.tools.registry import get_tool

New usage pattern:
    from src.shared.tools.registry import get_tool

    create_session = get_tool("create_research_session")
    result = await create_session.ainvoke({
        "query": query,
        "use_case": use_case,
        "variant": variant,
        "run_id": run_id,
        "config": config
    })
    session_data = json.loads(result)
    session_id = session_data["session_id"]

Why this change:
- NO direct database connections - all via context_engine
- Single source of truth in context_engine storage systems
- Consistent storage across Postgres, Neo4j, and MinIO
- Session isolation with UUID across all systems

This file is kept for backward compatibility during migration but will be removed.
The SessionManager class below has direct asyncpg connections which violates
the architecture principle: "Context logic should be handled in context_engine."
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger

# Import centralized constants
from ..constants import DEFAULT_POSTGRES_CONNECTION_STRING

import asyncpg

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


# =============================================================================
# DEPRECATION WARNINGS
# =============================================================================

def _deprecated_warn(message: stacklevel=2) -> None:
    """Issue a deprecation warning."""
    warnings.warn(
        f"DEPRECATED: {message}\n"
        f"Use context_engine session tools from src.shared.tools.registry instead.\n"
        f"See: src/extensions/features/tools/context_engine/tools/session.py",
        DeprecationWarning,
        stacklevel=stacklevel
    )


# =============================================================================
# EXCEPTIONS
# =============================================================================

class SessionManagerError(Exception):
    """Base exception for SessionManager errors.

    DEPRECATED: Use context_engine session tools instead.
    """
    pass


# =============================================================================
# SESSION STATUS ENUM
# =============================================================================

class SessionStatus(str, Enum):
    """Status of a research session."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


# =============================================================================
# SESSION DATA MODEL
# =============================================================================

@dataclass
class ResearchSession:
    """A research session - the single source of truth for a run.

    All storage systems (Neo4j, Postgres, MinIO, LangGraph Store)
    reference this session_id for complete data lineage.

    Attributes:
        session_id: UUID primary key (used across ALL storage systems)
        run_id: Human-readable ID from RunLogger (for logging/observability)
        thread_id: LangGraph thread_id (for checkpoint/recovery)
        user_id: User who initiated the research
        status: Current session status
        query: Original research query
        use_case: Research use case (rag, longform_gen, soc_media_gen)
        variant: Config variant
        config: Full config snapshot (JSONB)
        graph_pointer: Current position in LangGraph (for resume)
        error: Error message if failed
        metadata: Additional metadata (JSONB)
        created_at: Session start time
        updated_at: Last status update
        completed_at: Session completion time
    """
    session_id: str
    run_id: str
    thread_id: str
    user_id: Optional[str] = None
    status: SessionStatus = SessionStatus.CREATED
    query: Optional[str] = None
    use_case: Optional[str] = None
    variant: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    graph_pointer: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        """Set timestamps on creation."""
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at

    @property
    def is_active(self) -> bool:
        """Check if session is still active (not terminal)."""
        return self.status in (
            SessionStatus.CREATED,
            SessionStatus.RUNNING,
            SessionStatus.PAUSED
        )

    @property
    def is_terminal(self) -> bool:
        """Check if session is in a terminal state."""
        return self.status in (
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.CANCELLED,
            SessionStatus.ROLLED_BACK
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "query": self.query,
            "use_case": self.use_case,
            "variant": self.variant,
            "config": json.dumps(self.config) if self.config else None,
            "graph_pointer": self.graph_pointer,
            "error": self.error,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


# =============================================================================
# SESSION STORAGE (PostgreSQL)
# =============================================================================

class SessionStorage:
    """PostgreSQL storage for research sessions.

    DEPRECATED: Use context_engine session tools instead.

    This module is kept for backward compatibility during migration.
    All session storage now goes through context_engine tools.

    New usage:
        from src.shared.tools.registry import get_tool
        create_session = get_tool("create_research_session")
        result = await create_session.ainvoke({...})

    The SINGLE SOURCE OF TRUTH for research sessions is now in context_engine's
    PostgresVectorStore, not this direct PostgreSQL connection.
    """

    # SQL Schema
    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS research_sessions (
        session_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        thread_id TEXT UNIQUE NOT NULL,
        user_id TEXT,
        status TEXT NOT NULL DEFAULT 'created',
        query TEXT,
        use_case TEXT,
        variant TEXT,
        config JSONB,
        graph_pointer TEXT,
        error TEXT,
        metadata JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        completed_at TIMESTAMPTZ
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON research_sessions(status);
    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON research_sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_run_id ON research_sessions(run_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON research_sessions(created_at DESC);

    -- Trigger to update updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    DROP TRIGGER IF EXISTS update_research_sessions_updated_at ON research_sessions;
    CREATE TRIGGER update_research_sessions_updated_at
        BEFORE UPDATE ON research_sessions
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Initialize database connection and create schema."""
        self._pool = await asyncpg.create_pool(self.connection_string, min_size=1, max_size=10)

        # Create schema
        async with self._pool.acquire() as conn:
            await conn.execute(self.SCHEMA_SQL)

        logger.info("[SESSION_STORAGE] Initialized research_sessions table")

    async def close(self) -> None:
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            logger.info("[SESSION_STORAGE] Closed connection pool")

    async def create(self, session: ResearchSession) -> ResearchSession:
        """Create a new session."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO research_sessions (
                    session_id, run_id, thread_id, user_id, status,
                    query, use_case, variant, config, graph_pointer,
                    error, metadata, created_at, updated_at, completed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """,
                session.session_id, session.run_id, session.thread_id, session.user_id, session.status.value,
                session.query, session.use_case, session.variant, json.dumps(session.config),
                session.graph_pointer, session.error, json.dumps(session.metadata),
                self._parse_datetime(session.created_at),
                self._parse_datetime(session.updated_at),
                self._parse_datetime(session.completed_at),
            )

        logger.info(f"[SESSION_STORAGE] Created session: {session.session_id}")
        return session

    @staticmethod
    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string to datetime object."""
        if not dt_str:
            return None
        if isinstance(dt_str, datetime):
            return dt_str
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    async def get(self, session_id: str) -> Optional[ResearchSession]:
        """Get a session by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM research_sessions WHERE session_id = $1", session_id)

        if not row:
            return None

        return self._row_to_session(row)

    async def get_by_thread_id(self, thread_id: str) -> Optional[ResearchSession]:
        """Get a session by LangGraph thread_id."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM research_sessions WHERE thread_id = $1", thread_id)

        if not row:
            return None

        return self._row_to_session(row)

    async def get_by_run_id(self, run_id: str) -> Optional[ResearchSession]:
        """Get a session by RunLogger run_id."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM research_sessions WHERE run_id = $1", run_id)

        if not row:
            return None

        return self._row_to_session(row)

    async def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        error: Optional[str] = None,
        graph_pointer: Optional[str] = None
    ) -> Optional[ResearchSession]:
        """Update session status."""
        # Build SET clauses and params dynamically
        set_clauses = ["status = $1"]
        params = [status.value]
        param_idx = 2

        if error is not None:
            set_clauses.append(f"error = ${param_idx}")
            params.append(error)
            param_idx += 1

        if graph_pointer is not None:
            set_clauses.append(f"graph_pointer = ${param_idx}")
            params.append(graph_pointer)
            param_idx += 1

        if status in (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED, SessionStatus.ROLLED_BACK):
            set_clauses.append(f"completed_at = ${param_idx}")
            params.append(datetime.now(timezone.utc))
            param_idx += 1

        # Add session_id as the last parameter for WHERE clause
        params.append(session_id)

        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE research_sessions
                SET {', '.join(set_clauses)}
                WHERE session_id = ${param_idx}
            """, *params)

        return await self.get(session_id)

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 100
    ) -> List[ResearchSession]:
        """List sessions with optional filters."""
        conditions = []
        params = []
        param_count = 0

        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status.value)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM research_sessions
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_count + 1}
            """, *params)

        return [self._row_to_session(row) for row in rows]

    @staticmethod
    def _row_to_session(row) -> ResearchSession:
        """Convert database row to ResearchSession."""
        return ResearchSession(
            session_id=row["session_id"],
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            status=SessionStatus(row["status"]),
            query=row["query"],
            use_case=row["use_case"],
            variant=row["variant"],
            config=row["config"] if row["config"] else {},
            graph_pointer=row["graph_pointer"],
            error=row["error"],
            metadata=row["metadata"] if row["metadata"] else {},
            created_at=row["created_at"].isoformat() if row["created_at"] else None,
            updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
            completed_at=row["completed_at"].isoformat() if row["completed_at"] else None,
        )


# =============================================================================
# SESSION MANAGER (Main API)
# =============================================================================

class SessionManager:
    """Session lifecycle manager for deep research.

    DEPRECATED: Use context_engine session tools instead.

    This class has direct database connections which violates the architecture
    principle: "Context logic should be handled in context_engine."

    Migrate to:
        from src.shared.tools.registry import get_tool
        create_session = get_tool("create_research_session")
        update_status = get_tool("update_session_status")
        cleanup = get_tool("cleanup_session")

    Why migrate:
    - NO direct database connections - all via context_engine
    - Single source of truth in context_engine storage systems
    - Consistent storage across Postgres, Neo4j, and MinIO
    - Session isolation with UUID across all systems

    This class is kept for backward compatibility during migration but will be removed.

    ========== OLD USAGE (DEPRECATED) ==========

    session = await session_manager.create_session(
        query="What is AI?",
        use_case="rag",
        variant="default",
        run_logger=run_logger,
        config=yaml_config,
    )

    ========== NEW USAGE (RECOMMENDED) ==========

    result = await create_session.ainvoke({
        "query": "What is AI?",
        "use_case": "rag",
        "variant": "default",
        "run_id": run_id,
        "config": yaml_config
    })
    session_data = json.loads(result)
    session_id = session_data["session_id"]
    """

    def __init__(self, connection_string: str):
        self.storage = SessionStorage(connection_string)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the session manager."""
        await self.storage.initialize()
        self._initialized = True
        logger.info("[SESSION_MANAGER] Initialized")

    async def close(self) -> None:
        """Close the session manager."""
        await self.storage.close()
        logger.info("[SESSION_MANAGER] Closed")

    def _require_initialized(self):
        """Check if manager is initialized."""
        if not self._initialized:
            raise SessionManagerError("SessionManager not initialized. Call initialize() first.")

    async def create_session(
        self,
        query: str,
        use_case: str,
        variant: str,
        run_id: str,
        config: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchSession:
        """Create a new research session.

        This generates:
        - session_id: UUID (primary key across ALL storage systems)
        - thread_id: LangGraph thread_id for checkpoint/recovery

        Args:
            query: Research query
            use_case: Use case name
            variant: Config variant
            run_id: RunLogger run_id (for observability)
            config: Full config snapshot
            user_id: Optional user ID
            metadata: Optional metadata

        Returns:
            ResearchSession with all IDs generated
        """
        self._require_initialized()

        # Generate unique session_id (UUID)
        session_id = str(uuid4())

        # Generate thread_id for LangGraph
        thread_id = f"deep_research_{use_case}_{variant}_{run_id}"

        # Create session
        session = ResearchSession(
            session_id=session_id,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status=SessionStatus.CREATED,
            query=query,
            use_case=use_case,
            variant=variant,
            config=config,
            metadata=metadata or {},
        )

        # Store in database
        await self.storage.create(session)

        logger.info(f"[SESSION_MANAGER] Created session {session_id} for run {run_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get session by ID."""
        self._require_initialized()
        return await self.storage.get(session_id)

    async def get_session_by_thread(self, thread_id: str) -> Optional[ResearchSession]:
        """Get session by LangGraph thread_id."""
        self._require_initialized()
        return await self.storage.get_by_thread_id(thread_id)

    async def set_running(self, session_id: str) -> ResearchSession:
        """Mark session as running."""
        self._require_initialized()
        return await self.storage.update_status(session_id, SessionStatus.RUNNING)

    async def set_completed(self, session_id: str) -> ResearchSession:
        """Mark session as completed."""
        self._require_initialized()
        return await self.storage.update_status(session_id, SessionStatus.COMPLETED)

    async def set_failed(self, session_id: str, error: str) -> ResearchSession:
        """Mark session as failed."""
        self._require_initialized()
        return await self.storage.update_status(session_id, SessionStatus.FAILED, error=error)

    async def set_paused(self, session_id: str, graph_pointer: str) -> ResearchSession:
        """Mark session as paused (for resume)."""
        self._require_initialized()
        return await self.storage.update_status(session_id, SessionStatus.PAUSED, graph_pointer=graph_pointer)

    async def rollback(self, session_id: str) -> Dict[str, Any]:
        """Rollback a session - delete ALL data from ALL storage systems.

        This is the "nuclear option" for cleanup:
        - Deletes Neo4j nodes/relationships with session_id
        - Deletes Postgres artifacts with session_id
        - Deletes MinIO files in session_id directory
        - Marks session as rolled_back

        Args:
            session_id: Session to rollback

        Returns:
            Dict with counts of deleted items
        """
        self._require_initialized()

        logger.warning(f"[SESSION_MANAGER] Rolling back session {session_id}")

        results = {
            "session_id": session_id,
            "neo4j_nodes": 0,
            "neo4j_relationships": 0,
            "postgres_artifacts": 0,
            "minio_files": 0,
        }

        # Get session info
        session = await self.storage.get(session_id)
        if not session:
            logger.error(f"[SESSION_MANAGER] Session {session_id} not found")
            raise ValueError(f"Session {session_id} not found")

        # 1. Cleanup Neo4j (if enabled)
        try:
            results["neo4j_nodes"] = await self._cleanup_neo4j(session_id)
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Neo4j cleanup failed: {e}")

        # 2. Cleanup Postgres artifacts
        try:
            results["postgres_artifacts"] = await self._cleanup_postgres(session_id)
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Postgres cleanup failed: {e}")

        # 3. Cleanup MinIO files
        try:
            results["minio_files"] = await self._cleanup_minio(session_id)
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] MinIO cleanup failed: {e}")

        # 4. Mark session as rolled back
        await self.storage.update_status(session_id, SessionStatus.ROLLED_BACK)

        logger.info(f"[SESSION_MANAGER] Session {session_id} rolled back: {results}")

        return results

    async def _cleanup_neo4j(self, session_id: str) -> int:
        """Delete all Neo4j nodes for a session.

        CRITICAL: Uses session_ids list (not single session_id) because
        entities track multiple sessions. We remove session_id from the
        session_ids list, and optionally delete nodes that now have empty
        session_ids lists.

        Returns:
            Number of nodes cleaned up (session_id removed from list)
        """
        try:
            from neo4j import AsyncGraphDatabase

            neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

            driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

            async with driver.session() as session:
                # Step 1: Remove session_id from all entities' session_ids lists
                # This is the "soft delete" - entity still exists but not for this session
                result = await session.run("""
                    MATCH (n)
                    WHERE $session_id IN n.session_ids
                    SET n.session_ids = [s IN n.session_ids WHERE s <> $session_id]
                    RETURN count(n) as updated_count
                """, session_id=session_id)

                record = await result.single()
                updated_count = record["updated_count"] if record else 0

                # Step 2: Delete nodes that now have empty session_ids lists
                # These are nodes that ONLY belonged to this session
                result = await session.run("""
                    MATCH (n)
                    WHERE n.session_ids = []
                    DETACH DELETE n
                    RETURN count(n) as deleted_count
                """)

                record = await result.single()
                deleted_count = record["deleted_count"] if record else 0

            await driver.close()

            # Total cleaned = nodes that had session_id removed
            # (some may still exist if shared with other sessions)
            return updated_count

        except ImportError:
            logger.warning("[SESSION_MANAGER] Neo4j driver not available, skipping Neo4j cleanup")
            return 0
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Neo4j cleanup error: {e}")
            return 0

    async def _cleanup_postgres(self, session_id: str) -> int:
        """Delete all Postgres artifacts for a session."""
        try:
            from src.extensions.features.tools.context_engine.storage import PostgresVectorStore
            from src.shared.embeddings.factory import get_embeddings_from_config

            # Create a minimal PostgresVectorStore for cleanup
            connection_string = DEFAULT_POSTGRES_CONNECTION_STRING

            # Need an embedding model to create PostgresVectorStore
            embedding_model = get_embeddings_from_config({
                "tool_settings": {
                    "embeddings": {
                        "provider": "huggingface",
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                }
            })

            store = PostgresVectorStore(
                connection_string=connection_string,
                embedding_model=embedding_model,
            )

            # Delete all artifacts with this session_id
            async with store._pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM artifacts WHERE session_id = $1
                """, session_id)

                # Get count
                result = await conn.fetchval("""
                    SELECT COUNT(*) FROM artifacts WHERE session_id = $1
                """, session_id)

            # Actually, we need to get the count BEFORE deleting
            async with store._pool.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM artifacts WHERE session_id = $1
                """, session_id)

                await conn.execute("""
                    DELETE FROM artifacts WHERE session_id = $1
                """, session_id)

            return count or 0

        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Postgres cleanup error: {e}")
            return 0

    async def _cleanup_minio(self, session_id: str) -> int:
        """Delete all MinIO files for a session."""
        try:
            from ..storage.minio_helpers import get_artifacts_backend

            backend = get_artifacts_backend(session_id)

            # List all files for this session
            files = backend.ls_info("/")

            # Delete each file
            count = 0
            for file_info in files:
                if backend.s3_backend.delete(file_info.path):
                    count += 1

            return count

        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] MinIO cleanup error: {e}")
            return 0

    async def recall_session(self, session_id: str) -> Dict[str, Any]:
        """Recall ALL data for a session across ALL storage systems.

        This is the "mirror image" of rollback - instead of deleting, it retrieves:
        - Session metadata from research_sessions table
        - Neo4j entities/relationships with this session_id
        - Postgres artifacts with this session_id
        - MinIO files in session_id directory
        - LangGraph checkpoints (via thread_id lookup)

        Args:
            session_id: Session to recall

        Returns:
            Dict with all session data
        """
        self._require_initialized()

        logger.info(f"[SESSION_MANAGER] Recalling session {session_id}")

        # Get session metadata
        session = await self.storage.get(session_id)
        if not session:
            logger.error(f"[SESSION_MANAGER] Session {session_id} not found")
            raise ValueError(f"Session {session_id} not found")

        results = {
            "session": session,
            "neo4j": {"entities": [], "relationships": []},
            "postgres_artifacts": [],
            "minio_files": [],
            "checkpoint": None,
        }

        # 1. Get Neo4j entities with this session_id
        try:
            results["neo4j"] = await self._recall_neo4j(session_id)
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Neo4j recall failed: {e}")

        # 2. Get Postgres artifacts
        try:
            results["postgres_artifacts"] = await self._recall_postgres(session_id)
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Postgres recall failed: {e}")

        # 3. Get MinIO files
        try:
            results["minio_files"] = await self._recall_minio(session_id)
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] MinIO recall failed: {e}")

        # 4. Get LangGraph checkpoint info (thread_id in session)
        results["checkpoint"] = {
            "thread_id": session.thread_id,
            "note": "Use checkpointer.get_tuple(configurable={'thread_id': thread_id}) to retrieve checkpoint"
        }

        logger.info(f"[SESSION_MANAGER] Recalled session {session_id}: "
                   f"{len(results['neo4j']['entities'])} Neo4j entities, "
                   f"{len(results['postgres_artifacts'])} Postgres artifacts, "
                   f"{len(results['minio_files'])} MinIO files")

        return results

    async def _recall_neo4j(self, session_id: str) -> Dict[str, Any]:
        """Recall Neo4j data for a session."""
        try:
            from neo4j import AsyncGraphDatabase

            neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

            driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

            async with driver.session() as session:
                # Get all entities with this session_id
                result = await session.run("""
                    MATCH (n)
                    WHERE $session_id IN n.session_ids
                    RETURN n.name as name, n.types as types, labels(n) as labels
                """, session_id=session_id)

                entities = []
                async for record in result:
                    entities.append({
                        "name": record["name"],
                        "types": record["types"],
                        "labels": record["labels"],
                    })

                # Get all relationships (optional - more complex)
                # For now, just return entities

            await driver.close()
            return {"entities": entities, "relationships": []}

        except ImportError:
            logger.warning("[SESSION_MANAGER] Neo4j driver not available")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Neo4j recall error: {e}")
            return {"entities": [], "relationships": []}

    async def _recall_postgres(self, session_id: str) -> List[Dict[str, Any]]:
        """Recall Postgres artifacts for a session."""
        try:
            from src.extensions.features.tools.context_engine.storage import PostgresVectorStore
            from src.shared.embeddings.factory import get_embeddings

            connection_string = DEFAULT_POSTGRES_CONNECTION_STRING

            # Need embedding model for PostgresVectorStore
            embedding_model = get_embeddings(provider="huggingface", model=None)

            store = PostgresVectorStore(
                connection_string=connection_string,
                embedding_model=embedding_model,
            )

            # Query artifacts with this session_id
            async with store._pool.acquire() as conn:
                result = await conn.fetch("""
                    SELECT id, artifact_type, content, session_id,
                           created_at, relevance_score, metadata
                    FROM artifacts
                    WHERE session_id = $1
                    ORDER BY created_at DESC
                """, session_id)

                artifacts = []
                for row in result:
                    artifacts.append({
                        "id": row["id"],
                        "artifact_type": row["artifact_type"],
                        "content": row["content"][:500] + "..." if row["content"] and len(row["content"]) > 500 else row["content"],
                        "session_id": row["session_id"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "relevance_score": row["relevance_score"],
                        "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    })

            return artifacts

        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] Postgres recall error: {e}")
            return []

    async def _recall_minio(self, session_id: str) -> List[Dict[str, Any]]:
        """Recall MinIO files for a session."""
        try:
            from ..storage.minio_helpers import get_artifacts_backend

            backend = get_artifacts_backend(session_id)

            # List all files for this session
            files = backend.ls_info("/")

            return [
                {
                    "path": f.path,
                    "size": f.size,
                    "modified_at": f.modified_at,
                }
                for f in files
            ]

        except Exception as e:
            logger.warning(f"[SESSION_MANAGER] MinIO recall error: {e}")
            return []


# =============================================================================
# SESSION CONTEXT MANAGER (for use in run_v4_spawner)
# =============================================================================

class SessionContext:
    """Context manager for session lifecycle.

    .. deprecated:: 1.0.0
        Use context_engine session tools instead.
        Replace with: `create_research_session` tool from central registry.

    Usage (deprecated - use context_engine tools instead):
        async with SessionContext(session_manager, ...) as session:
            # Run the research
            result = await graph.ainvoke(state, config)
            # Auto-marked as completed on exit

    New pattern:
        create_session = get_tool("create_research_session")
        result = await create_session.ainvoke({...})
        session_id = json.loads(result)["session_id"]
    """

    def __init__(
        self,
        manager: SessionManager,
        query: str,
        use_case: str,
        variant: str,
        run_id: str,
        config: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        _deprecated_warn("SessionContext", "Use context_engine session tools: get_tool('create_research_session')")
        self.manager = manager
        self.query = query
        self.use_case = use_case
        self.variant = variant
        self.run_id = run_id
        self.config = config
        self.user_id = user_id
        self.session: Optional[ResearchSession] = None

    async def __aenter__(self) -> ResearchSession:
        """Create session and mark as running."""
        _deprecated_warn("SessionContext.__aenter__", "Use context_engine session tools: get_tool('create_research_session')")
        self.session = await self.manager.create_session(
            query=self.query,
            use_case=self.use_case,
            variant=self.variant,
            run_id=self.run_id,
            config=self.config,
            user_id=self.user_id,
        )
        await self.manager.set_running(self.session.session_id)
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Mark session as completed or failed."""
        if exc_type is not None:
            error = f"{exc_type.__name__}: {exc_val}"
            await self.manager.set_failed(self.session.session_id, error)
        else:
            await self.manager.set_completed(self.session.session_id)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_session_manager: Optional[SessionManager] = None


async def get_session_manager() -> SessionManager:
    """Get or create the global SessionManager instance.

    .. deprecated:: 1.0.0
        Use context_engine session tools instead.
        Replace with: `get_tool("create_research_session")` from central registry.
        See: src/extensions/features/tools/context_engine/tools/session.py
    """
    _deprecated_warn("get_session_manager", "Use context_engine session tools: get_tool('create_research_session')")
    global _session_manager

    if _session_manager is None:
        _session_manager = SessionManager(DEFAULT_POSTGRES_CONNECTION_STRING)
        await _session_manager.initialize()

    return _session_manager


def get_session_manager_sync() -> SessionManager:
    """Get SessionManager without async initialization (already initialized).

    .. deprecated:: 1.0.0
        Use context_engine session tools instead.
        Replace with: `get_tool("create_research_session")` from central registry.
        See: src/extensions/features/tools/context_engine/tools/session.py
    """
    _deprecated_warn("get_session_manager_sync", "Use context_engine session tools: get_tool('create_research_session')")
    global _session_manager

    if _session_manager is None:
        raise SessionManagerError("SessionManager not initialized. Call get_session_manager() first.")

    return _session_manager


__all__ = [
    "SessionStatus",
    "ResearchSession",
    "SessionStorage",
    "SessionManager",
    "SessionContext",
    "get_session_manager",
    "get_session_manager_sync",
    "SessionManagerError",
]
