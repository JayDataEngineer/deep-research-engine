"""
Session management tools for Context Engine.

Provides @tool decorated functions for research session management.
These tools enable deep_research to use context_engine for session
storage instead of direct asyncpg connections.

Tools:
    - create_research_session: Create a new research session
    - get_session_status: Get current session status
    - update_session_status: Update status (running, completed, failed)
    - list_sessions: List sessions for a user/query
    - cleanup_session: Clean up session data from all storage systems

Usage from deep_research:
    from src.shared.tools.registry import get_tool

    create_session = get_tool("create_research_session")
    result = await create_session.ainvoke({
        "query": "What is AI?",
        "use_case": "rag",
        "variant": "default",
        "run_id": "run_123",
        "config": {...}
    })
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import uuid4

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SESSION STATUS
# =============================================================================

class SessionStatus:
    """Session status constants."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"

    TERMINAL_STATES = [COMPLETED, FAILED, CANCELLED, ROLLED_BACK]
    ACTIVE_STATES = [CREATED, RUNNING, PAUSED]


# =============================================================================
# SESSION TOOLS
# =============================================================================

@tool
async def create_research_session(
    query: str,
    use_case: str,
    variant: Optional[str] = None,
    run_id: str = "",
    config: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    runtime: Optional[ToolRuntime] = None,
) -> str:
    """Create a new research session in context_engine.

    This generates:
    - session_id: UUID (primary key across ALL storage systems)
    - thread_id: LangGraph thread_id for checkpoint/recovery

    Args:
        query: Research query
        use_case: Use case name (rag, longform_gen, etc.)
        variant: Config variant
        run_id: RunLogger run_id (for observability)
        config: Full config snapshot
        user_id: Optional user ID
        metadata: Optional metadata
        runtime: ToolRuntime for access to context engine

    Returns:
        JSON string with session details

    Example:
        >>> result = await create_research_session.ainvoke({
        ...     "query": "What is AI?",
        ...     "use_case": "rag",
        ...     "variant": "default",
        ...     "run_id": "run_123"
        ... })
        >>> session = json.loads(result)
        >>> session["session_id"]
        "550e8400-e29b-41d4-a716-446655440000"
    """
    # Normalize variant to string (default to "default" if None)
    variant_str = variant or "default"

    # Generate IDs
    session_id = str(uuid4())
    thread_id = f"deep_research_{use_case}_{variant_str}_{run_id}"
    now = datetime.now(timezone.utc).isoformat()

    session_data = {
        "session_id": session_id,
        "run_id": run_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "status": SessionStatus.CREATED,
        "query": query,
        "use_case": use_case,
        "variant": variant_str,
        "config": config or {},
        "metadata": metadata or {},
        "created_at": now,
        "updated_at": now,
    }

    # Get context engine from state or create direct connection
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    # Store session in Postgres
    if context_engine and context_engine.postgres_store:
        try:
            await _store_session(context_engine.postgres_store, session_data)
            logger.info(f"[SESSION] Created session {session_id} via context_engine")
        except Exception as e:
            logger.error(f"[SESSION] Failed to create session: {e}")
            return json.dumps({"error": str(e), "session_id": session_id, "thread_id": thread_id})
    else:
        # Fallback: direct Postgres connection
        try:
            await _store_session_direct(session_data)
            logger.info(f"[SESSION] Created session {session_id} via direct connection")
        except Exception as e:
            logger.error(f"[SESSION] Failed to create session (direct): {e}")
            return json.dumps({"error": str(e), "session_id": session_id, "thread_id": thread_id})

    return json.dumps(session_data, default=str)


@tool
async def get_session_status(
    session_id: str,
    runtime: Optional[ToolRuntime] = None,
) -> str:
    """Get the status of a research session.

    Args:
        session_id: The session ID to query
        runtime: ToolRuntime for access to context engine

    Returns:
        JSON string with session status details
    """
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    try:
        if context_engine and context_engine.postgres_store:
            session = await _get_session(context_engine.postgres_store, session_id)
        else:
            session = await _get_session_direct(session_id)

        if session:
            return json.dumps(session, default=str)
        else:
            return json.dumps({"error": f"Session {session_id} not found"})

    except Exception as e:
        logger.error(f"[SESSION] Failed to get session: {e}")
        return json.dumps({"error": str(e)})


@tool
async def update_session_status(
    session_id: str,
    status: str,
    error: Optional[str] = None,
    graph_pointer: Optional[str] = None,
    runtime: Optional[ToolRuntime] = None,
) -> str:
    """Update the status of a research session.

    Args:
        session_id: The session ID to update
        status: New status (running, completed, failed, paused, cancelled)
        error: Optional error message (for failed status)
        graph_pointer: Optional graph position (for paused status)
        runtime: ToolRuntime for access to context engine

    Returns:
        JSON string with updated session details
    """
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    update_data = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if error is not None:
        update_data["error"] = error

    if graph_pointer is not None:
        update_data["graph_pointer"] = graph_pointer

    # Set completed_at for terminal states
    if status in SessionStatus.TERMINAL_STATES:
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

    try:
        if context_engine and context_engine.postgres_store:
            await _update_session(context_engine.postgres_store, session_id, update_data)
        else:
            await _update_session_direct(session_id, update_data)

        logger.info(f"[SESSION] Updated session {session_id} to status: {status}")

        # Return updated session using helper function directly
        if context_engine and context_engine.postgres_store:
            session = await _get_session(context_engine.postgres_store, session_id)
        else:
            session = await _get_session_direct(session_id)

        if session:
            return json.dumps(session, default=str)
        else:
            return json.dumps({"error": f"Session {session_id} not found", "session_id": session_id})

    except Exception as e:
        logger.error(f"[SESSION] Failed to update session: {e}")
        return json.dumps({"error": str(e), "session_id": session_id})


@tool
async def list_sessions(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    runtime: Optional[ToolRuntime] = None,
) -> str:
    """List research sessions with optional filters.

    Args:
        user_id: Filter by user ID
        status: Filter by status
        limit: Maximum number of results
        runtime: ToolRuntime for access to context engine

    Returns:
        JSON string with list of sessions
    """
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    try:
        if context_engine and context_engine.postgres_store:
            sessions = await _list_sessions(context_engine.postgres_store, user_id, status, limit)
        else:
            sessions = await _list_sessions_direct(user_id, status, limit)

        return json.dumps({"sessions": sessions, "count": len(sessions)}, default=str)

    except Exception as e:
        logger.error(f"[SESSION] Failed to list sessions: {e}")
        return json.dumps({"error": str(e), "sessions": []})


@tool
async def cleanup_session(
    session_id: str,
    runtime: Optional[ToolRuntime] = None,
) -> str:
    """Clean up session data from all storage systems.

    This deletes:
    - Postgres artifacts with this session_id
    - MinIO files in session_id directory
    - Neo4j entities (removes session_id from session_ids list)
    - Marks session as rolled_back

    Args:
        session_id: Session to clean up
        runtime: ToolRuntime for access to context engine

    Returns:
        JSON string with cleanup results
    """
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    results = {
        "session_id": session_id,
        "postgres_artifacts": 0,
        "minio_files": 0,
        "neo4j_nodes": 0,
        "status": "success",
    }

    # 1. Cleanup Postgres artifacts
    try:
        results["postgres_artifacts"] = await _cleanup_postgres_artifacts(session_id, context_engine)
    except Exception as e:
        logger.warning(f"[SESSION] Postgres cleanup failed: {e}")
        results["postgres_error"] = str(e)

    # 2. Cleanup MinIO files
    try:
        results["minio_files"] = await _cleanup_minio_files(session_id, context_engine)
    except Exception as e:
        logger.warning(f"[SESSION] MinIO cleanup failed: {e}")
        results["minio_error"] = str(e)

    # 3. Cleanup Neo4j entities
    try:
        results["neo4j_nodes"] = await _cleanup_neo4j_entities(session_id)
    except Exception as e:
        logger.warning(f"[SESSION] Neo4j cleanup failed: {e}")
        results["neo4j_error"] = str(e)

    # 4. Mark session as rolled back
    try:
        if context_engine and context_engine.postgres_store:
            await _update_session(context_engine.postgres_store, session_id, {"status": SessionStatus.ROLLED_BACK})
        else:
            await _update_session_direct(session_id, {"status": SessionStatus.ROLLED_BACK})
    except Exception as e:
        logger.warning(f"[SESSION] Failed to mark session as rolled back: {e}")

    logger.info(f"[SESSION] Cleaned up session {session_id}: {results}")
    return json.dumps(results, default=str)


# =============================================================================
# HELPER FUNCTIONS - Context Engine Storage
# =============================================================================

async def _store_session(postgres_store, session_data: Dict[str, Any]) -> None:
    """Store session in Postgres via context_engine."""
    import asyncpg

    async with postgres_store._pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO research_sessions (
                session_id, run_id, thread_id, user_id, status,
                query, use_case, variant, config, metadata, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
            session_data["session_id"],
            session_data["run_id"],
            session_data["thread_id"],
            session_data["user_id"],
            session_data["status"],
            session_data["query"],
            session_data["use_case"],
            session_data["variant"],
            json.dumps(session_data["config"]),
            json.dumps(session_data["metadata"]),
            session_data["created_at"],
            session_data["updated_at"],
        )


async def _get_session(postgres_store, session_id: str) -> Optional[Dict[str, Any]]:
    """Get session from Postgres via context_engine."""
    async with postgres_store._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM research_sessions WHERE session_id = $1",
            session_id
        )

    if not row:
        return None

    return _row_to_dict(row)


async def _update_session(postgres_store, session_id: str, update_data: Dict[str, Any]) -> None:
    """Update session in Postgres via context_engine."""
    from dateutil.parser import parse as parse_datetime

    set_clauses = []
    params = []
    param_idx = 1

    # Datetime fields that need conversion from string to datetime
    datetime_fields = {'updated_at', 'completed_at', 'created_at'}

    for key, value in update_data.items():
        set_clauses.append(f"{key} = ${param_idx}")
        # Convert datetime strings to actual datetime objects for asyncpg
        if key in datetime_fields and isinstance(value, str):
            value = parse_datetime(value)
        params.append(value)
        param_idx += 1

    params.append(session_id)

    async with postgres_store._pool.acquire() as conn:
        await conn.execute(f"""
            UPDATE research_sessions
            SET {', '.join(set_clauses)}
            WHERE session_id = ${param_idx}
        """, *params)


async def _list_sessions(postgres_store, user_id: Optional[str], status: Optional[str], limit: int) -> List[Dict]:
    """List sessions from Postgres via context_engine."""
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
        params.append(status)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    async with postgres_store._pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT * FROM research_sessions
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_count + 1}
        """, *params)

    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert database row to dictionary."""
    # Helper to safely convert JSONB columns
    def _jsonb_to_dict(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        # asyncpg returns Record objects for JSONB
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}

    return {
        "session_id": row["session_id"],
        "run_id": row["run_id"],
        "thread_id": row["thread_id"],
        "user_id": row["user_id"],
        "status": row["status"],
        "query": row["query"],
        "use_case": row["use_case"],
        "variant": row["variant"],
        "config": _jsonb_to_dict(row["config"]),
        "graph_pointer": row["graph_pointer"],
        "error": row["error"],
        "metadata": _jsonb_to_dict(row["metadata"]),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
    }


# =============================================================================
# HELPER FUNCTIONS - Direct Storage (Fallback)
# =============================================================================

async def _get_connection_string() -> str:
    """Get Postgres connection string."""
    from src.extensions.content.deep_research.constants import DEFAULT_POSTGRES_CONNECTION_STRING
    return DEFAULT_POSTGRES_CONNECTION_STRING


async def _store_session_direct(session_data: Dict[str, Any]) -> None:
    """Store session using direct asyncpg connection."""
    import asyncpg
    from dateutil.parser import parse as parse_datetime

    conn_string = await _get_connection_string()
    conn = await asyncpg.connect(conn_string)

    try:
        # Parse ISO strings to datetime objects for asyncpg
        created_at = session_data["created_at"]
        updated_at = session_data["updated_at"]

        # Convert string to datetime if needed
        if isinstance(created_at, str):
            created_at = parse_datetime(created_at)
        if isinstance(updated_at, str):
            updated_at = parse_datetime(updated_at)

        await conn.execute("""
            INSERT INTO research_sessions (
                session_id, run_id, thread_id, user_id, status,
                query, use_case, variant, config, metadata, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
            session_data["session_id"],
            session_data["run_id"],
            session_data["thread_id"],
            session_data["user_id"],
            session_data["status"],
            session_data["query"],
            session_data["use_case"],
            session_data["variant"],
            json.dumps(session_data["config"]),
            json.dumps(session_data["metadata"]),
            created_at,
            updated_at,
        )
    finally:
        await conn.close()


async def _get_session_direct(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session using direct asyncpg connection."""
    import asyncpg

    conn_string = await _get_connection_string()
    conn = await asyncpg.connect(conn_string)

    try:
        row = await conn.fetchrow(
            "SELECT * FROM research_sessions WHERE session_id = $1",
            session_id
        )

        if not row:
            return None

        return _row_to_dict(row)
    finally:
        await conn.close()


async def _update_session_direct(session_id: str, update_data: Dict[str, Any]) -> None:
    """Update session using direct asyncpg connection."""
    import asyncpg
    from dateutil.parser import parse as parse_datetime

    conn_string = await _get_connection_string()
    conn = await asyncpg.connect(conn_string)

    try:
        set_clauses = []
        params = []
        param_idx = 1

        # Datetime fields that need conversion from string to datetime
        datetime_fields = {'updated_at', 'completed_at', 'created_at'}

        for key, value in update_data.items():
            set_clauses.append(f"{key} = ${param_idx}")
            # Convert datetime strings to actual datetime objects for asyncpg
            if key in datetime_fields and isinstance(value, str):
                value = parse_datetime(value)
            params.append(value)
            param_idx += 1

        params.append(session_id)

        await conn.execute(f"""
            UPDATE research_sessions
            SET {', '.join(set_clauses)}
            WHERE session_id = ${param_idx}
        """, *params)
    finally:
        await conn.close()


async def _list_sessions_direct(user_id: Optional[str], status: Optional[str], limit: int) -> List[Dict]:
    """List sessions using direct asyncpg connection."""
    import asyncpg

    conn_string = await _get_connection_string()
    conn = await asyncpg.connect(conn_string)

    try:
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
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await conn.fetch(f"""
            SELECT * FROM research_sessions
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_count + 1}
        """, *params)

        return [_row_to_dict(row) for row in rows]
    finally:
        await conn.close()


# =============================================================================
# CLEANUP HELPERS
# =============================================================================

async def _cleanup_postgres_artifacts(session_id: str, context_engine) -> int:
    """Delete Postgres artifacts for a session."""
    if context_engine and context_engine.postgres_store:
        async with context_engine.postgres_store._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM artifacts WHERE session_id = $1",
                session_id
            )
            await conn.execute(
                "DELETE FROM artifacts WHERE session_id = $1",
                session_id
            )
            return count or 0

    # Fallback: direct connection
    import asyncpg
    conn_string = await _get_connection_string()
    conn = await asyncpg.connect(conn_string)

    try:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM artifacts WHERE session_id = $1",
            session_id
        )
        await conn.execute(
            "DELETE FROM artifacts WHERE session_id = $1",
            session_id
        )
        return count or 0
    finally:
        await conn.close()


async def _cleanup_minio_files(session_id: str, context_engine) -> int:
    """Delete MinIO files for a session."""
    from ..storage.s3_backend import S3Backend

    s3_backend = None
    if context_engine and context_engine.s3_backend:
        s3_backend = context_engine.s3_backend
    else:
        s3_backend = S3Backend(
            bucket=os.getenv("MINIO_BUCKET", "context-engine"),
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:8899/minio"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin")),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")),
        )

    # List and delete files
    try:
        files = s3_backend.ls(f"/raw/{session_id}/")
        count = 0
        for file_info in files:
            if s3_backend.delete(file_info.get("path", file_info.get("name", ""))):
                count += 1
        return count
    except Exception as e:
        logger.warning(f"[SESSION] MinIO cleanup error: {e}")
        return 0


async def _cleanup_neo4j_entities(session_id: str) -> int:
    """Remove session_id from Neo4j entities."""
    try:
        from neo4j import AsyncGraphDatabase

        neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        async with driver.session() as session:
            # Remove session_id from entities' session_ids lists
            result = await session.run("""
                MATCH (n)
                WHERE $session_id IN n.session_ids
                SET n.session_ids = [s IN n.session_ids WHERE s <> $session_id]
                RETURN count(n) as updated_count
            """, session_id=session_id)

            record = await result.single()
            updated_count = record["updated_count"] if record else 0

            # Delete nodes with empty session_ids
            await session.run("""
                MATCH (n)
                WHERE n.session_ids = []
                DETACH DELETE n
            """)

        await driver.close()
        return updated_count

    except ImportError:
        logger.warning("[SESSION] Neo4j driver not available")
        return 0
    except Exception as e:
        logger.warning(f"[SESSION] Neo4j cleanup error: {e}")
        return 0


# =============================================================================
# TOOL LISTS FOR REGISTRY
# =============================================================================

CONTEXT_ENGINE_SESSION_TOOLS = [
    create_research_session,
    get_session_status,
    update_session_status,
    list_sessions,
    cleanup_session,
]


__all__ = [
    "create_research_session",
    "get_session_status",
    "update_session_status",
    "list_sessions",
    "cleanup_session",
    "SessionStatus",
    "CONTEXT_ENGINE_SESSION_TOOLS",
]
