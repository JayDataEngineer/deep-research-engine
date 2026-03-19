"""Unified Memory Router for Context Engine.

This module provides a CompositeBackend that routes all memory operations
through a single interface, implementing the 4 entry points:

Entry Point A: Deep Agents dump to S3 when context runs out
    → /data/raw/dumps/

Entry Point B: Web agent synthesis after critic pass
    → /data/raw/web/

Entry Point C: Workers in fan-out write insights
    → /data/insights/ (Postgres with vectorization)

Entry Point D: Neo4j graph builder before synthesis
    → /data/graph/ (Knowledge graph entities)

Cross-thread Memory: LangGraph Store for semantic search
    → /memories/

Usage:
    from deepagents import create_deep_agent
    from src.extensions.features.tools.context_engine.storage.unified_memory_router import (
        create_unified_memory_router
    )

    agent = create_deep_agent(
        model=llm,
        backend=create_unified_memory_router,  # Factory function
        store=langgraph_store,
    )

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     DEEP AGENT                                  │
    │                                                                  │
    │  write_file("/data/raw/...")     → S3 (raw dumps, web scrapes) │
    │  write_file("/data/insights/...") → Postgres (vectorized)      │
    │  write_file("/data/graph/...")    → Neo4j (knowledge graph)    │
    │  write_file("/memories/...")      → LangGraph Store (search)   │
    │  write_file("/tmp/...")           → StateBackend (ephemeral)   │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES (matching deepagents BackendProtocol)
# =============================================================================

@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    name: str
    size: int
    modified_at: str
    is_dir: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "modified_at": self.modified_at,
            "is_dir": self.is_dir,
        }


@dataclass
class WriteResult:
    """Result of a write operation."""
    path: str
    error: Optional[str] = None
    files_update: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "error": self.error,
            "files_update": self.files_update,
        }


@dataclass
class EditResult:
    """Result of an edit operation."""
    path: str
    error: Optional[str] = None
    files_update: Optional[dict] = None
    occurrences: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "error": self.error,
            "files_update": self.files_update,
            "occurrences": self.occurrences,
        }


@dataclass
class GrepMatch:
    """A single grep match."""
    path: str
    line_number: int
    line: str
    match_start: int
    match_end: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "line_number": self.line_number,
            "line": self.line,
            "match_start": self.match_start,
            "match_end": self.match_end,
        }


# =============================================================================
# POSTGRES BACKEND (BackendProtocol implementation)
# =============================================================================

class PostgresBackend:
    """Postgres backend implementing deepagents BackendProtocol.

    Routes write operations to Postgres with automatic vectorization.
    This implements Entry Point C: Workers in fan-out write insights.

    Features:
    - Automatic content hashing for deduplication
    - Async vectorization via background tasks (optional)
    - Session isolation via session_id
    - Connection pooling via shared pool

    Path Format:
        /data/insights/{session_id}/{artifact_type}/{filename}

    Example:
        write_file("/data/insights/session123/findings/worker_0.md", content)
    """

    def __init__(
        self,
        runtime: Optional["ToolRuntime"] = None,
        postgres_store: Optional[Any] = None,
        session_id: Optional[str] = None,
        enable_async_vectorization: bool = False,
    ):
        """Initialize Postgres backend.

        Args:
            runtime: ToolRuntime for accessing session context
            postgres_store: PostgresVectorStore instance (shared connection pool)
            session_id: Explicit session_id (overrides runtime)
            enable_async_vectorization: Enable background vectorization
        """
        self.runtime = runtime
        self._postgres_store = postgres_store
        self._session_id = session_id
        self.enable_async_vectorization = enable_async_vectorization

    @property
    def session_id(self) -> str:
        """Get session_id from runtime or explicit setting."""
        if self._session_id:
            return self._session_id
        if self.runtime:
            state = getattr(self.runtime, "state", {})
            if isinstance(state, dict):
                return state.get("session_id", "global")
        return "global"

    @property
    def postgres_store(self):
        """Lazy-load PostgresVectorStore."""
        if self._postgres_store is None:
            from .postgres_store import PostgresVectorStore
            # Get from context engine or create new
            try:
                from ..context_engine_integration import get_context_engine_from_context
                context_engine = get_context_engine_from_context()
                if context_engine and hasattr(context_engine, 'postgres_store'):
                    self._postgres_store = context_engine.postgres_store
            except Exception as e:
                logger.warning(f"[PostgresBackend] Could not get context engine: {e}")
        return self._postgres_store

    def _parse_path(self, file_path: str) -> dict:
        """Parse path to extract components.

        Expected format: /data/insights/{session_id}/{artifact_type}/{filename}
        """
        parts = file_path.strip("/").split("/")
        result = {
            "session_id": self.session_id,
            "artifact_type": "finding",
            "filename": file_path.split("/")[-1],
        }

        # Try to extract session_id and artifact_type from path
        if len(parts) >= 3:
            # /data/insights/{session_id}/{artifact_type}/{filename}
            if parts[0] == "data" and parts[1] == "insights":
                result["session_id"] = parts[2] if len(parts) > 2 else self.session_id
                result["artifact_type"] = parts[3] if len(parts) > 3 else "finding"
                result["filename"] = "/".join(parts[4:]) if len(parts) > 4 else parts[-1]

        return result

    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        """List artifacts in Postgres (not a typical filesystem)."""
        # Postgres doesn't have a traditional filesystem structure
        # Return empty list - use search_memories tool for querying
        return []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read artifact content by ID or path.

        For Postgres, we read by artifact_id or search by filename.
        """
        # This is a non-standard operation for Postgres
        # Typically you'd use search_memories tool instead
        return f"Error: Use search_memories tool to query Postgres artifacts"

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content to Postgres with automatic vectorization.

        This is the main entry point for storing insights.

        CRITICAL: Raises exceptions on failure (no silent errors).
        """
        try:
            parsed = self._parse_path(file_path)
            store = self.postgres_store

            if store is None:
                # Fallback: log and return error (no silent failure)
                error_msg = "PostgresVectorStore not available"
                logger.error(f"[PostgresBackend] {error_msg}")
                raise RuntimeError(error_msg)

            # Create artifact
            from ..artifacts.base import BaseArtifact, ArtifactType

            # Map artifact_type string to enum
            artifact_type_map = {
                "finding": ArtifactType.FINDING,
                "memory": ArtifactType.MEMORY,
                "entity": ArtifactType.ENTITY,
                "claim": ArtifactType.CLAIM,
                "citation": ArtifactType.CITATION,
            }
            artifact_type = artifact_type_map.get(
                parsed["artifact_type"].lower(),
                ArtifactType.FINDING
            )

            # Generate ID from content hash + path
            content_hash = hashlib.sha256(
                f"{parsed['session_id']}:{file_path}:{content}".encode()
            ).hexdigest()[:32]

            artifact = BaseArtifact(
                id=content_hash,
                content=content,
                artifact_type=artifact_type,
                session_id=parsed["session_id"],
                metadata={
                    "file_path": file_path,
                    "filename": parsed["filename"],
                    "source": "deep_agent_write",
                }
            )

            # Add to store (async internally)
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in async context, schedule the write
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    artifact_id, is_new = loop.run_in_executor(
                        pool,
                        asyncio.run,
                        store.add(artifact)
                    )
            else:
                artifact_id, is_new = asyncio.run(store.add(artifact))

            logger.info(
                f"[PostgresBackend] Wrote artifact {artifact_id} "
                f"(new={is_new}) to session {parsed['session_id']}"
            )

            return WriteResult(path=file_path)

        except Exception as e:
            logger.error(f"[PostgresBackend] Write failed: {e}")
            # CRITICAL: Raise instead of returning error (no silent failures)
            raise RuntimeError(f"PostgresBackend write failed: {e}") from e

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> EditResult:
        """Edit not supported for Postgres backend."""
        return EditResult(
            path=file_path,
            error="Edit not supported for Postgres backend. Use write() to update."
        )

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None
    ) -> Union[list[GrepMatch], str]:
        """Search not supported via filesystem interface."""
        return "Error: Use search_memories tool for semantic search"

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Glob not supported for Postgres backend."""
        return []

    # =========================================================================
    # ASYNC OPERATIONS
    # =========================================================================

    async def als_info(self, path: str = "/") -> list[FileInfo]:
        return self.ls_info(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self.read(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async write with proper async store access."""
        try:
            parsed = self._parse_path(file_path)
            store = self.postgres_store

            if store is None:
                raise RuntimeError("PostgresVectorStore not available")

            # Create artifact
            from ..artifacts.base import BaseArtifact, ArtifactType

            artifact_type_map = {
                "finding": ArtifactType.FINDING,
                "memory": ArtifactType.MEMORY,
                "entity": ArtifactType.ENTITY,
            }
            artifact_type = artifact_type_map.get(
                parsed["artifact_type"].lower(),
                ArtifactType.FINDING
            )

            content_hash = hashlib.sha256(
                f"{parsed['session_id']}:{file_path}:{content}".encode()
            ).hexdigest()[:32]

            artifact = BaseArtifact(
                id=content_hash,
                content=content,
                artifact_type=artifact_type,
                session_id=parsed["session_id"],
                metadata={
                    "file_path": file_path,
                    "filename": parsed["filename"],
                    "source": "deep_agent_awrite",
                }
            )

            # Async add
            artifact_id, is_new = await store.add(artifact)

            logger.info(
                f"[PostgresBackend] Async wrote artifact {artifact_id} "
                f"(new={is_new}) to session {parsed['session_id']}"
            )

            # Optional: Trigger background vectorization
            if self.enable_async_vectorization:
                await self._trigger_background_vectorization(
                    artifact_id, file_path, content
                )

            return WriteResult(path=file_path)

        except Exception as e:
            logger.error(f"[PostgresBackend] Async write failed: {e}")
            raise RuntimeError(f"PostgresBackend async write failed: {e}") from e

    async def _trigger_background_vectorization(
        self,
        artifact_id: str,
        file_path: str,
        content: str
    ):
        """Trigger background vectorization task.

        Uses LangGraph background runs to process embeddings off the main thread.
        """
        # TODO: Implement with LangGraph client
        # from langgraph_sdk import get_client
        # client = get_client()
        # await client.runs.create(
        #     thread_id=self.session_id,
        #     assistant_id="vectorization_worker",
        #     input={"artifact_id": artifact_id, "content": content}
        # )
        pass

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None
    ) -> Union[list[GrepMatch], str]:
        return self.grep_raw(pattern, path, glob)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self.glob_info(pattern, path)


# =============================================================================
# NEO4J BACKEND (BackendProtocol implementation)
# =============================================================================

class Neo4jBackend:
    """Neo4j backend implementing deepagents BackendProtocol.

    Routes entity/relationship writes to Neo4j knowledge graph.
    This implements Entry Point D: Neo4j graph builder.

    Features:
    - Session isolation via session_ids array
    - MERGE semantics for idempotent entity creation
    - Concurrent worker support via accumulator lists

    Path Format:
        /data/graph/{session_id}/entities/{entity_name}
        /data/graph/{session_id}/relationships/{from}_{rel}_{to}

    Example:
        write_file("/data/graph/session123/entities/Apple", "Company|Technology company")
    """

    def __init__(
        self,
        runtime: Optional["ToolRuntime"] = None,
        neo4j_store: Optional[Any] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize Neo4j backend.

        Args:
            runtime: ToolRuntime for accessing session context
            neo4j_store: Neo4jStore instance
            session_id: Explicit session_id (overrides runtime)
        """
        self.runtime = runtime
        self._neo4j_store = neo4j_store
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        """Get session_id from runtime or explicit setting."""
        if self._session_id:
            return self._session_id
        if self.runtime:
            state = getattr(self.runtime, "state", {})
            if isinstance(state, dict):
                return state.get("session_id", "global")
        return "global"

    @property
    def neo4j_store(self):
        """Lazy-load Neo4jStore."""
        if self._neo4j_store is None:
            try:
                from ..context_engine_integration import get_context_engine_from_context
                context_engine = get_context_engine_from_context()
                if context_engine and hasattr(context_engine, 'neo4j_store'):
                    self._neo4j_store = context_engine.neo4j_store
            except Exception as e:
                logger.warning(f"[Neo4jBackend] Could not get context engine: {e}")
        return self._neo4j_store

    def _parse_entity_path(self, file_path: str) -> dict:
        """Parse entity path.

        Format: /data/graph/{session_id}/entities/{entity_name}
        """
        parts = file_path.strip("/").split("/")
        result = {
            "session_id": self.session_id,
            "entity_name": None,
            "entity_type": "Entity",
            "description": "",
        }

        if "entities" in parts:
            entities_idx = parts.index("entities")
            if len(parts) > entities_idx + 1:
                result["entity_name"] = parts[entities_idx + 1]
            if len(parts) > entities_idx + 2:
                result["entity_type"] = parts[entities_idx + 2]

        return result

    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        """List entities in Neo4j."""
        return []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read entity by name."""
        return "Error: Use graph_entity_search tool to query Neo4j"

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write entity or relationship to Neo4j.

        Content format for entities:
            "EntityType|Description text here"

        Content format for relationships:
            "FROM_ENTITY|RELATIONSHIP_TYPE|TO_ENTITY"
        """
        try:
            store = self.neo4j_store

            if store is None:
                raise RuntimeError("Neo4jStore not available")

            parsed = self._parse_entity_path(file_path)

            if parsed["entity_name"]:
                # Parse content: "EntityType|Description"
                content_parts = content.split("|", 1)
                entity_type = content_parts[0].strip() if content_parts else "Entity"
                description = content_parts[1].strip() if len(content_parts) > 1 else ""

                # Create entity artifact
                from ..artifacts.osint import EntityArtifact

                artifact = EntityArtifact(
                    id=hashlib.sha256(
                        f"{parsed['session_id']}:{parsed['entity_name']}".encode()
                    ).hexdigest()[:32],
                    content=description,
                    entity_name=parsed["entity_name"],
                    entity_type=entity_type,
                    session_id=parsed["session_id"],
                    metadata={
                        "file_path": file_path,
                        "source": "deep_agent_write",
                    }
                )

                # Add to Neo4j (async internally)
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        loop.run_in_executor(
                            pool,
                            asyncio.run,
                            store.add(artifact)
                        )
                else:
                    asyncio.run(store.add(artifact))

                logger.info(
                    f"[Neo4jBackend] Created entity '{parsed['entity_name']}' "
                    f"of type '{entity_type}' in session {parsed['session_id']}"
                )

            return WriteResult(path=file_path)

        except Exception as e:
            logger.error(f"[Neo4jBackend] Write failed: {e}")
            raise RuntimeError(f"Neo4jBackend write failed: {e}") from e

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> EditResult:
        """Edit not supported for Neo4j backend."""
        return EditResult(
            path=file_path,
            error="Edit not supported for Neo4j backend. Use write() to add entities."
        )

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None
    ) -> Union[list[GrepMatch], str]:
        return "Error: Use graph_entity_search tool for graph queries"

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return []

    # =========================================================================
    # ASYNC OPERATIONS
    # =========================================================================

    async def als_info(self, path: str = "/") -> list[FileInfo]:
        return self.ls_info(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self.read(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async write to Neo4j."""
        try:
            store = self.neo4j_store

            if store is None:
                raise RuntimeError("Neo4jStore not available")

            parsed = self._parse_entity_path(file_path)

            if parsed["entity_name"]:
                content_parts = content.split("|", 1)
                entity_type = content_parts[0].strip() if content_parts else "Entity"
                description = content_parts[1].strip() if len(content_parts) > 1 else ""

                from ..artifacts.osint import EntityArtifact

                artifact = EntityArtifact(
                    id=hashlib.sha256(
                        f"{parsed['session_id']}:{parsed['entity_name']}".encode()
                    ).hexdigest()[:32],
                    content=description,
                    entity_name=parsed["entity_name"],
                    entity_type=entity_type,
                    session_id=parsed["session_id"],
                    metadata={
                        "file_path": file_path,
                        "source": "deep_agent_awrite",
                    }
                )

                await store.add(artifact)

                logger.info(
                    f"[Neo4jBackend] Async created entity '{parsed['entity_name']}' "
                    f"of type '{entity_type}' in session {parsed['session_id']}"
                )

            return WriteResult(path=file_path)

        except Exception as e:
            logger.error(f"[Neo4jBackend] Async write failed: {e}")
            raise RuntimeError(f"Neo4jBackend async write failed: {e}") from e

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None
    ) -> Union[list[GrepMatch], str]:
        return self.grep_raw(pattern, path, glob)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self.glob_info(pattern, path)


# =============================================================================
# UNIFIED MEMORY ROUTER FACTORY
# =============================================================================

def create_unified_memory_router(
    runtime: "ToolRuntime",
    context_engine: Optional[Any] = None,
    session_id: Optional[str] = None,
    include_s3: bool = True,
    include_postgres: bool = True,
    include_neo4j: bool = True,
    include_memories: bool = True,
    enable_async_vectorization: bool = False,
) -> "CompositeBackend":
    """Create a Unified Memory Router using CompositeBackend.

    This is the main entry point for creating a memory router that handles
    all 4 entry points for deep research:

    Entry Point A: Deep Agents dump to S3 when context runs out
        → /data/raw/dumps/ → S3Backend

    Entry Point B: Web agent synthesis after critic pass
        → /data/raw/web/ → S3Backend

    Entry Point C: Workers in fan-out write insights
        → /data/insights/ → PostgresBackend (with vectorization)

    Entry Point D: Neo4j graph builder before synthesis
        → /data/graph/ → Neo4jBackend

    Cross-thread Memory: LangGraph Store for semantic search
        → /memories/ → StoreBackend

    Ephemeral State: Temporary scratchpad
        → /* → StateBackend (default)

    Args:
        runtime: ToolRuntime from deepagents
        context_engine: ContextEngineExtension instance (optional, will try to get from context)
        session_id: Explicit session_id (overrides runtime)
        include_s3: Include S3/MinIO routes
        include_postgres: Include Postgres route
        include_neo4j: Include Neo4j route
        include_memories: Include LangGraph Store route
        enable_async_vectorization: Enable background vectorization for Postgres

    Returns:
        CompositeBackend with all routes configured

    Example:
        from deepagents import create_deep_agent
        from src.extensions.features.tools.context_engine.storage.unified_memory_router import (
            create_unified_memory_router
        )

        agent = create_deep_agent(
            model=llm,
            backend=create_unified_memory_router,
            store=langgraph_store,
        )

        # Now agents can write to:
        # - /data/raw/dumps/session123/... → S3
        # - /data/raw/web/session123/... → S3
        # - /data/insights/session123/... → Postgres (vectorized)
        # - /data/graph/session123/... → Neo4j
        # - /memories/... → LangGraph Store
    """
    try:
        from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
    except ImportError:
        raise ImportError(
            "deepagents package required. Install with: uv add deepagents"
        )

    # Get context engine if not provided
    if context_engine is None:
        try:
            from ..context_engine_integration import get_context_engine_from_context
            context_engine = get_context_engine_from_context()
        except Exception as e:
            logger.warning(f"[UnifiedMemoryRouter] Could not get context engine: {e}")

    # Default state backend for ephemeral scratchpad
    default_backend = StateBackend(runtime)

    # Build routes
    routes = {}

    # Entry Point A & B: S3/MinIO for raw data
    if include_s3:
        from .minio_backend import MinioBackend

        # Raw dumps (context overflow)
        routes["/data/raw/dumps/"] = MinioBackend(
            prefix="dumps/session",
            runtime=runtime,
            session_id=session_id,
        )

        # Web scrapes and raw content
        routes["/data/raw/web/"] = MinioBackend(
            prefix="web/session",
            runtime=runtime,
            session_id=session_id,
        )

        # Also support legacy /dumps/ and /web/ paths
        routes["/dumps/"] = routes["/data/raw/dumps/"]
        routes["/web/"] = routes["/data/raw/web/"]

    # Entry Point C: Postgres for insights (vectorized)
    if include_postgres:
        postgres_store = None
        if context_engine and hasattr(context_engine, 'postgres_store'):
            postgres_store = context_engine.postgres_store

        routes["/data/insights/"] = PostgresBackend(
            runtime=runtime,
            postgres_store=postgres_store,
            session_id=session_id,
            enable_async_vectorization=enable_async_vectorization,
        )

    # Entry Point D: Neo4j for knowledge graph
    if include_neo4j:
        neo4j_store = None
        if context_engine and hasattr(context_engine, 'neo4j_store'):
            neo4j_store = context_engine.neo4j_store

        routes["/data/graph/"] = Neo4jBackend(
            runtime=runtime,
            neo4j_store=neo4j_store,
            session_id=session_id,
        )

    # Cross-thread memory: LangGraph Store
    if include_memories:
        routes["/memories/"] = StoreBackend(runtime)

    logger.info(
        f"[UnifiedMemoryRouter] Created router with routes: {list(routes.keys())}"
    )

    return CompositeBackend(default=default_backend, routes=routes)


def unified_memory_router_factory(
    session_id: Optional[str] = None,
    context_engine: Optional[Any] = None,
    **kwargs
) -> Callable[["ToolRuntime"], "CompositeBackend"]:
    """Factory function for creating unified memory routers.

    Use with create_deep_agent:

        agent = create_deep_agent(
            model=llm,
            backend=unified_memory_router_factory(session_id="research_123"),
            store=langgraph_store,
        )

    Args:
        session_id: Explicit session_id
        context_engine: ContextEngineExtension instance
        **kwargs: Additional arguments for create_unified_memory_router

    Returns:
        Factory function
    """
    def factory(runtime: "ToolRuntime") -> "CompositeBackend":
        return create_unified_memory_router(
            runtime=runtime,
            context_engine=context_engine,
            session_id=session_id,
            **kwargs
        )

    return factory


__all__ = [
    # Types
    "FileInfo",
    "WriteResult",
    "EditResult",
    "GrepMatch",
    # Backends
    "PostgresBackend",
    "Neo4jBackend",
    # Factory
    "create_unified_memory_router",
    "unified_memory_router_factory",
]
