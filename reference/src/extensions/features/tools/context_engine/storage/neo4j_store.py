"""
Neo4j graph store for entity relationships (Level 4).

Perfect for OSINT investigations where you need to trace connections
between people, orgs, locations.
"""
from typing import List, Dict, Any, Tuple, Optional
import logging
import os

from neo4j import AsyncGraphDatabase

from .base import StorageBackend, StorageCapability
from ..artifacts.osint import EntityArtifact, RelationshipArtifact
from ..artifacts.base import BaseArtifact, ArtifactType
from ..health_check import HealthCheckResult, HealthStatus

logger = logging.getLogger(__name__)

# Default credentials from environment variables
DEFAULT_NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
DEFAULT_NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


class Neo4jStore(StorageBackend):
    """Neo4j graph store for entity relationships.

    This is your "Knowledge Graph" - perfect for OSINT investigations
    where you need to trace connections between people, orgs, locations.

    Uses AsyncGraphDatabase for proper async/non-blocking operations.
    """

    def __init__(
        self,
        name: str = "neo4j",
        url: str = None,
        username: str = None,
        password: str = None,
        database: str = "neo4j",
        config: dict = None
    ):
        super().__init__(name, config or {})
        # Use environment variables as defaults if not provided
        self._url = url or DEFAULT_NEO4J_URL
        self._username = username or DEFAULT_NEO4J_USERNAME
        self._password = password or DEFAULT_NEO4J_PASSWORD

        if not self._password:
            raise ValueError(
                "Neo4j password required. Set NEO4J_PASSWORD environment variable "
                "or pass password parameter."
            )

        self.driver = AsyncGraphDatabase.driver(
            self._url,
            auth=(self._username, self._password)
        )
        self.database = database

    @property
    def url(self) -> str:
        """Get the Neo4j connection URL."""
        return self._url

    @property
    def username(self) -> str:
        """Get the Neo4j username."""
        return self._username

    @property
    def password(self) -> str:
        """Get the Neo4j password."""
        return self._password

    async def close(self):
        """Close the Neo4j driver connection."""
        await self.driver.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @property
    def capabilities(self) -> set[StorageCapability]:
        """Neo4j supports graph traversal and entity operations."""
        return {
            StorageCapability.SEARCH,
            StorageCapability.GRAPH_TRAVERSAL
        }

    async def _add_internal(self, artifact: EntityArtifact) -> Tuple[str, bool]:
        """Add entity node to graph with session isolation.

        CRITICAL: Multiple sessions can reference the same entity (e.g., "Apple")
        but each session must be tracked separately for cleanup and rollback.

        Session isolation behavior:
        - ON CREATE: Initialize session_ids list with current session_id
        - ON MATCH: ADD current session_id to session_ids list (don't overwrite!)
        - This enables rollback: "DELETE all entities with session_id = X"

        Returns:
            Tuple of (artifact_id, is_new)
        """
        # Extract worker_id from metadata
        worker_id = artifact.metadata.get("worker_id", "unknown")

        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.first_seen = datetime(),
            e.types = [$type],
            e.descriptions = [$description],
            e.workers = [$worker_id],
            e.session_ids = [$session_id],
            e.relevance_score = $relevance
        ON MATCH SET
            e.types = CASE WHEN $type IN e.types THEN e.types ELSE e.types + $type END,
            e.descriptions = CASE WHEN $description IN e.descriptions THEN e.descriptions ELSE e.descriptions + $description END,
            e.workers = CASE WHEN $worker_id IN e.workers THEN e.workers ELSE e.workers + $worker_id END,
            e.relevance_score = CASE WHEN e.relevance_score < $relevance THEN $relevance ELSE e.relevance_score END,
            e.session_ids = CASE WHEN $session_id IN e.session_ids THEN e.session_ids ELSE e.session_ids + $session_id END
        RETURN e.name as entity_name, size(e.types) as type_count, size(e.workers) as worker_count, size(e.session_ids) as session_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                {
                    "name": artifact.entity_name,
                    "type": artifact.entity_type,
                    "description": artifact.content[:1000],  # Truncate long descriptions
                    "worker_id": worker_id,
                    "session_id": artifact.session_id,
                    "relevance": artifact.relevance_score,
                    "source_id": artifact.id
                }
            )
            record = await result.single()
            entity_name = record["entity_name"]
            type_count = record["type_count"]
            worker_count = record["worker_count"]
            session_count = record["session_count"]

        logger.debug(f"[NEO4J] Entity '{entity_name}' now has {type_count} type(s), {worker_count} worker(s), {session_count} session(s)")

        return (artifact.id, True)

    async def _get_internal(self, artifact_id: str) -> Optional[BaseArtifact]:
        """Neo4j doesn't support direct get by artifact_id."""
        return None

    async def _search_internal(
        self,
        query: str,
        k: int,
        session_id: Optional[str],
        **filters
    ) -> List[BaseArtifact]:
        """Graph traversal search with session filtering.

        CRITICAL: Only returns entities from the specified session_id.
        If session_id is None, returns entities from ALL sessions (global view).

        For entity_name search, use filters['entity_name']
        For max_depth, use filters['max_depth']
        """
        entity_name = filters.get("entity_name", query)
        max_depth = filters.get("max_depth", 2)

        paths = await self.traverse(entity_name, max_depth, session_id)

        # Convert paths to artifacts
        artifacts = []
        for path_data in paths:
            for node in path_data.get("path", []):
                artifacts.append(BaseArtifact(
                    id=f"entity_{node['name']}",
                    content=f"Entity: {node['name']} (Type: {node['type']})",
                    artifact_type=ArtifactType.ENTITY,
                    session_id=session_id or "",
                    metadata={
                        "entity_name": node['name'],
                        "entity_type": node['type']
                    }
                ))

        return artifacts[:k]

    async def traverse(
        self,
        entity_name: str,
        max_depth: int = 2,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Traverse relationships from an entity with session filtering.

        This is how OSINT mode discovers "breadcrumbs" - showing the
        frontier of the investigation graph.

        CRITICAL: When session_id is provided, only returns entities
        that belong to that session. This prevents data leakage between
        different research runs.

        Args:
            entity_name: Starting entity name
            max_depth: Maximum traversal depth (1-3 recommended)
            session_id: Optional session filter (None = all sessions)

        Returns:
            List of path dictionaries
        """
        # Build query with optional session_id filter
        if session_id:
            query = f"""
            MATCH path = (start:Entity {{name: $name}})-[*1..{max_depth}]-(connected)
            WHERE $session_id IN start.session_ids AND $session_id IN connected.session_ids
            RETURN [n in nodes(path) | {{name: n.name, type: n.types[0]}}] as path
            LIMIT 50
            """
            params = {"name": entity_name, "session_id": session_id}
        else:
            query = f"""
            MATCH path = (start:Entity {{name: $name}})-[*1..{max_depth}]-(connected)
            RETURN [n in nodes(path) | {{name: n.name, type: n.types[0]}}] as path
            LIMIT 50
            """
            params = {"name": entity_name}

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, params)
            records = await result.data()

        return records

    async def add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Add relationship between two entities with concurrent worker support.

        Handles multiple workers creating the same relationship:
        - Merges on (from, to, type) combination
        - Accumulates metadata from multiple workers
        - Tracks which workers contributed

        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relationship_type: Type of relationship
            metadata: Additional relationship data

        Returns:
            Relationship identifier
        """
        # Extract worker_id from metadata
        worker_id = metadata.get("worker_id", metadata.get("extraction_method", "unknown"))

        # Sanitize relationship type for Cypher
        rel_type = relationship_type.replace("-", "_").replace(" ", "_").upper()

        query = f"""
        MATCH (from:Entity {{name: $from_name}})
        MATCH (to:Entity {{name: $to_name}})
        MERGE (from)-[r:{rel_type}]->(to)
        ON CREATE SET
            r.first_seen = datetime(),
            r.workers = [$worker_id],
            r.confidence = $confidence,
            r.description = $description
        ON MATCH SET
            r.workers = CASE WHEN $worker_id IN r.workers THEN r.workers ELSE r.workers + $worker_id END,
            r.confidence = CASE WHEN r.confidence < $confidence THEN $confidence ELSE r.confidence END,
            r.description = CASE WHEN r.description IS NULL OR size(r.description) < size($description) THEN $description ELSE r.description END
        RETURN from.name as from_name, type(r) as rel_type, to.name as to_name, size(r.workers) as worker_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                {
                    "from_name": from_entity,
                    "to_name": to_entity,
                    "worker_id": worker_id,
                    "confidence": metadata.get("confidence", 0.8),
                    "description": str(metadata.get("description", ""))[:500]
                }
            )
            record = await result.single()
            if record:
                worker_count = record["worker_count"]
                from_name = record["from_name"]
                rel_type = record["rel_type"]
                to_name = record["to_name"]
                logger.debug(f"[NEO4J] Relationship '{from_name}-{rel_type}->{to_name}' now has {worker_count} worker(s)")

        return f"{from_entity}_{rel_type}_{to_entity}"

    async def health_check(self) -> HealthCheckResult:
        """Neo4j-specific health check."""
        import time
        start = time.time()

        try:
            # Test query
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                if record:
                    record["test"]

            latency_ms = (time.time() - start) * 1000

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"Neo4j connected, latency: {latency_ms:.1f}ms",
                latency_ms=latency_ms,
                metadata={"uri": self._url}
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Neo4j health check failed: {str(e)}"
            )
