"""Neo4j storage backend for knowledge graph."""

import uuid
from datetime import datetime
from typing import Any

from ..base import StorageBackend, StorageResult
from ..errors import StorageInitializationError
from ..registry import StorageRegistry


@StorageRegistry.register("neo4j")
class Neo4jStorage(StorageBackend):
    """Neo4j knowledge graph storage backend.

    Features:
    - Graph-based data storage with nodes and relationships
    - Cypher query support
    - Automatic indexing
    - Knowledge graph construction

    Configuration:
    ```yaml
    storage:
      neo4j:
        uri: "bolt://localhost:7687"
        username: "neo4j"
        password: "your-password"
        database: "neo4j"
    ```

    Data Model:
    - Nodes: Generic "Item" nodes with labels for types
    - Relationships: Generic "RELATED_TO" with types for semantics
    - Properties: All stored as node/relationship properties
    """

    backend_name = "neo4j"
    description = "Neo4j knowledge graph for relationships and graph data"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.uri = config.get("uri", "bolt://localhost:7687")
        self.username = config.get("username", "neo4j")
        self.password = config.get("password", "password")
        self.database = config.get("database", "neo4j")
        self.driver = None

    async def initialize(self) -> None:
        """Initialize Neo4j driver and create constraints."""
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as e:
            raise StorageInitializationError(
                f"Missing dependencies for Neo4j: {e}. Install with: pip install neo4j"
            )

        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=50,
            )

            # Verify connection
            await self._verify_connection()

            # Create constraints and indexes
            await self._create_constraints()

            self.initialized = True
            self.health_status = "healthy"
            self._log(f"Initialized Neo4j storage at {self.uri}")

        except Exception as e:
            self.health_status = "unhealthy"
            raise StorageInitializationError(f"Failed to initialize Neo4j: {e}")

    async def _verify_connection(self) -> None:
        """Verify Neo4j connection."""
        async with self.driver.session(database=self.database) as session:
            await session.run("RETURN 1")

    async def _create_constraints(self) -> None:
        """Create unique constraints and indexes."""
        async with self.driver.session(database=self.database) as session:
            # Unique constraint on Item ID
            await session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (i:Item)
                REQUIRE i.id IS UNIQUE
            """)

            # Index on namespace
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (i:Item)
                ON (i.namespace)
            """)

            # Index on labels
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (i:Item)
                ON (i.labels)
            """)

            # Index on created_at
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (i:Item)
                ON (i.created_at)
            """)

            self._log("Constraints and indexes created")

    async def cleanup(self) -> None:
        """Close Neo4j driver."""
        if self.driver:
            await self.driver.close()
            self.initialized = False
            self._log("Neo4j driver closed")

    async def health_check(self) -> bool:
        """Check Neo4j connection health."""
        if not self.driver or not self.initialized:
            return False

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            self.health_status = "unhealthy"
            return False

    async def store(
        self,
        data: dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        """Store data as a node in Neo4j.

        Data can include:
        - id: Unique identifier (auto-generated if not provided)
        - labels: List of additional labels (e.g., ["Document", "Entity"])
        - relationships: List of relationships to create
        - properties: All other fields become node properties
        """
        if not self.initialized:
            return StorageResult(
                success=False,
                error="Neo4j not initialized",
            )

        try:
            resource_id = data.get("id", str(uuid.uuid4()))
            labels = data.get("labels", ["Item"])
            relationships = data.pop("relationships", [])
            properties = {k: v for k, v in data.items() if k not in ["id", "labels"]}

            # Ensure standard properties
            properties["namespace"] = namespace
            properties["created_at"] = properties.get("created_at", datetime.now().isoformat())
            properties["updated_at"] = datetime.now().isoformat()

            async with self.driver.session(database=self.database) as session:
                # Build labels string
                labels_str = ":".join(["Item", *labels])

                # Create or merge node
                await session.run(
                    f"""
                    MERGE (n:{labels_str} {{id: $id, namespace: $namespace}})
                    SET n += $properties
                """,
                    {
                        "id": resource_id,
                        "namespace": namespace,
                        "properties": properties,
                    },
                )

                # Create relationships if provided
                for rel in relationships:
                    target_id = rel.get("target_id")
                    rel_type = rel.get("type", "RELATED_TO")
                    rel_properties = rel.get("properties", {})

                    if target_id:
                        await session.run(
                            f"""
                            MATCH (source {{id: $source_id, namespace: $namespace}})
                            MATCH (target {{id: $target_id, namespace: $namespace}})
                            MERGE (source)-[r:{rel_type}]->(target)
                            SET r += $properties
                        """,
                            {
                                "source_id": resource_id,
                                "target_id": target_id,
                                "namespace": namespace,
                                "properties": rel_properties,
                            },
                        )

            self._log(f"Stored node {resource_id} with {len(relationships)} relationships")

            return StorageResult(
                success=True,
                id=resource_id,
                metadata={
                    "namespace": namespace,
                    "labels": labels,
                    "relationships": len(relationships),
                },
            )

        except Exception as e:
            self._log(f"Store failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve node by ID with optional relationships."""
        if not self.initialized:
            return None

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    """
                    MATCH (n:Item {id: $id, namespace: $namespace})
                    RETURN n
                """,
                    {"id": id, "namespace": namespace},
                )

                record = await result.single()
                if not record:
                    return None

                node = record["n"]
                data = dict(node)
                data["labels"] = list(node.labels)

                return data

        except Exception as e:
            self._log(f"Retrieve failed: {e}", level="error")
            return None

    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        """Delete node by ID (detaches delete to remove relationships)."""
        if not self.initialized:
            return False

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(
                    """
                    MATCH (n:Item {id: $id, namespace: $namespace})
                    DETACH DELETE n
                """,
                    {"id": id, "namespace": namespace},
                )

            self._log(f"Deleted node {id} from namespace {namespace}")
            return True

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
        """Execute Cypher query or search by text.

        Args:
            query: Can be:
                - A Cypher query string (if cypher=True)
                - A text search term (searches in node properties)
            namespace: Namespace to query
            limit: Maximum results
            **kwargs: Additional options:
                - cypher: If True, treat query as raw Cypher
                - labels: Filter by specific labels
                - relationships: Include relationships in results
        """
        if not self.initialized:
            return []

        cypher_query = kwargs.get("cypher", False)
        include_relationships = kwargs.get("relationships", False)
        labels = kwargs.get("labels", [])

        try:
            async with self.driver.session(database=self.database) as session:
                if cypher_query:
                    # Execute raw Cypher query
                    result = await session.run(query, {"limit": limit})
                    return [dict(record) for record in await result.fetch(limit)]

                else:
                    # Text search
                    label_filter = ""
                    if labels:
                        label_str = ":".join(labels)
                        label_filter = f"(n:Item:{label_str} {{namespace: $namespace}})"
                    else:
                        label_filter = "(n:Item {namespace: $namespace})"

                    # Search in all properties
                    result = await session.run(
                        f"""
                        MATCH {label_filter}
                        WHERE ANY(k IN keys(n) WHERE
                            toString(n[k]) CONTAINS $query OR
                            n[k] CONTAINS $query
                        )
                        RETURN n
                        LIMIT $limit
                    """,
                        {"namespace": namespace, "query": query, "limit": limit},
                    )

                    nodes = []
                    async for record in result:
                        node = record["n"]
                        data = dict(node)
                        data["labels"] = list(node.labels)

                        if include_relationships:
                            # Get relationships
                            rels = await session.run(
                                """
                                MATCH (n)-[r]->(m)
                                WHERE n.id = $id AND n.namespace = $namespace
                                RETURN type(r) as type, m.id as target, properties(r) as props
                            """,
                                {"id": data["id"], "namespace": namespace},
                            )

                            data["relationships"] = [
                                {
                                    "type": rel["type"],
                                    "target": rel["target"],
                                    "properties": dict(rel["props"]),
                                }
                                async for rel in rels
                            ]

                        nodes.append(data)

                    return nodes

        except Exception as e:
            self._log(f"Query failed: {e}", level="error")
            return []

    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List items in namespace."""
        if not self.initialized:
            return []

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    """
                    MATCH (n:Item {namespace: $namespace})
                    RETURN n.id as id, n.labels as labels, n.created_at as created_at
                    ORDER BY n.created_at DESC
                    LIMIT $limit
                """,
                    {"namespace": namespace, "limit": limit},
                )

                return [
                    {
                        "id": record["id"],
                        "labels": list(record["labels"]) if record["labels"] else ["Item"],
                        "created_at": record["created_at"],
                    }
                    async for record in result
                ]

        except Exception as e:
            self._log(f"List failed: {e}", level="error")
            return []

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        namespace: str = "default",
        properties: dict[str, Any] | None = None,
    ) -> StorageResult:
        """Create a relationship between two nodes."""
        if not self.initialized:
            return StorageResult(success=False, error="Neo4j not initialized")

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(
                    f"""
                    MATCH (source:Item {{id: $source_id, namespace: $namespace}})
                    MATCH (target:Item {{id: $target_id, namespace: $namespace}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r += $properties
                """,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "namespace": namespace,
                        "properties": properties or {},
                    },
                )

            self._log(f"Created relationship {rel_type} from {source_id} to {target_id}")

            return StorageResult(
                success=True,
                id=f"{source_id}-{rel_type}-{target_id}",
                metadata={"type": rel_type, "source": source_id, "target": target_id},
            )

        except Exception as e:
            self._log(f"Create relationship failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        namespace: str = "default",
        max_depth: int = 5,
    ) -> list[dict[str, Any]] | None:
        """Find path between two nodes."""
        if not self.initialized:
            return None

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    """
                    MATCH path = shortestPath(
                        (start:Item {id: $start_id, namespace: $namespace})
                        -[*..$max_depth]-
                        (end:Item {id: $end_id, namespace: $namespace})
                    )
                    RETURN path
                """,
                    {
                        "start_id": start_id,
                        "end_id": end_id,
                        "namespace": namespace,
                        "max_depth": max_depth,
                    },
                )

                record = await result.single()
                if not record:
                    return None

                path = record["path"]
                return [dict(node) for node in path.nodes]

        except Exception as e:
            self._log(f"Find path failed: {e}", level="error")
            return None

    async def get_neighbors(
        self,
        node_id: str,
        namespace: str = "default",
        direction: str = "both",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes."""
        if not self.initialized:
            return []

        try:
            async with self.driver.session(database=self.database) as session:
                if direction == "outgoing":
                    pattern = "(n)-[r]->(m)"
                elif direction == "incoming":
                    pattern = "(n)<-[r]-(m)"
                else:
                    pattern = "(n)-[r]-(m)"

                result = await session.run(
                    f"""
                    MATCH {pattern}
                    WHERE n.id = $id AND n.namespace = $namespace
                    RETURN m, type(r) as relationship_type, r as rel
                    LIMIT $limit
                """,
                    {"id": node_id, "namespace": namespace, "limit": limit},
                )

                neighbors = []
                async for record in result:
                    node = record["m"]
                    data = dict(node)
                    data["labels"] = list(node.labels)
                    data["relationship_type"] = record["relationship_type"]
                    data["relationship_properties"] = dict(record["rel"])
                    neighbors.append(data)

                return neighbors

        except Exception as e:
            self._log(f"Get neighbors failed: {e}", level="error")
            return []
