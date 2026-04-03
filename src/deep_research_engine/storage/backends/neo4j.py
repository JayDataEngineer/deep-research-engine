"""Neo4j storage backend for knowledge graph."""

import re
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

    # =========================================================================
    # Knowledge Graph Extensions (typed nodes, dynamic schema)
    # =========================================================================

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Sanitize a Neo4j label or relationship type.

        Strips non-[A-Za-z0-9_] chars, limits to 64 chars.
        Must start with a letter.
        """
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", label)[:64]
        if not sanitized:
            return "Unknown"
        # Ensure starts with letter
        if sanitized[0].isdigit():
            sanitized = "N_" + sanitized
        return sanitized

    async def store_typed_node(
        self,
        labels: list[str],
        properties: dict[str, Any],
        merge_on: list[str] | None = None,
        namespace: str = "default",
    ) -> StorageResult:
        """Create/merge a node with arbitrary labels.

        Args:
            labels: Node labels e.g. ["Person", "Entity"].
            properties: Node properties.
            merge_on: Property names to MERGE on (default: ["name"]).
            namespace: Namespace for isolation.
        """
        if not self.initialized:
            return StorageResult(success=False, error="Neo4j not initialized")

        if not labels:
            labels = ["Entity"]
        sanitized_labels = [self._sanitize_label(lbl) for lbl in labels]

        if merge_on is None:
            merge_on = ["name"]

        # Build MERGE clause using merge_on properties
        merge_props = ", ".join(f"{k}: $merge_{k}" for k in merge_on if k in properties)
        if not merge_props:
            # Fallback: merge on first property
            first_key = next(iter(properties), "name")
            merge_on = [first_key]
            merge_props = f"{first_key}: $merge_{first_key}"

        label_str = ":".join(sanitized_labels)
        merge_clause = "{" + merge_props + "}"

        # Build params
        params: dict[str, Any] = {
            "namespace": namespace,
            "properties": {
                **properties,
                "namespace": namespace,
                "updated_at": datetime.now().isoformat(),
            },
        }
        for k in merge_on:
            if k in properties:
                params[f"merge_{k}"] = properties[k]

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(
                    f"MERGE (n:{label_str} {merge_clause}) SET n += $properties",
                    params,
                )
            return StorageResult(success=True, metadata={"labels": sanitized_labels})
        except Exception as e:
            self._log(f"store_typed_node failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def create_typed_relationship(
        self,
        source_labels: list[str],
        source_match: dict[str, Any],
        target_labels: list[str],
        target_match: dict[str, Any],
        rel_type: str,
        properties: dict[str, Any] | None = None,
        namespace: str = "default",
    ) -> StorageResult:
        """Create relationship between typed nodes by property match.

        Args:
            source_labels: Labels for the source node.
            source_match: Properties to match the source node on.
            target_labels: Labels for the target node.
            target_match: Properties to match the target node on.
            rel_type: Relationship type (e.g. "SPOKE_IN").
            properties: Optional relationship properties.
            namespace: Namespace for isolation.
        """
        if not self.initialized:
            return StorageResult(success=False, error="Neo4j not initialized")

        safe_rel = self._sanitize_label(rel_type)
        src_label_str = (
            ":".join(self._sanitize_label(lbl) for lbl in source_labels) if source_labels else ""
        )
        tgt_label_str = (
            ":".join(self._sanitize_label(lbl) for lbl in target_labels) if target_labels else ""
        )

        src_where = " AND ".join(f"s.{k} = $src_{k}" for k in source_match)
        tgt_where = " AND ".join(f"t.{k} = $tgt_{k}" for k in target_match)

        params: dict[str, Any] = {"namespace": namespace, "rel_props": properties or {}}
        for k, v in source_match.items():
            params[f"src_{k}"] = v
        for k, v in target_match.items():
            params[f"tgt_{k}"] = v

        src_label = f":{src_label_str}" if src_label_str else ""
        tgt_label = f":{tgt_label_str}" if tgt_label_str else ""

        cypher = (
            f"MATCH (s{src_label} {{namespace: $namespace}}) WHERE {src_where} "
            f"MATCH (t{tgt_label} {{namespace: $namespace}}) WHERE {tgt_where} "
            f"MERGE (s)-[r:{safe_rel}]->(t) "
            f"SET r += $rel_props"
        )

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(cypher, params)
            return StorageResult(success=True, metadata={"rel_type": safe_rel})
        except Exception as e:
            self._log(f"create_typed_relationship failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def bulk_write(
        self,
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        namespace: str = "default",
    ) -> StorageResult:
        """Batch create nodes + relationships in one transaction using UNWIND.

        Args:
            nodes: List of dicts with keys: labels, properties, merge_on (optional).
            relationships: List of dicts with keys: source_labels, source_match,
                target_labels, target_match, rel_type, properties (optional).
            namespace: Namespace for isolation.
        """
        if not self.initialized:
            return StorageResult(success=False, error="Neo4j not initialized")

        nodes_written = 0
        rels_written = 0

        try:
            async with self.driver.session(database=self.database) as session:
                # Batch write nodes using UNWIND
                if nodes:
                    # Group by label combination for efficient MERGE
                    for node_spec in nodes:
                        labels = [
                            self._sanitize_label(lbl) for lbl in node_spec.get("labels", ["Entity"])
                        ]
                        properties = node_spec.get("properties", {})
                        merge_on = node_spec.get("merge_on", ["name"])
                        label_str = ":".join(labels)

                        merge_props = ", ".join(f"{k}: $n.{k}" for k in merge_on if k in properties)
                        if not merge_props and properties:
                            first_key = next(iter(properties))
                            merge_on = [first_key]
                            merge_props = f"{first_key}: $n.{first_key}"

                        if not merge_props:
                            continue

                        cypher = (
                            f"UNWIND $batch AS n "
                            f"MERGE (node:{label_str} {{{merge_props}}}) "
                            f"SET node += n.props"
                        )
                        # Collect all nodes with same labels
                        batch = []
                        for ns in nodes:
                            ns_labels = [
                                self._sanitize_label(lbl) for lbl in ns.get("labels", ["Entity"])
                            ]
                            if ns_labels == labels:
                                props = ns.get("properties", {})
                                batch.append(
                                    {
                                        **{k: props[k] for k in merge_on if k in props},
                                        "props": {
                                            **props,
                                            "namespace": namespace,
                                            "updated_at": datetime.now().isoformat(),
                                        },
                                    }
                                )

                        if batch:
                            await session.run(cypher, {"batch": batch})
                            nodes_written += len(batch)

                # Write relationships
                for rel_spec in relationships:
                    result = await self.create_typed_relationship(
                        source_labels=rel_spec.get("source_labels", []),
                        source_match=rel_spec.get("source_match", {}),
                        target_labels=rel_spec.get("target_labels", []),
                        target_match=rel_spec.get("target_match", {}),
                        rel_type=rel_spec.get("rel_type", "RELATED_TO"),
                        properties=rel_spec.get("properties"),
                        namespace=namespace,
                    )
                    if result.success:
                        rels_written += 1

            return StorageResult(
                success=True,
                metadata={"nodes_written": nodes_written, "rels_written": rels_written},
            )
        except Exception as e:
            self._log(f"bulk_write failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def execute_cypher(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        namespace: str = "default",
        read_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute parameterized Cypher query safely.

        Args:
            cypher: Cypher query string.
            params: Query parameters (never interpolated into Cypher).
            namespace: Namespace for isolation.
            read_only: If True, prepends EXPLAIN to validate it's a read query.

        Returns:
            List of result records as dicts.
        """
        if not self.initialized:
            return []

        params = params or {}
        params.setdefault("namespace", namespace)

        try:
            async with self.driver.session(database=self.database) as session:
                if read_only:
                    # EXPLAIN a read-only query to validate safety
                    try:
                        await session.run(f"EXPLAIN {cypher}", params)
                    except Exception as e:
                        return [{"error": f"Query validation failed (must be read-only): {e}"}]

                result = await session.run(cypher, params)
                return [dict(record) async for record in result]
        except Exception as e:
            self._log(f"execute_cypher failed: {e}", level="error")
            return [{"error": str(e)}]

    async def get_existing_labels(self) -> list[str]:
        """Get all labels currently in the database."""
        if not self.initialized:
            return []
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("CALL db.labels()")
                return [r["label"] async for r in result]
        except Exception as e:
            self._log(f"get_existing_labels failed: {e}", level="error")
            return []

    async def get_neighbors_by_name(
        self,
        name: str,
        namespace: str = "default",
        depth: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get neighbors of a node matched by name, with configurable depth.

        Args:
            name: Name property of the node to start from.
            namespace: Namespace for isolation.
            depth: Traversal depth (1-4).
            limit: Max results.
        """
        if not self.initialized:
            return []
        depth = max(1, min(depth, 4))
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    f"MATCH (n {{name: $name, namespace: $namespace}})-[r*1..{depth}]-(m) "
                    f"RETURN DISTINCT m, labels(m) as m_labels, "
                    f"[rel in r | type(rel)] as rel_types "
                    f"LIMIT $limit",
                    {"name": name, "namespace": namespace, "limit": limit},
                )
                neighbors = []
                async for record in result:
                    node = record["m"]
                    data = dict(node)
                    data["labels"] = list(record["m_labels"])
                    data["relationship_types"] = record["rel_types"]
                    neighbors.append(data)
                return neighbors
        except Exception as e:
            self._log(f"get_neighbors_by_name failed: {e}", level="error")
            return []

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
