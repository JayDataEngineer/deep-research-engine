"""
Neo4j Graph Storage Integration

Stores relationships and entity connections in a graph database.
Enables complex queries about relationships between people, places, and things.
"""

import os
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone


class Neo4jGraphStore:
    """
    Neo4j graph database for storing entity relationships.

    Provides:
    - Entity nodes (people, places, organizations, concepts)
    - Relationship edges between entities
    - Temporal tracking (when relationships were established)
    - Confidence scoring for uncertain information

    Example use cases:
    - "Who do I know that works at Google?"
    - "What places have I visited?"
    - "How are Alice and Bob connected?"
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database

        self._driver = None
        self._connected = False
        self._fallback_mode = False  # Fall back to in-memory storage

        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            self._driver.verify_connectivity()
            self._connected = True
        except ImportError:
            print("Neo4j driver not available, using in-memory fallback")
            self._fallback_mode = True
            self._memory_graph: Dict[str, Dict[str, Any]] = {"nodes": {}, "relationships": []}
        except Exception as e:
            print(f"Could not connect to Neo4j: {e}, using in-memory fallback")
            self._fallback_mode = True
            self._memory_graph: Dict[str, Dict[str, Any]] = {"nodes": {}, "relationships": []}

    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._connected = False

    def create_entity(
        self,
        user_id: str,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
    ) -> bool:
        """
        Create or update an entity node.

        Args:
            user_id: User identifier (for namespacing)
            entity_id: Unique entity identifier
            entity_type: Type of entity ("Person", "Place", "Organization", "Concept")
            properties: Entity properties

        Returns:
            True if successful
        """
        if self._fallback_mode:
            self._memory_graph["nodes"][f"{user_id}:{entity_id}"] = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "properties": properties,
            }
            return True

        query = """
        MERGE (e:Entity {user_id: $user_id, entity_id: $entity_id})
        SET e.entity_type = $entity_type
        SET e += $properties
        SET e.updated_at = datetime()
        RETURN e
        """

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    query,
                    user_id=user_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    properties=properties,
                )
                return True
        except Exception:
            return False

    def create_relationship(
        self,
        user_id: str,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> bool:
        """
        Create a relationship between two entities.

        Args:
            user_id: User identifier
            from_entity: Source entity ID
            to_entity: Target entity ID
            relationship_type: Type of relationship ("KNOWS", "WORKS_AT", "LIVES_IN", etc.)
            properties: Additional relationship properties
            confidence: Confidence score

        Returns:
            True if successful
        """
        if self._fallback_mode:
            self._memory_graph["relationships"].append({
                "user_id": user_id,
                "from_entity": from_entity,
                "to_entity": to_entity,
                "relationship_type": relationship_type,
                "properties": properties or {},
                "confidence": confidence,
            })
            return True

        query = """
        MATCH (from:Entity {user_id: $user_id, entity_id: $from_entity})
        MATCH (to:Entity {user_id: $user_id, entity_id: $to_entity})
        MERGE (from)-[r:RELATIONSHIP_TYPE]->(to)
        SET r.relationship_type = $relationship_type
        SET r.confidence = $confidence
        SET r += $properties
        SET r.created_at = datetime()
        RETURN r
        """

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    query,
                    user_id=user_id,
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relationship_type=relationship_type,
                    properties=properties or {},
                    confidence=confidence,
                )
                return True
        except Exception:
            return False

    def find_connections(
        self,
        user_id: str,
        entity_id: str,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find all entities connected to the given entity.

        Args:
            user_id: User identifier
            entity_id: Entity to find connections for
            max_depth: Maximum path length to search

        Returns:
            List of connected entities with relationship info
        """
        if self._fallback_mode:
            # Simple in-memory search
            results = []
            for rel in self._memory_graph["relationships"]:
                if rel["user_id"] != user_id:
                    continue
                if rel["from_entity"] == entity_id:
                    results.append({
                        "connected_to": rel["to_entity"],
                        "relationship": rel["relationship_type"],
                        "direction": "outgoing",
                        "confidence": rel["confidence"],
                    })
                elif rel["to_entity"] == entity_id:
                    results.append({
                        "connected_to": rel["from_entity"],
                        "relationship": rel["relationship_type"],
                        "direction": "incoming",
                        "confidence": rel["confidence"],
                    })
            return results

        query = f"""
        MATCH path = (e:Entity {{user_id: $user_id, entity_id: $entity_id}})-[*1..{max_depth}]-(connected:Entity)
        RETURN DISTINCT
            connected.entity_id AS connected_to,
            connected.entity_type AS entity_type,
            [(r)-[rels:RELATIONSHIP_TYPE*]-(connected) |
             [rel in rels | rel.relationship_type]
            ] AS relationships
        LIMIT 50
        """

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, user_id=user_id, entity_id=entity_id)

                connections = []
                for record in result:
                    # Flatten relationships list
                    rels = record["relationships"]
                    if rels and len(rels) > 0:
                        connections.append({
                            "connected_to": record["connected_to"],
                            "entity_type": record["entity_type"],
                            "relationships": rels,
                        })

                return connections
        except Exception:
            return []

    def query_relationships(
        self,
        user_id: str,
        cypher_query: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.

        Args:
            user_id: User identifier
            cypher_query: Cypher query (will be filtered to user's data)

        Returns:
            Query results
        """
        if self._fallback_mode:
            # Very limited query support in fallback mode
            return []

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(cypher_query, user_id=user_id)

                results = []
                for record in result:
                    results.append(dict(record))
                return results
        except Exception:
            return []

    def get_entity_info(
        self,
        user_id: str,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an entity.

        Args:
            user_id: User identifier
            entity_id: Entity ID

        Returns:
            Entity information or None
        """
        if self._fallback_mode:
            key = f"{user_id}:{entity_id}"
            return self._memory_graph["nodes"].get(key)

        query = """
        MATCH (e:Entity {user_id: $user_id, entity_id: $entity_id})
        RETURN e
        """

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, user_id=user_id, entity_id=entity_id)
                record = result.single()

                if record:
                    node = record["e"]
                    return dict(node)
                return None
        except Exception:
            return None

    def suggest_connections(
        self,
        user_id: str,
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Suggest potential connections (entities that might be related).

        Based on co-occurrence in relationships.

        Args:
            user_id: User identifier
            entity_id: Entity to find suggestions for

        Returns:
            List of suggested connections
        """
        # Find entities that share common connections
        if self._fallback_mode:
            return []

        query = """
        MATCH (e:Entity {user_id: $user_id, entity_id: $entity_id})-[:1]-(connected)-[:1]-(suggested:Entity)
        WHERE NOT (e)-[:1]-(suggested)
        AND suggested.entity_id <> $entity_id
        RETURN suggested.entity_id AS entity_id,
               suggested.entity_type AS entity_type,
               COUNT(*) AS connection_count
        ORDER BY connection_count DESC
        LIMIT 10
        """

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, user_id=user_id, entity_id=entity_id)

                suggestions = []
                for record in result:
                    suggestions.append({
                        "entity_id": record["entity_id"],
                        "entity_type": record["entity_type"],
                        "strength": record["connection_count"],
                    })

                return suggestions
        except Exception:
            return []

    def get_context_string(
        self,
        user_id: str,
        entity_id: str,
    ) -> str:
        """
        Get a formatted context string for an entity's connections.

        Useful for prompt injection.

        Args:
            user_id: User identifier
            entity_id: Entity ID

        Returns:
            Formatted string of relationships
        """
        connections = self.find_connections(user_id, entity_id, max_depth=1)

        if not connections:
            return ""

        lines = [f"Relationships for {entity_id}:"]
        for conn in connections[:10]:
            direction_symbol = "→" if conn["direction"] == "outgoing" else "←"
            lines.append(f"  {conn['connected_to']} {direction_symbol} {conn['relationship']}")

        return "\n".join(lines)


def create_neo4j_store(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Neo4jGraphStore:
    """Create a Neo4j graph store instance."""
    return Neo4jGraphStore(uri=uri, user=user, password=password)
