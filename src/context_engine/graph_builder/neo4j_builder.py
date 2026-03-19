"""NEO4j Graph Builder - Extracts entities and relationships from LangGraph Store.

Reads from store and builds a knowledge graph in Neo4j.
This is an OPTIONAL post-processing step, not part of the main pipeline.

Usage:
    uv run context-engine build-graph --session-id abc123
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from context_engine.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, get_model
from context_engine.store.setup import get_store


class EntityType(str, Enum):
    """Types of entities that can be extracted."""

    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    CONCEPT = "Concept"
    TECHNOLOGY = "Technology"
    EVENT = "Event"
    TOPIC = "Topic"


class RelationType(str, Enum):
    """Types of relationships between entities."""

    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    CREATED_BY = "CREATED_BY"
    LOCATED_IN = "LOCATED_IN"
    MENTIONS = "MENTIONS"
    INFLUENCES = "INFLUENCES"
    OCCURRED_AT = "OCCURRED_AT"
    USES = "USES"


@dataclass
class Entity:
    """Extracted entity."""

    name: str
    entity_type: EntityType
    description: str = ""
    source_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_neo4j_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "description": self.description,
            "source_ids": json.dumps(self.source_ids),
            "confidence": self.confidence,
            "metadata": json.dumps(self.metadata),
            "created_at": datetime.utcnow().isoformat(),
        }


@dataclass
class Relationship:
    """Relationship between two entities."""

    source_entity: str  # Entity name
    target_entity: str  # Entity name
    relation_type: RelationType
    context: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_neo4j_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relation_type.value,
            "context": self.context,
            "confidence": self.confidence,
            "metadata": json.dumps(self.metadata),
            "created_at": datetime.utcnow().isoformat(),
        }


# Prompts for entity extraction
EXTRACTION_PROMPT = """You are an entity extraction expert. Analyze the following text and extract entities and relationships.

TEXT TO ANALYZE:
{text}

Extract:
1. ENTITIES - Important people, organizations, locations, concepts, technologies, events, or topics
2. RELATIONSHIPS - How these entities relate to each other

Return JSON in this exact format:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Person|Organization|Location|Concept|Technology|Event|Topic",
      "description": "Brief description",
      "confidence": 0.0-1.0
    }}
  ],
  "relationships": [
    {{
      "source": "Entity A name",
      "target": "Entity B name",
      "relation": "RELATED_TO|PART_OF|CREATED_BY|LOCATED_IN|MENTIONS|INFLUENCES|OCCURRED_AT|USES",
      "context": "How they are related",
      "confidence": 0.0-1.0
    }}
  ]
}}

IMPORTANT:
- Only extract entities that are clearly mentioned in the text
- Use confidence scores: 1.0 for explicit mentions, 0.7-0.9 for implied, lower for uncertain
- Keep descriptions concise (1-2 sentences)
- Use consistent entity names (don't create duplicates)

JSON OUTPUT:"""


class Neo4jBuilder:
    """Builds knowledge graph in Neo4j from LangGraph Store content.

    This reads all memories from a session and extracts entities/relationships.
    """

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
    ) -> None:
        """Initialize Neo4j builder.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.uri = neo4j_uri or NEO4J_URI
        self.user = neo4j_user or NEO4J_USER
        self.password = neo4j_password or NEO4J_PASSWORD

        self._driver = None
        self._model = None

        logger.info(f"Neo4j builder initialized (uri: {self.uri})")

    @property
    def driver(self):
        """Lazy-load Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                )
                logger.info("Neo4j driver connected")
            except ImportError:
                raise ImportError(
                    "neo4j package not installed. Install with: uv add neo4j"
                )
        return self._driver

    @property
    def model(self):
        """Lazy-load model for extraction."""
        if self._model is None:
            self._model = get_model("smart")  # Use Sonnet for extraction
        return self._model

    def is_available(self) -> bool:
        """Check if Neo4j connection is available."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1").single()
            return True
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            return False

    async def build_graph(
        self,
        session_id: str,
        namespace_prefix: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Build knowledge graph from session's stored content.

        Args:
            session_id: Session ID to read from
            namespace_prefix: Optional namespace prefix (default: (session_id, "memories"))

        Returns:
            Dict with stats: entities_created, relationships_created, errors
        """
        logger.info(f"Building graph for session: {session_id}")

        # Get store
        store = await get_store()

        # Default namespace for memories
        if namespace_prefix is None:
            namespace_prefix = (session_id, "memories")

        # Get all items from store
        items = await self._get_all_items(store, namespace_prefix)

        if not items:
            logger.warning(f"No items found in {namespace_prefix}")
            return {
                "entities_created": 0,
                "relationships_created": 0,
                "errors": ["No items found in store"],
            }

        logger.info(f"Found {len(items)} items to process")

        # Extract entities and relationships
        all_entities: dict[str, Entity] = {}
        all_relationships: list[Relationship] = []
        errors: list[str] = []

        for key, item in items.items():
            content = item.get("content", "")
            if not content:
                continue

            try:
                entities, relationships = await self._extract_from_text(
                    content, source_id=key
                )

                # Merge entities (avoid duplicates)
                for entity in entities:
                    if entity.name not in all_entities:
                        all_entities[entity.name] = entity
                    else:
                        # Merge source IDs
                        all_entities[entity.name].source_ids.extend(entity.source_ids)

                all_relationships.extend(relationships)

            except Exception as e:
                error_msg = f"Error extracting from {key}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Store in Neo4j
        entities_created = 0
        relationships_created = 0

        for entity in all_entities.values():
            try:
                self._create_entity(session_id, entity)
                entities_created += 1
            except Exception as e:
                errors.append(f"Failed to create entity {entity.name}: {e}")

        for rel in all_relationships:
            try:
                self._create_relationship(session_id, rel)
                relationships_created += 1
            except Exception as e:
                errors.append(f"Failed to create relationship: {e}")

        result = {
            "entities_created": entities_created,
            "relationships_created": relationships_created,
            "total_items_processed": len(items),
            "errors": errors,
        }

        logger.info(
            f"Graph built: {entities_created} entities, {relationships_created} relationships"
        )

        return result

    async def _get_all_items(
        self,
        store: Any,
        namespace_prefix: tuple[str, ...],
    ) -> dict[str, Any]:
        """Get all items from store with given namespace prefix.

        Args:
            store: LangGraph store
            namespace_prefix: Namespace prefix to search

        Returns:
            Dict of key -> item value
        """
        items = {}

        # Try to list items in namespace
        try:
            # LangGraph Store has asearch for semantic search and alist for listing
            # We use semantic search with empty query to get all
            results = await store.asearch(namespace_prefix, query="", limit=1000)
            for result in results:
                items[result.key] = result.value
        except Exception as e:
            logger.warning(f"Could not search store: {e}, trying alist")
            try:
                # Fallback to alist if available
                results = await store.alist(namespace_prefix)
                for result in results:
                    items[result.key] = result.value
            except Exception as e2:
                logger.error(f"Failed to get items from store: {e2}")

        return items

    async def _extract_from_text(
        self,
        text: str,
        source_id: str,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from text using LLM.

        Args:
            text: Text to extract from
            source_id: Source identifier

        Returns:
            Tuple of (entities, relationships)
        """
        # Truncate text if too long
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"

        prompt = EXTRACTION_PROMPT.format(text=text)

        response = await self.model.ainvoke(prompt)

        # Parse JSON response
        content = response.content

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            logger.warning(f"No JSON found in extraction response")
            return [], []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction JSON: {e}")
            return [], []

        # Parse entities
        entities = []
        for e in data.get("entities", []):
            try:
                entity_type = EntityType(e.get("type", "Concept"))
            except ValueError:
                entity_type = EntityType.CONCEPT

            entity = Entity(
                name=e.get("name", ""),
                entity_type=entity_type,
                description=e.get("description", ""),
                source_ids=[source_id],
                confidence=e.get("confidence", 1.0),
            )
            entities.append(entity)

        # Parse relationships
        relationships = []
        for r in data.get("relationships", []):
            try:
                relation_type = RelationType(r.get("relation", "RELATED_TO"))
            except ValueError:
                relation_type = RelationType.RELATED_TO

            rel = Relationship(
                source_entity=r.get("source", ""),
                target_entity=r.get("target", ""),
                relation_type=relation_type,
                context=r.get("context", ""),
                confidence=r.get("confidence", 1.0),
            )
            relationships.append(rel)

        return entities, relationships

    def _create_entity(self, session_id: str, entity: Entity) -> None:
        """Create entity node in Neo4j.

        Args:
            session_id: Session ID for namespacing
            entity: Entity to create
        """
        query = """
        MERGE (e:Entity {name: $name, session_id: $session_id})
        SET e.type = $type,
            e.description = $description,
            e.source_ids = $source_ids,
            e.confidence = $confidence,
            e.metadata = $metadata,
            e.created_at = $created_at
        """

        data = entity.to_neo4j_dict()
        data["session_id"] = session_id

        with self.driver.session() as session:
            session.run(
                query,
                name=entity.name,
                session_id=session_id,
                type=data["type"],
                description=data["description"],
                source_ids=data["source_ids"],
                confidence=data["confidence"],
                metadata=data["metadata"],
                created_at=data["created_at"],
            )

    def _create_relationship(self, session_id: str, relationship: Relationship) -> None:
        """Create relationship in Neo4j.

        Args:
            session_id: Session ID for namespacing
            relationship: Relationship to create
        """
        # Dynamic relationship type
        rel_type = relationship.relation_type.value

        query = f"""
        MATCH (source:Entity {{name: $source_name, session_id: $session_id}})
        MATCH (target:Entity {{name: $target_name, session_id: $session_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.context = $context,
            r.confidence = $confidence,
            r.metadata = $metadata,
            r.created_at = $created_at
        """

        data = relationship.to_neo4j_dict()

        with self.driver.session() as session:
            session.run(
                query,
                source_name=relationship.source_entity,
                target_name=relationship.target_entity,
                session_id=session_id,
                context=data["context"],
                confidence=data["confidence"],
                metadata=data["metadata"],
                created_at=data["created_at"],
            )

    def query_graph(
        self,
        session_id: str,
        entity_name: str | None = None,
        entity_type: EntityType | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query the knowledge graph.

        Args:
            session_id: Session ID
            entity_name: Filter by entity name (optional)
            entity_type: Filter by entity type (optional)
            limit: Max results

        Returns:
            List of entity dicts with relationships
        """
        conditions = ["e.session_id = $session_id"]
        params = {"session_id": session_id, "limit": limit}

        if entity_name:
            conditions.append("e.name CONTAINS $entity_name")
            params["entity_name"] = entity_name

        if entity_type:
            conditions.append("e.type = $entity_type")
            params["entity_type"] = entity_type.value

        query = f"""
        MATCH (e:Entity)
        WHERE {' AND '.join(conditions)}
        OPTIONAL MATCH (e)-[r]->(target:Entity)
        RETURN e.name as name, e.type as type, e.description as description,
               collect({{target: target.name, relation: type(r), context: r.context}}) as relationships
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def clear_session(self, session_id: str) -> int:
        """Clear all entities for a session.

        Args:
            session_id: Session ID to clear

        Returns:
            Number of nodes deleted
        """
        query = """
        MATCH (e:Entity {session_id: $session_id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """

        with self.driver.session() as session:
            result = session.run(query, session_id=session_id)
            record = result.single()
            return record["deleted"] if record else 0

    def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")


# Convenience function for CLI
async def build_knowledge_graph(session_id: str) -> dict[str, Any]:
    """Build knowledge graph for a session.

    Args:
        session_id: Session ID to process

    Returns:
        Build result stats
    """
    builder = Neo4jBuilder()

    if not builder.is_available():
        return {
            "entities_created": 0,
            "relationships_created": 0,
            "errors": ["Neo4j connection not available"],
        }

    try:
        result = await builder.build_graph(session_id)
        return result
    finally:
        builder.close()
