"""Neo4j Graph Builder - Builds the knowledge graph in Neo4j.

Creates the graph structure with session isolation:
- Document node for the session
- Chunk nodes with PART_OF relationship
- Entity nodes with MENTIONED_IN relationship to chunks
- Entity-to-Entity relationships
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import asyncio

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger
from .extractor import GraphDocument

logger = get_logger(__name__)


class Neo4jGraphBuilder:
    """Builds the knowledge graph in Neo4j with session isolation.

    Graph structure:
    - Document: Root node for each research session
    - Chunk: Text chunks from findings (PART_OF → Document)
    - Entity: Extracted entities (MENTIONED_IN ← Chunk)
    - Community: Detected communities (IN_COMMUNITY ← Entity)

    All nodes have session_ids for isolation.
    """

    def __init__(
        self,
        neo4j_driver: Any = None,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "neo4j",
        context: Any = None,
    ):
        """Initialize the graph builder.

        Args:
            neo4j_driver: Existing Neo4j driver (optional)
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            context: ExtensionContext (optional)
        """
        self.driver = neo4j_driver
        self.uri = uri
        self.username = username
        self.password = password
        self.context = context
        self._driver_owned = False

    async def _get_driver(self):
        """Get or create Neo4j driver."""
        if self.driver is not None:
            return self.driver

        try:
            from neo4j import AsyncGraphDatabase

            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self._driver_owned = True
            return self.driver

        except ImportError:
            raise RuntimeError(
                "neo4j package not installed. Install with: pip install neo4j"
            )

    async def build_graph(
        self,
        graph_documents: List[GraphDocument],
        session_id: str,
    ) -> Dict[str, int]:
        """Build the knowledge graph in Neo4j.

        Args:
            graph_documents: List of GraphDocument objects with entities and relationships
            session_id: Research session ID for isolation

        Returns:
            Dict with counts: {chunks: N, entities: N, relationships: N}
        """
        if not graph_documents:
            logger.warning("[GRAPH_BUILDER] No graph documents to build")
            return {"chunks": 0, "entities": 0, "relationships": 0}

        driver = await self._get_driver()

        logger.info(f"[GRAPH_BUILDER] Building graph for session {session_id}")

        async with driver.session() as session:
            # 1. Create Document node
            await self._create_document(session, session_id, len(graph_documents))

            # 2. Create Chunk nodes
            await self._create_chunks(session, graph_documents, session_id)

            # 3. Create Entity nodes
            entity_count = await self._create_entities(session, graph_documents, session_id)

            # 4. Create Entity relationships
            rel_count = await self._create_relationships(session, graph_documents, session_id)

        counts = {
            "chunks": len(graph_documents),
            "entities": entity_count,
            "relationships": rel_count,
        }

        logger.info(f"[GRAPH_BUILDER] Graph built: {counts}")
        return counts

    async def _create_document(
        self,
        session: Any,
        session_id: str,
        chunk_count: int,
    ):
        """Create the Document node for this session."""
        query = """
            MERGE (d:Document {session_id: $session_id})
            SET d.created_at = datetime(),
                d.chunk_count = $chunk_count
        """
        await session.run(query, {
            "session_id": session_id,
            "chunk_count": chunk_count,
        })
        logger.debug(f"[GRAPH_BUILDER] Created Document node for session {session_id}")

    async def _create_chunks(
        self,
        session: Any,
        graph_documents: List[GraphDocument],
        session_id: str,
    ):
        """Create Chunk nodes with PART_OF relationship to Document."""
        chunks_data = []

        for i, doc in enumerate(graph_documents):
            chunk_id = doc.chunk_id or f"chunk_{i}"

            # Get source text if available
            source_text = ""
            if hasattr(doc, 'source_text'):
                source_text = doc.source_text

            chunks_data.append({
                "id": chunk_id,
                "position": i,
                "source_url": doc.source_metadata.get("source_url"),
                "source_type": doc.source_metadata.get("source_type"),
                "artifact_type": doc.source_metadata.get("artifact_type"),
                "session_id": session_id,
            })

        if not chunks_data:
            return

        query = """
            UNWIND $chunks AS chunk
            MERGE (c:Chunk {id: chunk.id})
            SET c.position = chunk.position,
                c.source_url = chunk.source_url,
                c.source_type = chunk.source_type,
                c.artifact_type = chunk.artifact_type
            WITH c, chunk
            MATCH (d:Document {session_id: chunk.session_id})
            MERGE (c)-[:PART_OF]->(d)
        """
        await session.run(query, {"chunks": chunks_data})
        logger.debug(f"[GRAPH_BUILDER] Created {len(chunks_data)} Chunk nodes")

    async def _create_entities(
        self,
        session: Any,
        graph_documents: List[GraphDocument],
        session_id: str,
    ) -> int:
        """Create Entity nodes with MENTIONED_IN relationship to chunks."""
        entities_data = []
        seen_entities = set()

        for doc in graph_documents:
            chunk_id = doc.chunk_id

            for entity in doc.entities:
                # Deduplicate entities within session
                entity_key = f"{entity.type}:{entity.id}"
                if entity_key in seen_entities:
                    continue
                seen_entities.add(entity_key)

                entities_data.append({
                    "id": entity.id,
                    "type": entity.type,
                    "description": entity.description or "",
                    "chunk_id": chunk_id,
                    "source_url": entity.source_url,
                    "session_id": session_id,
                    "properties": entity.properties or {},
                })

        if not entities_data:
            return 0

        # Create entities with session isolation
        # Entities can exist across sessions, so we use session_ids array
        query = """
            UNWIND $entities AS e
            MERGE (entity:Entity {id: e.id})
            SET entity.type = e.type,
                entity.description = COALESCE(entity.description, e.description),
                entity.session_ids = CASE
                    WHEN e.session_id IN COALESCE(entity.session_ids, [])
                    THEN entity.session_ids
                    ELSE COALESCE(entity.session_ids, []) + [e.session_id]
                END
            WITH entity, e
            MATCH (chunk:Chunk {id: e.chunk_id})
            MERGE (chunk)-[:MENTIONED_IN]->(entity)
        """
        await session.run(query, {"entities": entities_data})

        logger.debug(f"[GRAPH_BUILDER] Created {len(entities_data)} Entity nodes")
        return len(entities_data)

    async def _create_relationships(
        self,
        session: Any,
        graph_documents: List[GraphDocument],
        session_id: str,
    ) -> int:
        """Create relationships between entities."""
        rels_data = []
        seen_rels = set()

        for doc in graph_documents:
            for rel in doc.relationships:
                # Deduplicate relationships
                rel_key = f"{rel.source_id}:{rel.type}:{rel.target_id}"
                if rel_key in seen_rels:
                    continue
                seen_rels.add(rel_key)

                rels_data.append({
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "type": rel.type.upper().replace(" ", "_"),
                    "session_id": session_id,
                })

        if not rels_data:
            return 0

        # Use APOC to create relationships with dynamic type
        # Fall back to static relationship creation if APOC not available
        try:
            query = """
                UNWIND $rels AS rel
                MATCH (source:Entity {id: rel.source_id})
                MATCH (target:Entity {id: rel.target_id})
                CALL apoc.create.relationship(source, rel.type, {session_id: rel.session_id}, target)
                YIELD relationship
                RETURN count(relationship) as count
            """
            result = await session.run(query, {"rels": rels_data})
            logger.debug(f"[GRAPH_BUILDER] Created relationships using APOC")

        except Exception as e:
            # Fallback: Create generic RELATED relationship with type as property
            logger.warning(f"[GRAPH_BUILDER] APOC not available, using generic relationships: {e}")
            query = """
                UNWIND $rels AS rel
                MATCH (source:Entity {id: rel.source_id})
                MATCH (target:Entity {id: rel.target_id})
                MERGE (source)-[r:RELATED]->(target)
                SET r.type = rel.type,
                    r.session_id = rel.session_id
            """
            await session.run(query, {"rels": rels_data})

        logger.debug(f"[GRAPH_BUILDER] Created {len(rels_data)} relationships")
        return len(rels_data)

    async def create_vector_indexes(self, session_id: str):
        """Create vector indexes for entity and community embeddings.

        This should be called after embeddings are created.
        """
        driver = await self._get_driver()

        async with driver.session() as session:
            # Entity vector index
            entity_index_query = """
                CREATE VECTOR INDEX entity_vector IF NOT EXISTS
                FOR (e:Entity) ON e.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """
            try:
                await session.run(entity_index_query)
                logger.info("[GRAPH_BUILDER] Created entity_vector index")
            except Exception as e:
                logger.warning(f"[GRAPH_BUILDER] Could not create entity vector index: {e}")

            # Community vector index
            community_index_query = """
                CREATE VECTOR INDEX community_vector IF NOT EXISTS
                FOR (c:Community) ON c.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """
            try:
                await session.run(community_index_query)
                logger.info("[GRAPH_BUILDER] Created community_vector index")
            except Exception as e:
                logger.warning(f"[GRAPH_BUILDER] Could not create community vector index: {e}")

    async def clear_session_graph(self, session_id: str):
        """Clear all graph data for a session.

        Useful for re-running graph building.
        """
        driver = await self._get_driver()

        async with driver.session() as session:
            # Delete all relationships involving session entities
            await session.run("""
                MATCH (e:Entity)
                WHERE $session_id IN e.session_ids
                WITH e
                OPTIONAL MATCH (e)-[r]-()
                DELETE r
            """, {"session_id": session_id})

            # Remove session_id from entities (don't delete - may be shared)
            await session.run("""
                MATCH (e:Entity)
                WHERE $session_id IN e.session_ids
                SET e.session_ids = [s IN e.session_ids WHERE s <> $session_id]
            """, {"session_id": session_id})

            # Delete orphaned entities (no sessions left)
            await session.run("""
                MATCH (e:Entity)
                WHERE e.session_ids = [] OR e.session_ids IS NULL
                DELETE e
            """)

            # Delete chunks for this session
            await session.run("""
                MATCH (c:Chunk)-[:PART_OF]->(d:Document {session_id: $session_id})
                DETACH DELETE c
            """, {"session_id": session_id})

            # Delete communities for this session
            await session.run("""
                MATCH (c:Community {session_id: $session_id})
                DETACH DELETE c
            """, {"session_id": session_id})

            # Delete document
            await session.run("""
                MATCH (d:Document {session_id: $session_id})
                DELETE d
            """, {"session_id": session_id})

        logger.info(f"[GRAPH_BUILDER] Cleared graph data for session {session_id}")

    async def close(self):
        """Close the Neo4j driver if we own it."""
        if self._driver_owned and self.driver:
            await self.driver.close()
            self.driver = None
            self._driver_owned = False
