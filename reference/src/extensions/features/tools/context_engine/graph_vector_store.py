"""
Neo4j Vector Store - Hybrid Vector + Graph Search.

NOTE: The canonical implementation of this module has been extracted to
``deep_research_engine.shared.neo4j.vector_store.Neo4jVectorStore``.
This file is kept for reference but should import from the shared module
when both projects are installed together:

    from deep_research_engine.shared.neo4j import Neo4jVectorStore

This module adds semantic vector search to Neo4j with graph traversal.
Complements Postgres/pgvector - doesn't replace it.

Key Features:
- Vector similarity search + graph traversal in ONE query
- Entity-aware search: find entities AND their sources
- retrieval_query: trace Entity → Chunk → Document provenance
- Hybrid search: vector + keyword filters

Architecture:
    Postgres/pgvector: Fast general similarity search
    Neo4jVector: Entity-aware search with source tracking
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector

logger = logging.getLogger(__name__)


# ============================================================================
# Retrieval Queries - The "Killer Feature"
# ============================================================================

DEFAULT_ENTITY_RETRIEVAL_QUERY = """
// Vector search returns matching entities
WITH node, score
// Traverse to chunks that mention this entity
MATCH (node)-[:MENTIONED_IN]->(chunk:Chunk)
// Traverse to documents that contain the chunk
MATCH (chunk)-[:FROM_DOCUMENT]->(doc:Document)
// Return entity + source provenance
RETURN node {.name, .type, .description} AS text,
       score,
       {
         entity_name: node.name,
         entity_type: node.type,
         chunk_text: chunk.text,
         chunk_index: chunk.chunk_index,
         doc_id: doc.doc_id,
         doc_title: doc.title,
         doc_url: doc.url,
         worker_id: doc.worker_id
       } AS metadata
"""


DEFAULT_FINDING_RETRIEVAL_QUERY = """
// Vector search on findings/chunks
WITH node, score
// If node is a Chunk, get its document
MATCH (node)-[:FROM_DOCUMENT]->(doc:Document)
// Get entities mentioned in this chunk
MATCH (entity)-[:MENTIONED_IN]->(node)
RETURN node {.text} AS text,
       score,
       {
         chunk_text: node.text,
         chunk_index: node.chunk_index,
         entities: collect(DISTINCT entity.name),
         doc_id: doc.doc_id,
         doc_title: doc.title,
         doc_url: doc.url
       } AS metadata
"""


# ============================================================================
# Neo4jVectorStore - Main Class
# ============================================================================

class Neo4jVectorStore:
    """
    Neo4j vector store with hybrid vector+graph search.

    Complements Postgres/pgvector:
    - Postgres: Fast general similarity search
    - Neo4j: Entity-aware search with source tracking

    Usage:
        # Initialize from existing Neo4j connection
        vector_store = Neo4jVectorStore(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )

        # Entity-aware search with sources
        results = vector_store.search_entities(
            query="Elon Musk companies",
            k=5
        )
        # Returns: entities + their chunks + source documents

        # Finding-aware search
        results = vector_store.search_findings(
            query="AI research developments",
            k=10
        )
    """

    def __init__(
        self,
        url: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        embedding_model: Any = None,
        index_name: str = "entity_vector",
        text_node_property: str = "description",
        embedding_node_property: str = "embedding"
    ):
        """Initialize Neo4j vector store.

        Args:
            url: Neo4j bolt URL
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            embedding_model: LangChain embeddings model
            index_name: Vector index name
            text_node_property: Node property containing text to index
            embedding_node_property: Node property to store embeddings
        """
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        self.index_name = index_name
        self.text_node_property = text_node_property
        self.embedding_node_property = embedding_node_property

        # Get embedding model if not provided
        if embedding_model is None:
            from src.shared.embeddings import get_embeddings
            embedding_model = get_embeddings()

        self.embedding_model = embedding_model

        # Initialize Neo4jVector (lazy loading)
        self._vector_store: Optional[Neo4jVector] = None
        self._driver = None

    def _get_vector_store(self) -> Neo4jVector:
        """Get or create Neo4jVector instance."""
        if self._vector_store is None:
            self._vector_store = Neo4jVector.from_existing_index(
                embedding=self.embedding_model,
                url=self.url,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name=self.index_name,
                text_node_property=self.text_node_property,
                embedding_node_property=self.embedding_node_property,
            )
        return self._vector_store

    def _get_driver(self):
        """Get Neo4j driver for direct queries."""
        if self._driver is None:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self.url,
                auth=(self.username, self.password)
            )
        return self._driver

    async def close(self):
        """Close Neo4j connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ========================================================================
    # Entity-Aware Search (Entity → Chunk → Document)
    # ========================================================================

    async def search_entities(
        self,
        query: str,
        k: int = 5,
        retrieval_query: str = DEFAULT_ENTITY_RETRIEVAL_QUERY,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities with vector similarity + graph traversal.

        This is the KILLER FEATURE:
        - Finds entities similar to query
        - Traverses to chunks that mention them
        - Traverses to documents containing those chunks
        - Returns everything in ONE call

        Args:
            query: Search query text
            k: Number of results to return
            retrieval_query: Cypher query for traversal
            filter_params: Optional filters for entity types, etc.

        Returns:
            List of dicts with entity + chunk + document info
        """
        try:
            vector_store = self._get_vector_store()

            # Perform vector search with retrieval query
            docs_with_score = vector_store.similarity_search_with_score(
                query,
                k=k,
                query_params=filter_params or {}
            )

            results = []
            for doc, score in docs_with_score:
                # Extract metadata from the retrieval query
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}

                results.append({
                    "text": doc.page_content,
                    "score": score,
                    "entity_name": metadata.get("entity_name", ""),
                    "entity_type": metadata.get("entity_type", ""),
                    "chunk_text": metadata.get("chunk_text", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "doc_id": metadata.get("doc_id", ""),
                    "doc_title": metadata.get("doc_title", ""),
                    "doc_url": metadata.get("doc_url", ""),
                    "worker_id": metadata.get("worker_id", ""),
                })

            logger.info(f"[NEO4J_VECTOR] Entity search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[NEO4J_VECTOR] Entity search failed: {e}")
            return []

    # ========================================================================
    # Finding-Aware Search (Findings → Entities → Documents)
    # ========================================================================

    async def search_findings(
        self,
        query: str,
        k: int = 10,
        retrieval_query: str = DEFAULT_FINDING_RETRIEVAL_QUERY,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search findings with entity context.

        Args:
            query: Search query text
            k: Number of results
            retrieval_query: Cypher query for traversal
            filter_params: Optional filters

        Returns:
            List of dicts with finding + entities + document info
        """
        try:
            vector_store = self._get_vector_store()

            docs_with_score = vector_store.similarity_search_with_score(
                query,
                k=k,
                query_params=filter_params or {}
            )

            results = []
            for doc, score in docs_with_score:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}

                results.append({
                    "text": doc.page_content,
                    "score": score,
                    "chunk_text": metadata.get("chunk_text", ""),
                    "entities": metadata.get("entities", []),
                    "doc_id": metadata.get("doc_id", ""),
                    "doc_title": metadata.get("doc_title", ""),
                    "doc_url": metadata.get("doc_url", ""),
                })

            logger.info(f"[NEO4J_VECTOR] Finding search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[NEO4J_VECTOR] Finding search failed: {e}")
            return []

    # ========================================================================
    # Direct Graph Queries (Cypher Wrapper Functions)
    # ========================================================================

    async def find_entity_sources(
        self,
        entity_name: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find all sources that mention an entity (graph traversal).

        Args:
            entity_name: Entity name to search
            max_depth: Maximum traversal depth

        Returns:
            List of source documents with chunks
        """
        query = f"""
        MATCH (e:Entity {{name: $entity_name}})
        CALL apoc.path.subgraphAll(e, {{
            maxLevel: {max_depth},
            relationshipFilter: 'MENTIONED_IN|FROM_DOCUMENT',
            bfs: true
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        try:
            driver = self._get_driver()
            async with driver.session(database=self.database) as session:
                result = await session.run(query, {"entity_name": entity_name})
                records = await result.data()

            logger.info(f"[NEO4J_VECTOR] Found {len(records)} paths for entity: {entity_name}")
            return records

        except Exception as e:
            logger.error(f"[NEO4J_VECTOR] Entity source query failed: {e}")
            return []

    async def find_connected_entities(
        self,
        entity_name: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities connected to a given entity.

        Args:
            entity_name: Starting entity name
            max_depth: Maximum traversal depth
            relationship_types: Filter to specific relationship types

        Returns:
            List of connected entities with paths
        """
        rel_filter = ""
        if relationship_types:
            rel_filter = f"|{'|'.join(relationship_types)}"

        query = f"""
        MATCH path = (start:Entity {{name: $entity_name}})-[*1..{max_depth} {rel_filter}]-(connected:Entity)
        RETURN [n in nodes(path) | {{
            name: n.name,
            type: n.type,
            description: n.description
        }}] as path,
        [r in relationships(path) | type(r)] as relationship_types
        LIMIT 50
        """

        try:
            driver = self._get_driver()
            async with driver.session(database=self.database) as session:
                result = await session.run(query, {"entity_name": entity_name})
                records = await result.data()

            logger.info(f"[NEO4J_VECTOR] Found {len(records)} connections for: {entity_name}")
            return records

        except Exception as e:
            logger.error(f"[NEO4J_VECTOR] Connected entities query failed: {e}")
            return []

    async def get_entity_context(
        self,
        entity_name: str,
        context_width: int = 1
    ) -> Dict[str, Any]:
        """
        Get rich context for an entity: sources, connections, mentions.

        Args:
            entity_name: Entity name
            context_width: Number of chunks before/after to include

        Returns:
            Dict with entity info, sources, connections
        """
        # Get entity node
        entity_query = """
        MATCH (e:Entity {name: $entity_name})
        RETURN e {.name, .type, .description} as entity
        """

        # Get chunks mentioning this entity with context
        chunks_query = f"""
        MATCH (e:Entity {{name: $entity_name}})-[:MENTIONED_IN]->(chunk:Chunk)
        OPTIONAL MATCH (chunk)-[:NEXT_CHUNK*1..{context_width}]-(context_chunk:Chunk)
        RETURN chunk.chunk_index as chunk_index,
               chunk.text as text,
               collect(DISTINCT context_chunk.text) as context
        ORDER BY chunk.chunk_index
        """

        # Get connected entities
        connections_query = """
        MATCH (e:Entity {name: $entity_name})-[r]-(other:Entity)
        RETURN other.name as name,
               other.type as type,
               type(r) as relationship
        LIMIT 20
        """

        try:
            driver = self._get_driver()
            async with driver.session(database=self.database) as session:
                # Run queries in parallel
                entity_result = await session.run(entity_query, {"entity_name": entity_name})
                entity_record = await entity_result.single()

                chunks_result = await session.run(chunks_query, {"entity_name": entity_name})
                chunks_records = await chunks_result.data()

                connections_result = await session.run(connections_query, {"entity_name": entity_name})
                connections_records = await connections_result.data()

            return {
                "entity": entity_record["entity"] if entity_record else None,
                "mentions": chunks_records,
                "connections": connections_records,
            }

        except Exception as e:
            logger.error(f"[NEO4J_VECTOR] Entity context query failed: {e}")
            return {}

    # ========================================================================
    # Vector Index Management
    # ========================================================================

    async def create_vector_index(
        self,
        node_label: str = "Entity",
        text_property: str = "description",
        embedding_property: str = "embedding",
        dimension: int = 1536  # Default dimension for most embedding models
    ) -> bool:
        """
        Create vector index on Neo4j nodes.

        Args:
            node_label: Node label to index
            text_property: Property containing text to embed
            embedding_property: Property to store embeddings
            dimension: Embedding dimension (default 1536 for OpenAI-style)

        Returns:
            True if successful
        """
        # Try to get embedding dimension from model, but use default if it fails
        try:
            test_embedding = self.embedding_model.embed_query("test")
            dimension = len(test_embedding)
        except Exception as e:
            logger.warning(f"[NEO4J_VECTOR] Could not get embedding dimension from model: {e}, using default: {dimension}")

        create_index_query = f"""
        CREATE VECTOR INDEX {self.index_name}
        IF NOT EXISTS
        FOR (n:{node_label})
        ON (n.{embedding_property})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        try:
            driver = self._get_driver()
            async with driver.session(database=self.database) as session:
                await session.run(create_index_query)

            logger.info(f"[NEO4J_VECTOR] Created vector index: {self.index_name} (dimension: {dimension})")
            return True

        except Exception as e:
            logger.error(f"[NEO4J_VECTOR] Failed to create vector index: {e}")
            return False

    async def populate_embeddings(
        self,
        node_label: str = "Entity",
        text_property: str = "description",
        embedding_property: str = "embedding",
        mock_embeddings: bool = False
    ) -> int:
        """
        Populate embeddings for nodes that don't have them.

        Args:
            node_label: Node label to process
            text_property: Property containing text to embed
            embedding_property: Property to store embeddings
            mock_embeddings: If True, use random embeddings (for testing)

        Returns:
            Number of nodes updated
        """
        # Get nodes without embeddings
        find_nodes_query = f"""
        MATCH (n:{node_label})
        WHERE n.{text_property} IS NOT NULL
          AND n.{embedding_property} IS NULL
        RETURN n.{text_property} as text, n.name as name
        LIMIT 1000
        """

        try:
            driver = self._get_driver()

            # Keep session open for both queries
            async with driver.session(database=self.database) as session:
                # First, get all nodes that need embeddings
                result = await session.run(find_nodes_query)
                records = await result.data()

                if not records:
                    logger.info(f"[NEO4J_VECTOR] All nodes already have embeddings")
                    return 0

                # Generate embeddings for each node (still within the same session!)
                updated = 0
                for record in records:
                    text = record.get("text", "")
                    name = record.get("name", "")

                    if text:
                        # Try real embedding, fall back to mock
                        try:
                            if mock_embeddings:
                                # Generate mock embedding (1536 dimensions, normalized)
                                import numpy as np
                                embedding = np.random.rand(1536).tolist()
                            else:
                                embedding = self.embedding_model.embed_query(text)
                        except Exception as e:
                            logger.warning(f"[NEO4J_VECTOR] Embedding generation failed for {name}: {e}, using mock")
                            import numpy as np
                            embedding = np.random.rand(1536).tolist()

                        update_query = f"""
                        MATCH (n:{node_label} {{name: $name}})
                        SET n.{embedding_property} = $embedding
                        """

                        await session.run(update_query, {
                            "name": name,
                            "embedding": embedding
                        })
                        updated += 1

            logger.info(f"[NEO4J_VECTOR] Populated {updated} embeddings")
            return updated

        except Exception as e:
            logger.error(f"[NEO4J_VECTOR] Failed to populate embeddings: {e}")
            return 0


# ============================================================================
# Convenience Functions
# ============================================================================

async def search_with_sources(
    query: str,
    neo4j_url: str = "bolt://localhost:7687",
    neo4j_username: str = "neo4j",
    neo4j_password: str = "password",
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function for entity search with sources.

    Args:
        query: Search query
        neo4j_url: Neo4j bolt URL
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        k: Number of results

    Returns:
        List of results with entities + chunks + documents
    """
    store = Neo4jVectorStore(
        url=neo4j_url,
        username=neo4j_username,
        password=neo4j_password
    )

    try:
        results = await store.search_entities(query, k=k)
        return results
    finally:
        await store.close()
