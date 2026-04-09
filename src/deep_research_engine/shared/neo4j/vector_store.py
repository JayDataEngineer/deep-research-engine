"""Neo4j Vector Store — Hybrid Vector + Graph Search.

Adds semantic vector search to Neo4j with graph traversal.
Complements Postgres/pgvector — doesn't replace it.

Uses ``langchain-neo4j.Neo4jVector`` for the heavy lifting.
"""

from __future__ import annotations

from typing import Any

from ...utils.logging import get_loguru_logger

logger = get_loguru_logger("shared.neo4j.vector_store")

# ---------------------------------------------------------------------------
# Retrieval queries
# ---------------------------------------------------------------------------

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


class Neo4jVectorStore:
    """Neo4j vector store with hybrid vector+graph search.

    Usage::

        store = Neo4jVectorStore(
            url="bolt://neo4j.local:7687",
            username="neo4j",
            password="secret",
            embedding_model=my_embeddings,
        )
        results = await store.search_entities("Elon Musk companies", k=5)
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
        embedding_node_property: str = "embedding",
    ) -> None:
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        self.index_name = index_name
        self.text_node_property = text_node_property
        self.embedding_node_property = embedding_node_property
        self.embedding_model = embedding_model

        self._vector_store: Any | None = None
        self._driver: Any | None = None

    # ------------------------------------------------------------------
    # Lazy internals
    # ------------------------------------------------------------------

    def _get_vector_store(self) -> Any:
        """Get or create ``Neo4jVector`` instance."""
        if self._vector_store is None:
            if self.embedding_model is None:
                raise ValueError("embedding_model is required for Neo4jVectorStore")

            from langchain_neo4j import Neo4jVector

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

    def _get_driver(self) -> Any:
        """Get or create async Neo4j driver."""
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.url, auth=(self.username, self.password)
            )
        return self._driver

    async def close(self) -> None:
        """Close Neo4j connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Entity-Aware Search (Entity -> Chunk -> Document)
    # ------------------------------------------------------------------

    async def search_entities(
        self,
        query: str,
        k: int = 5,
        retrieval_query: str = DEFAULT_ENTITY_RETRIEVAL_QUERY,
        filter_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search entities with vector similarity + graph traversal.

        Finds entities similar to query, traverses to chunks and documents,
        returns everything in one call.
        """
        try:
            vector_store = self._get_vector_store()
            docs_with_score = vector_store.similarity_search_with_score(
                query, k=k, query_params=filter_params or {}
            )

            results = []
            for doc, score in docs_with_score:
                metadata = getattr(doc, "metadata", {})
                results.append(
                    {
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
                    }
                )

            logger.info(f"Entity search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Vector Index Management
    # ------------------------------------------------------------------

    async def create_vector_index(
        self,
        node_label: str = "Entity",
        embedding_property: str = "embedding",
        dimension: int = 1536,
    ) -> bool:
        """Create vector index on Neo4j nodes."""
        if self.embedding_model is not None:
            try:
                test_embedding = self.embedding_model.embed_query("test")
                dimension = len(test_embedding)
            except Exception:
                pass

        create_query = f"""
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
                await session.run(create_query)
            logger.info(f"Created vector index: {self.index_name} (dim={dimension})")
            return True
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False

    async def populate_embeddings(
        self,
        node_label: str = "Entity",
        text_property: str = "description",
        embedding_property: str = "embedding",
    ) -> int:
        """Populate embeddings for nodes that don't have them."""
        find_query = f"""
        MATCH (n:{node_label})
        WHERE n.{text_property} IS NOT NULL
          AND n.{embedding_property} IS NULL
        RETURN n.{text_property} as text, n.name as name
        LIMIT 1000
        """

        try:
            driver = self._get_driver()
            async with driver.session(database=self.database) as session:
                result = await session.run(find_query)
                records = await result.data()

                if not records:
                    return 0

                updated = 0
                for record in records:
                    text = record.get("text", "")
                    name = record.get("name", "")
                    if text:
                        try:
                            embedding = self.embedding_model.embed_query(text)
                        except Exception:
                            continue

                        await session.run(
                            f"MATCH (n:{node_label} {{name: $name}}) "
                            f"SET n.{embedding_property} = $embedding",
                            {"name": name, "embedding": embedding},
                        )
                        updated += 1

            logger.info(f"Populated {updated} embeddings")
            return updated
        except Exception as e:
            logger.error(f"Failed to populate embeddings: {e}")
            return 0
