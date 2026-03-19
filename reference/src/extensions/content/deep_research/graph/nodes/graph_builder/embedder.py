"""Entity Embedder - Creates embeddings for entities and communities.

Creates vector embeddings for:
- Entity nodes (id + description) for similarity search
- Community summaries for high-level topic search

Embeddings are created AFTER entities/communities exist - you can't embed
something that doesn't exist yet.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import asyncio

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger

logger = get_logger(__name__)


class EntityEmbedder:
    """Creates embeddings for entities and communities.

    WHY embeddings come AFTER extraction:
    - You can't embed an entity until it EXISTS
    - LLM creates entity objects from text
    - THEN we embed: (entity.id + " " + entity.description) → vector

    Same for communities:
    - Leiden detects communities
    - LLM summarizes communities
    - THEN we embed the summaries
    """

    def __init__(
        self,
        neo4j_driver: Any = None,
        embeddings: Any = None,
        context: Any = None,
    ):
        """Initialize the embedder.

        Args:
            neo4j_driver: Neo4j driver instance
            embeddings: Embeddings model (OpenAI, etc.)
            context: ExtensionContext (optional)
        """
        self.driver = neo4j_driver
        self.embeddings = embeddings
        self.context = context

    async def embed_entities(
        self,
        session_id: str,
        batch_size: int = 50,
    ) -> int:
        """Create embeddings for entities in this session.

        Args:
            session_id: Research session ID
            batch_size: Number of entities to process per batch

        Returns:
            Number of entities embedded
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        if not self.embeddings:
            raise RuntimeError("Embeddings model not initialized")

        async with self.driver.session() as session:
            # Get entities without embeddings for this session
            result = await session.run("""
                MATCH (e:Entity)
                WHERE $session_id IN e.session_ids
                AND (e.embedding IS NULL OR e.embedding = [])
                RETURN e.id as id, e.type as type, e.description as description
            """, {"session_id": session_id})

            entities = [record async for record in result]

        if not entities:
            logger.info("[GRAPH_BUILDER] No entities to embed")
            return 0

        logger.info(f"[GRAPH_BUILDER] Embedding {len(entities)} entities")

        embedded_count = 0

        # Process in batches
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]

            # Create texts to embed
            texts = []
            for entity in batch:
                # Embed: type + id + description
                text = f"{entity['type']}: {entity['id']}"
                if entity['description']:
                    text += f" - {entity['description']}"
                texts.append(text)

            # Generate embeddings
            try:
                if hasattr(self.embeddings, 'aembed_documents'):
                    embeddings = await self.embeddings.aembed_documents(texts)
                else:
                    # Fallback: embed one at a time
                    embeddings = []
                    for text in texts:
                        if hasattr(self.embeddings, 'aembed_query'):
                            emb = await self.embeddings.aembed_query(text)
                        else:
                            emb = self.embeddings.embed_query(text)
                        embeddings.append(emb)

                # Store embeddings in Neo4j
                async with self.driver.session() as session:
                    for entity, embedding in zip(batch, embeddings):
                        await session.run("""
                            MATCH (e:Entity {id: $id})
                            CALL db.create.setNodeVectorProperty(e, "embedding", $embedding)
                        """, {"id": entity['id'], "embedding": embedding})

                embedded_count += len(batch)

                logger.debug(f"[GRAPH_BUILDER] Embedded batch {i//batch_size + 1}: {len(batch)} entities")

            except Exception as e:
                logger.error(f"[GRAPH_BUILDER] Failed to embed batch: {e}")
                continue

        logger.info(f"[GRAPH_BUILDER] Embedded {embedded_count} entities")
        return embedded_count

    async def embed_communities(
        self,
        session_id: str,
        batch_size: int = 20,
    ) -> int:
        """Create embeddings for community summaries.

        Args:
            session_id: Research session ID
            batch_size: Number of communities to process per batch

        Returns:
            Number of communities embedded
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        if not self.embeddings:
            raise RuntimeError("Embeddings model not initialized")

        async with self.driver.session() as session:
            # Get communities with summaries but no embeddings
            result = await session.run("""
                MATCH (c:Community {session_id: $session_id})
                WHERE c.summary IS NOT NULL
                AND (c.embedding IS NULL OR c.embedding = [])
                RETURN c.id as id, c.title as title, c.summary as summary
            """, {"session_id": session_id})

            communities = [record async for record in result]

        if not communities:
            logger.info("[GRAPH_BUILDER] No communities to embed")
            return 0

        logger.info(f"[GRAPH_BUILDER] Embedding {len(communities)} communities")

        embedded_count = 0

        for i in range(0, len(communities), batch_size):
            batch = communities[i:i + batch_size]

            # Create texts to embed
            texts = []
            for community in batch:
                text = f"{community['title']}: {community['summary']}"
                texts.append(text)

            # Generate embeddings
            try:
                if hasattr(self.embeddings, 'aembed_documents'):
                    embeddings = await self.embeddings.aembed_documents(texts)
                else:
                    embeddings = []
                    for text in texts:
                        if hasattr(self.embeddings, 'aembed_query'):
                            emb = await self.embeddings.aembed_query(text)
                        else:
                            emb = self.embeddings.embed_query(text)
                        embeddings.append(emb)

                # Store embeddings in Neo4j
                async with self.driver.session() as session:
                    for community, embedding in zip(batch, embeddings):
                        await session.run("""
                            MATCH (c:Community {id: $id})
                            CALL db.create.setNodeVectorProperty(c, "embedding", $embedding)
                        """, {"id": community['id'], "embedding": embedding})

                embedded_count += len(batch)

            except Exception as e:
                logger.error(f"[GRAPH_BUILDER] Failed to embed community batch: {e}")
                continue

        logger.info(f"[GRAPH_BUILDER] Embedded {embedded_count} communities")
        return embedded_count

    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings model."""
        if hasattr(self.embeddings, 'dimension'):
            return self.embeddings.dimension

        # Test with a dummy embedding
        try:
            if hasattr(self.embeddings, 'aembed_query'):
                emb = await self.embeddings.aembed_query("test")
            else:
                emb = self.embeddings.embed_query("test")
            return len(emb)
        except:
            # Default for OpenAI embeddings
            return 1536
