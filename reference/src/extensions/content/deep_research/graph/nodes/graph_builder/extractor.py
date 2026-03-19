"""Concept Extractor - Extracts entities and concepts using LLMGraphTransformer.

Focuses on CONCEPTS and FINDINGS, not just named entities (people, orgs).
This is the key difference from the LLM Graph Builder approach.
"""

from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import logging
import asyncio

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# CANONICAL ENTITY AND RELATIONSHIP TYPES
# =============================================================================

# Entity types focused on CONCEPTS, not just people
CANONICAL_ENTITY_TYPES = [
    "Concept",      # Ideas, theories, frameworks, paradigms
    "Finding",      # Specific claims, results, discoveries
    "Method",       # Methodologies, approaches, techniques
    "Tool",         # Technologies, software, instruments
    "Person",       # People, researchers, authors
    "Organization", # Companies, institutions, groups
    "Dataset",      # Data sources, corpora
    "Topic",        # High-level subject areas
    "Event",        # Conferences, milestones, dates
    "Location",     # Physical places
]

# Relationship types for concept-focused graphs
CANONICAL_RELATIONSHIPS = [
    # Concept relationships
    "RELATES_TO",       # General connection
    "SUPPORTS",         # A provides evidence for B
    "CONTRADICTS",      # A conflicts with B
    "DERIVED_FROM",     # A builds on B
    "SUBSUMES",         # A contains/is broader than B
    "APPLIES",          # A uses B

    # Attribution relationships
    "ATTRIBUTED_TO",    # Finding → Person/Org
    "FOUND_BY",         # Finding → Method
    "MEASURED_IN",      # Finding → Dataset
    "PUBLISHED_IN",     # Finding → Event/Location

    # Standard relationships
    "WORKS_AT",
    "LOCATED_IN",
    "PART_OF",
    "USED_BY",
]


class ExtractedEntity(BaseModel):
    """An extracted entity/concept from text."""

    id: str = Field(..., description="Unique entity identifier")
    type: str = Field(..., description="Entity type (Concept, Finding, Person, etc.)")
    description: Optional[str] = Field(default=None, description="Entity description")
    chunk_id: Optional[str] = Field(default=None, description="Source chunk ID")
    source_url: Optional[str] = Field(default=None, description="Source URL")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class ExtractedRelationship(BaseModel):
    """An extracted relationship between entities."""

    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class GraphDocument(BaseModel):
    """A graph document containing entities and relationships extracted from a chunk."""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    chunk_id: Optional[str] = Field(default=None)
    source_metadata: Dict[str, Any] = Field(default_factory=dict)


class ConceptExtractor:
    """Extracts concepts and entities using LLMGraphTransformer.

    Key difference from standard entity extraction:
    - Focus on CONCEPTS and FINDINGS, not just named entities
    - Extracts semantic relationships (SUPPORTS, CONTRADICTS, DERIVED_FROM)
    - Includes methodological context (FOUND_BY, MEASURED_IN)
    """

    def __init__(self, llm: Any = None, context: Any = None):
        """Initialize the concept extractor.

        Args:
            llm: Language model to use for extraction
            context: ExtensionContext (optional)
        """
        self.llm = llm
        self.context = context
        self._transformer = None

    async def _get_transformer(self):
        """Get or create the LLMGraphTransformer."""
        if self._transformer is not None:
            return self._transformer

        try:
            from langchain_experimental.graph_transformers import LLMGraphTransformer

            self._transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=CANONICAL_ENTITY_TYPES,
                allowed_relationships=CANONICAL_RELATIONSHIPS,
                node_properties=["description", "confidence"],
                strict_mode=False,  # Allow some flexibility
            )

            return self._transformer

        except ImportError:
            logger.warning("[GRAPH_BUILDER] langchain_experimental not available, using fallback extraction")
            return None

    async def extract(
        self,
        chunks: List[Any],  # List of EntityChunk
        batch_size: int = 5,
    ) -> List[GraphDocument]:
        """Extract entities and relationships from chunks.

        Args:
            chunks: List of EntityChunk objects
            batch_size: Number of chunks to process in parallel

        Returns:
            List of GraphDocument objects with extracted entities and relationships
        """
        if not chunks:
            logger.warning("[GRAPH_BUILDER] No chunks to extract entities from")
            return []

        logger.info(f"[GRAPH_BUILDER] Extracting entities from {len(chunks)} chunks")

        transformer = await self._get_transformer()

        if transformer is not None:
            return await self._extract_with_transformer(chunks, transformer, batch_size)
        else:
            return await self._extract_with_fallback(chunks, batch_size)

    async def _extract_with_transformer(
        self,
        chunks: List[Any],
        transformer: Any,
        batch_size: int,
    ) -> List[GraphDocument]:
        """Extract using LLMGraphTransformer."""
        from langchain_core.documents import Document

        graph_documents = []

        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            tasks = []
            for chunk in batch:
                doc = Document(
                    page_content=chunk.text,
                    metadata={
                        "chunk_id": chunk.id,
                        "source_url": chunk.source_url,
                        "source_type": chunk.source_type,
                        "artifact_type": chunk.artifact_type,
                    }
                )
                tasks.append(self._extract_single(doc, transformer, chunk))

            # Process batch in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"[GRAPH_BUILDER] Extraction failed: {result}")
                elif result:
                    graph_documents.append(result)

            logger.info(f"[GRAPH_BUILDER] Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        return graph_documents

    async def _extract_single(
        self,
        doc: Any,
        transformer: Any,
        chunk: Any,
    ) -> Optional[GraphDocument]:
        """Extract entities from a single document."""
        try:
            # Use the transformer to extract graph documents
            graph_docs = await transformer.aconvert_to_graph_documents([doc])

            if not graph_docs:
                return None

            graph_doc = graph_docs[0]

            # Convert to our format
            entities = []
            for node in graph_doc.nodes:
                entity = ExtractedEntity(
                    id=node.id,
                    type=node.type,
                    description=node.properties.get("description", ""),
                    chunk_id=chunk.id,
                    source_url=chunk.source_url,
                    properties=node.properties,
                )
                entities.append(entity)

            relationships = []
            for rel in graph_doc.relationships:
                relationship = ExtractedRelationship(
                    source_id=rel.source.id,
                    target_id=rel.target.id,
                    type=rel.type,
                    properties=getattr(rel, 'properties', {}) or {},
                )
                relationships.append(relationship)

            return GraphDocument(
                entities=entities,
                relationships=relationships,
                chunk_id=chunk.id,
                source_metadata={
                    "source_url": chunk.source_url,
                    "source_type": chunk.source_type,
                    "artifact_type": chunk.artifact_type,
                }
            )

        except Exception as e:
            logger.warning(f"[GRAPH_BUILDER] Failed to extract from chunk {chunk.id}: {e}")
            return None

    async def _extract_with_fallback(
        self,
        chunks: List[Any],
        batch_size: int,
    ) -> List[GraphDocument]:
        """Fallback extraction using direct LLM calls when LLMGraphTransformer is unavailable.

        This uses a structured prompt to extract entities and relationships.
        """
        from pydantic import BaseModel, Field
        from typing import List as TypingList

        class EntityExtraction(BaseModel):
            """Extraction result for a single chunk."""
            entities: TypingList[Dict[str, Any]] = Field(default_factory=list)
            relationships: TypingList[Dict[str, Any]] = Field(default_factory=list)

        graph_documents = []

        extraction_prompt = """Extract entities and relationships from the following research text.

Focus on:
1. CONCEPTS: Ideas, theories, frameworks, paradigms
2. FINDINGS: Specific claims, results, discoveries
3. METHODS: Methodologies, approaches, techniques
4. TOOLS: Technologies, software, instruments
5. PEOPLE: Researchers, authors
6. ORGANIZATIONS: Companies, institutions

For each entity, provide:
- id: A unique identifier (short, no spaces)
- type: One of Concept, Finding, Method, Tool, Person, Organization, Dataset, Topic
- description: A brief description

For relationships, provide:
- source: Source entity id
- target: Target entity id
- type: One of RELATES_TO, SUPPORTS, CONTRADICTS, DERIVED_FROM, APPLIES, FOUND_BY, ATTRIBUTED_TO

Text:
{text}

Extract entities and relationships as JSON."""

        for i, chunk in enumerate(chunks):
            try:
                # Call LLM directly
                prompt = extraction_prompt.format(text=chunk.text[:2000])  # Limit text length

                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                else:
                    response = self.llm(prompt)
                    content = str(response)

                # Parse the response
                import json
                import re

                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = {"entities": [], "relationships": []}

                # Convert to GraphDocument
                entities = []
                for e in data.get("entities", []):
                    entities.append(ExtractedEntity(
                        id=e.get("id", f"entity_{i}"),
                        type=e.get("type", "Concept"),
                        description=e.get("description", ""),
                        chunk_id=chunk.id,
                        source_url=chunk.source_url,
                    ))

                relationships = []
                for r in data.get("relationships", []):
                    relationships.append(ExtractedRelationship(
                        source_id=r.get("source", ""),
                        target_id=r.get("target", ""),
                        type=r.get("type", "RELATES_TO"),
                    ))

                graph_documents.append(GraphDocument(
                    entities=entities,
                    relationships=relationships,
                    chunk_id=chunk.id,
                ))

            except Exception as e:
                logger.warning(f"[GRAPH_BUILDER] Fallback extraction failed for chunk {chunk.id}: {e}")
                continue

        return graph_documents
