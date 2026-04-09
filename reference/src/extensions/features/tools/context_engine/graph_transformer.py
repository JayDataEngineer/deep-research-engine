"""
ResearchGraphTransformer - Local LLM-based Knowledge Graph Extraction.

NOTE: The canonical implementation of this module has been extracted to
``deep_research_engine.shared.neo4j.graph_transformer.ResearchGraphTransformer``.
This file is kept for reference but should import from the shared module
when both projects are installed together:

    from deep_research_engine.shared.neo4j import ResearchGraphTransformer

This is our own "Diffbot" that runs entirely locally using your LLM.
No external APIs - uses LangChain's LLMGraphTransformer with your configured LLM.

Key Features:
- Dynamic schema extraction: LLM creates Neo4j structure on-the-fly
- Lexical graph: Document → Chunk → Entity provenance tracking
- Structured output: Uses Pydantic for type-safe entity/relationship extraction
- Multi-domain: Adapts to OSINT, research, general content automatically
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Dynamic Schema Extraction
# ============================================================================

class PropertySchema(BaseModel):
    """Property definition with example values."""
    name: str = Field(description="Property name")
    type: str = Field(description="Property type (STRING, INTEGER, etc.)")
    example_values: List[str] = Field(
        default_factory=list,
        description="Example values from actual graph data"
    )


class NodeType(BaseModel):
    """Neo4j node type definition."""
    label: str = Field(description="Node label (e.g., Person, Organization)")
    properties: List[PropertySchema] = Field(
        default_factory=list,
        description="Properties this node type should have (with example values)"
    )
    description: str = Field(
        default="",
        description="Description of what this node represents"
    )
    count: int = Field(
        default=0,
        description="Number of nodes of this type in the graph"
    )


class RelationshipType(BaseModel):
    """Neo4j relationship type definition."""
    label: str = Field(description="Relationship label (e.g., WORKS_FOR, CONNECTED_TO)")
    description: str = Field(
        default="",
        description="Description of what this relationship means"
    )
    properties: List[PropertySchema] = Field(
        default_factory=list,
        description="Properties this relationship should have (with example values)"
    )
    count: int = Field(
        default=0,
        description="Number of relationships of this type in the graph"
    )


class GraphPattern(BaseModel):
    """Triplet pattern: (source_label, relationship_label, target_label)."""
    source: str = Field(description="Source node label")
    relationship: str = Field(description="Relationship label")
    target: str = Field(description="Target node label")


class DynamicGraphSchema(BaseModel):
    """Dynamic Neo4j schema extracted from text by LLM."""
    node_types: List[NodeType] = Field(description="Node types found in the text")
    relationship_types: List[RelationshipType] = Field(description="Relationship types found")
    patterns: List[GraphPattern] = Field(description="Valid relationship patterns")


class ExtractedEntity(BaseModel):
    """Single extracted entity with properties."""
    name: str = Field(description="Entity name/identifier")
    type: str = Field(description="Entity type (Person, Organization, etc.)")
    description: str = Field(default="", description="Entity description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class ExtractedRelationship(BaseModel):
    """Single extracted relationship."""
    head: str = Field(description="Source entity name")
    tail: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship type")
    description: str = Field(default="", description="Relationship description")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedChunk(BaseModel):
    """Text chunk with entities."""
    text: str = Field(description="Chunk text content")
    index: int = Field(description="Chunk index in document")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Entities in this chunk")


class ExtractedDocument(BaseModel):
    """Document with chunks and graph structure."""
    doc_id: str = Field(description="Document ID")
    title: str = Field(default="", description="Document title")
    url: str = Field(default="", description="Document URL")
    chunks: List[ExtractedChunk] = Field(description="Document chunks")
    graph_schema: DynamicGraphSchema = Field(description="Extracted graph schema")
    entities: List[ExtractedEntity] = Field(description="All entities in document")
    relationships: List[ExtractedRelationship] = Field(description="All relationships in document")


# ============================================================================
# ResearchGraphTransformer - Main Extraction Class
# ============================================================================

@dataclass
class TransformResult:
    """Result of graph transformation."""
    document: ExtractedDocument
    nodes_created: int
    relationships_created: int
    chunks_created: int


class ResearchGraphTransformer:
    """
    Local LLM-based knowledge graph extraction.

    This is our own "Diffbot" that:
    - Uses YOUR LLM (no external API keys)
    - Extracts schema dynamically from data
    - Builds lexical graph (Document → Chunk → Entity)
    - Returns structured GraphDocuments for Neo4j storage
    - Enhanced schema: Scans actual graph values for better Cypher generation

    Usage:
        transformer = ResearchGraphTransformer(llm=my_llm)
        result = await transformer.transform(text, metadata={"url": "...", "title": "..."})
        # result.document.schema = Dynamic Neo4j schema
        # result.document.entities = List of entities
        # result.document.relationships = List of relationships

        # Enhanced schema with value scanning:
        enhanced_schema = await transformer.enhance_schema_with_values(schema, neo4j_store)
    """

    def __init__(
        self,
        llm,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        extract_schema: bool = True,
        use_structured_output: bool = True,
        enhance_schema: bool = True,
        value_sample_size: int = 5
    ):
        """Initialize the transformer.

        Args:
            llm: LangChain LLM instance (your configured LLM)
            chunk_size: Text chunk size for processing
            chunk_overlap: Overlap between chunks
            extract_schema: Whether to extract schema dynamically
            use_structured_output: Use Pydantic structured output (if LLM supports it)
            enhance_schema: Whether to enhance schema with actual graph values
            value_sample_size: Number of example values to collect per property
        """
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_schema = extract_schema
        self.use_structured_output = use_structured_output
        self.enhance_schema = enhance_schema
        self.value_sample_size = value_sample_size

        # Initialize LangChain's LLMGraphTransformer for entity extraction
        self.langchain_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=False,  # We'll handle properties ourselves
        )

    async def transform(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_case: str = "general",
        neo4j_store=None
    ) -> TransformResult:
        """
        Transform text into knowledge graph structure.

        Args:
            text: Input text to extract from
            metadata: Document metadata (url, title, etc.)
            use_case: Domain hint (osint, research, general)
            neo4j_store: Optional Neo4jStore for schema enhancement

        Returns:
            TransformResult with ExtractedDocument
        """
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        title = metadata.get("title", "")
        url = metadata.get("url", "")
        session_id = metadata.get("session_id")

        logger.info(f"[RESEARCH_GRAPH_TRANSFORMER] Processing document: {doc_id}")
        logger.info(f"[RESEARCH_GRAPH_TRANSFORMER] Text length: {len(text)} chars")

        # Step 1: Split text into chunks
        chunks = self._split_text(text)
        logger.info(f"[RESEARCH_GRAPH_TRANSFORMER] Created {len(chunks)} chunks")

        # Step 2: Extract schema dynamically (if enabled)
        schema = None
        if self.extract_schema:
            schema = await self._extract_schema(text, use_case)
            logger.info(f"[RESEARCH_GRAPH_TRANSFORMER] Extracted schema: {len(schema.node_types)} node types, {len(schema.relationship_types)} relationship types")

            # Enhance schema with actual graph values if neo4j_store provided
            if self.enhance_schema and neo4j_store:
                try:
                    schema = await self.enhance_schema_with_values(schema, neo4j_store, session_id)
                    logger.info(f"[RESEARCH_GRAPH_TRANSFORMER] Enhanced schema with graph values")
                except Exception as e:
                    logger.warning(f"[RESEARCH_GRAPH_TRANSFORMER] Schema enhancement failed: {e}")

        # Step 3: Extract entities from each chunk
        extracted_chunks = []
        all_entities = {}  # name -> ExtractedEntity
        all_relationships = []

        for i, chunk_text in enumerate(chunks):
            chunk_result = await self._extract_from_chunk(
                chunk_text, i, schema, use_case
            )
            extracted_chunks.append(chunk_result)

            # Merge entities (deduplicate by name)
            for entity in chunk_result.entities:
                if entity.name not in all_entities:
                    all_entities[entity.name] = entity
                else:
                    # Merge properties if entity already exists
                    existing = all_entities[entity.name]
                    for k, v in entity.properties.items():
                        if k not in existing.properties:
                            existing.properties[k] = v

            all_relationships.extend(chunk_result.entities)  # Temporarily using entities field for relationships

        # Actually extract relationships using LangChain transformer
        document = Document(page_content=text, metadata=metadata)
        graph_docs = await self.langchain_transformer.aconvert_to_graph_documents([document])

        # Extract relationships from graph docs
        all_relationships = []
        if graph_docs:
            for rel in graph_docs[0].relationships:
                # Extract node IDs from Node objects (handle both string and Node types)
                head_id = rel.source.id if hasattr(rel.source, 'id') else str(rel.source)
                tail_id = rel.target.id if hasattr(rel.target, 'id') else str(rel.target)

                all_relationships.append(ExtractedRelationship(
                    head=head_id,
                    tail=tail_id,
                    relation=rel.type,
                    description=str(rel.properties) if rel.properties else "",
                    confidence=0.8
                ))

        # Build extracted document
        extracted_doc = ExtractedDocument(
            doc_id=doc_id,
            title=title,
            url=url,
            chunks=extracted_chunks,
            graph_schema=schema or DynamicGraphSchema(node_types=[], relationship_types=[], patterns=[]),
            entities=list(all_entities.values()),
            relationships=all_relationships
        )

        return TransformResult(
            document=extracted_doc,
            nodes_created=len(extracted_doc.entities),
            relationships_created=len(extracted_doc.relationships),
            chunks_created=len(extracted_doc.chunks)
        )

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    async def _extract_schema(
        self,
        text: str,
        use_case: str
    ) -> DynamicGraphSchema:
        """
        Extract Neo4j schema dynamically from text using LLM.

        The LLM analyzes the text and determines:
        - What node types are needed (Person, Organization, etc.)
        - What relationship types exist (WORKS_FOR, CONNECTED_TO, etc.)
        - What properties each type should have
        - What patterns are valid (Person -> WORKS_FOR -> Organization)
        """
        # Sample text for schema extraction (first 3000 chars)
        sample_text = text[:3000]

        schema_prompt = f"""Analyze this text and extract a Neo4j graph schema.

You are building a knowledge graph for {use_case} domain.

TEXT SAMPLE:
{sample_text}

Extract the graph schema:
1. NODE TYPES: What entity types appear? (Person, Organization, Location, etc.)
2. RELATIONSHIP TYPES: How do entities connect? (WORKS_FOR, LOCATED_AT, etc.)
3. PATTERNS: Valid triplets like (Person, WORKS_FOR, Organization)

For each node/relationship type, include relevant properties that should be tracked.
"""

        if self.use_structured_output and self._supports_structured_output():
            # Use Pydantic structured output
            try:
                result = await self.llm.with_structured_output(DynamicGraphSchema).ainvoke([
                    HumanMessage(content=schema_prompt)
                ])
                return result
            except Exception as e:
                logger.warning(f"[RESEARCH_GRAPH_TRANSFORMER] Structured output failed: {e}, falling back to JSON")

        # Fallback to JSON parsing
        response = await self.llm.ainvoke([HumanMessage(content=schema_prompt)])
        import json
        import re

        response_text = str(response.content)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            try:
                schema_data = json.loads(json_match.group())
                return DynamicGraphSchema(**schema_data)
            except Exception as e:
                logger.warning(f"[RESEARCH_GRAPH_TRANSFORMER] JSON parsing failed: {e}")

        # Default minimal schema
        return DynamicGraphSchema(
            node_types=[NodeType(label="Entity", properties=[{"name": "name", "type": "STRING"}])],
            relationship_types=[RelationshipType(label="RELATED_TO")],
            patterns=[GraphPattern(source="Entity", relationship="RELATED_TO", target="Entity")]
        )

    async def _extract_from_chunk(
        self,
        chunk_text: str,
        index: int,
        schema: Optional[DynamicGraphSchema],
        use_case: str
    ) -> ExtractedChunk:
        """Extract entities from a single chunk."""
        # Use LangChain's LLMGraphTransformer
        doc = Document(page_content=chunk_text, metadata={"chunk_index": index})
        graph_docs = await self.langchain_transformer.aconvert_to_graph_documents([doc])

        entities = []
        if graph_docs and graph_docs[0].nodes:
            for node in graph_docs[0].nodes:
                entities.append(ExtractedEntity(
                    name=node.id,
                    type=node.type,
                    description=str(node.properties) if node.properties else "",
                    properties=dict(node.properties) if node.properties else {}
                ))

        return ExtractedChunk(
            text=chunk_text,
            index=index,
            entities=entities
        )

    def _supports_structured_output(self) -> bool:
        """Check if LLM supports native structured output."""
        # Check for common structured output providers
        provider = getattr(self.llm, '_llm_type', '').lower()
        return any(x in provider for x in ['openai', 'anthropic', 'claude', 'vertex'])

    async def enhance_schema_with_values(
        self,
        schema: DynamicGraphSchema,
        neo4j_store,
        session_id: Optional[str] = None
    ) -> DynamicGraphSchema:
        """
        Enhance schema by scanning actual graph values (like Neo4j's enhanced_schema=True).

        This queries Neo4j to find:
        - Actual node labels and their counts
        - Property names and types
        - Example values for each property (for better LLM understanding)

        Args:
            schema: Base schema to enhance
            neo4j_store: Neo4jStore instance
            session_id: Optional session filter

        Returns:
            Enhanced DynamicGraphSchema with example values
        """
        try:
            driver = neo4j_store.driver

            # Get node labels and their properties with example values
            node_types = []

            # Query to get all node labels and their properties
            labels_query = """
            CALL db.labels() YIELD label
            RETURN label ORDER BY label
            """

            async with driver.session(database=neo4j_store.database) as session:
                labels_result = await session.run(labels_query)
                labels_records = await labels_result.data()

                for record in labels_records:
                    label = record["label"]

                    # Count nodes of this type
                    count_query = f"""
                    MATCH (n:{label})
                    {'WHERE n.session_id = $session_id' if session_id else ''}
                    RETURN count(n) as count
                    """
                    count_result = await session.run(
                        count_query,
                        {"session_id": session_id} if session_id else {}
                    )
                    count_record = await count_result.single()
                    node_count = count_record["count"] if count_record else 0

                    # Get properties with example values
                    properties = await self._scan_properties_for_label(
                        session, label, session_id
                    )

                    node_types.append(NodeType(
                        label=label,
                        properties=properties,
                        count=node_count,
                        description=f"Nodes of type {label}"
                    ))

                # Get relationship types and their counts
                relationship_types = []

                rel_types_query = """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN relationshipType ORDER BY relationshipType
                """
                rels_result = await session.run(rel_types_query)
                rels_records = await rels.data()

                for record in rels_records:
                    rel_type = record["relationshipType"]

                    # Count relationships of this type
                    rel_count_query = f"""
                    MATCH ()-[r:{rel_type}]->()
                    RETURN count(r) as count
                    """
                    rel_count_result = await session.run(rel_count_query)
                    rel_count_record = await rel_count_result.single()
                    rel_count = rel_count_record["count"] if rel_count_record else 0

                    # Get relationship properties
                    rel_props = await self._scan_properties_for_relationship(
                        session, rel_type
                    )

                    relationship_types.append(RelationshipType(
                        label=rel_type,
                        properties=rel_props,
                        count=rel_count,
                        description=f"Relationships of type {rel_type}"
                    ))

            # Update schema with enhanced data
            schema.node_types = node_types
            schema.relationship_types = relationship_types

            logger.info(
                f"[RESEARCH_GRAPH_TRANSFORMER] Enhanced schema: "
                f"{len(node_types)} node types, {len(relationship_types)} relationship types"
            )

            return schema

        except Exception as e:
            logger.warning(f"[RESEARCH_GRAPH_TRANSFORMER] Schema enhancement failed: {e}")
            # Return original schema if enhancement fails
            return schema

    async def _scan_properties_for_label(
        self,
        session,
        label: str,
        session_id: Optional[str] = None
    ) -> List[PropertySchema]:
        """
        Scan properties for a node label and collect example values.

        Args:
            session: Neo4j session
            label: Node label
            session_id: Optional session filter

        Returns:
            List of PropertySchema with example values
        """
        properties = []

        # Get properties for this label
        props_query = f"""
        MATCH (n:{label})
        {'WHERE n.session_id = $session_id' if session_id else ''}
        RETURN keys(n) as props
        LIMIT 100
        """

        try:
            result = await session.run(
                props_query,
                {"session_id": session_id} if session_id else {}
            )
            records = await result.data()

            # Collect all unique property names
            all_props = set()
            for record in records:
                all_props.update(record.get("props", []))

            # For each property, get sample values
            for prop_name in sorted(all_props):
                # Get sample values for this property
                sample_query = f"""
                MATCH (n:{label})
                {'WHERE n.session_id = $session_id' if session_id else ''}
                RETURN n.{prop_name} as value
                LIMIT {self.value_sample_size}
                """

                sample_result = await session.run(
                    sample_query,
                    {"session_id": session_id} if session_id else {}
                )
                sample_records = await sample_result.data()

                # Extract example values
                example_values = []
                prop_type = "STRING"
                for sample in sample_records:
                    value = sample.get("value")
                    if value is not None:
                        value_str = str(value)
                        # Truncate long values
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        example_values.append(value_str)

                        # Detect type from first non-null value
                        if isinstance(value, int):
                            prop_type = "INTEGER"
                        elif isinstance(value, float):
                            prop_type = "FLOAT"
                        elif isinstance(value, bool):
                            prop_type = "BOOLEAN"
                        elif isinstance(value, list):
                            prop_type = "ARRAY"
                        elif isinstance(value, dict):
                            prop_type = "MAP"

                if example_values:
                    properties.append(PropertySchema(
                        name=prop_name,
                        type=prop_type,
                        example_values=example_values[:self.value_sample_size]
                    ))

        except Exception as e:
            logger.debug(f"[RESEARCH_GRAPH_TRANSFORMER] Property scan failed for {label}: {e}")

        return properties

    async def _scan_properties_for_relationship(
        self,
        session,
        rel_type: str
    ) -> List[PropertySchema]:
        """
        Scan properties for a relationship type.

        Args:
            session: Neo4j session
            rel_type: Relationship type

        Returns:
            List of PropertySchema with example values
        """
        properties = []

        try:
            # Get properties for this relationship type
            props_query = f"""
            MATCH ()-[r:{rel_type}]->()
            RETURN keys(r) as props
            LIMIT 100
            """

            result = await session.run(props_query)
            records = await result.data()

            # Collect all unique property names
            all_props = set()
            for record in records:
                all_props.update(record.get("props", []))

            # For each property, get sample values
            for prop_name in sorted(all_props):
                sample_query = f"""
                MATCH ()-[r:{rel_type}]->()
                RETURN r.{prop_name} as value
                LIMIT {self.value_sample_size}
                """

                sample_result = await session.run(sample_query)
                sample_records = await sample_result.data()

                # Extract example values
                example_values = []
                prop_type = "STRING"
                for sample in sample_records:
                    value = sample.get("value")
                    if value is not None:
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        example_values.append(value_str)

                        if isinstance(value, int):
                            prop_type = "INTEGER"
                        elif isinstance(value, float):
                            prop_type = "FLOAT"
                        elif isinstance(value, bool):
                            prop_type = "BOOLEAN"

                if example_values:
                    properties.append(PropertySchema(
                        name=prop_name,
                        type=prop_type,
                        example_values=example_values[:self.value_sample_size]
                    ))

        except Exception as e:
            logger.debug(f"[RESEARCH_GRAPH_TRANSFORMER] Relationship property scan failed for {rel_type}: {e}")

        return properties


# ============================================================================
# Helper Functions
# ============================================================================

async def extract_graph_from_findings(
    findings_text: str,
    metadata: Dict[str, Any],
    llm,
    use_case: str = "general"
) -> TransformResult:
    """
    Convenience function to extract graph from worker findings.

    Args:
        findings_text: Worker findings text
        metadata: Document metadata (url, title, worker_id, etc.)
        llm: LangChain LLM instance
        use_case: Domain type (osint, research, general)

    Returns:
        TransformResult with ExtractedDocument
    """
    transformer = ResearchGraphTransformer(llm=llm)
    return await transformer.transform(findings_text, metadata, use_case)


# ============================================================================
# Neo4j Cypher Builders for Dynamic Schema
# ============================================================================

def build_create_constraints(schema: DynamicGraphSchema) -> List[str]:
    """Build Cypher constraints for the schema."""
    statements = []
    for node_type in schema.node_types:
        # Create uniqueness constraint on name property
        statements.append(
            f"CREATE CONSTRAINT entity_{node_type.label.lower()}_name "
            f"IF NOT EXISTS FOR (n:{node_type.label}) REQUIRE n.name IS UNIQUE"
        )
    return statements


def build_create_indexes(schema: DynamicGraphSchema) -> List[str]:
    """Build Cypher indexes for the schema."""
    statements = []
    for node_type in schema.node_types:
        for prop in node_type.properties:
            if prop.get("indexed", False):
                statements.append(
                    f"CREATE INDEX idx_{node_type.label.lower()}_{prop['name']} "
                    f"IF NOT EXISTS FOR (n:{node_type.label}) ON (n.{prop['name']})"
                )
    return statements


async def store_extracted_document(
    extracted_doc: ExtractedDocument,
    neo4j_store,
    session_id: str,
    worker_id: Optional[str] = None
) -> Dict[str, int]:
    """
    Store extracted document in Neo4j with lexical graph structure.

    Creates:
    - Document node
    - Chunk nodes with NEXT_CHUNK relationships
    - Entity nodes
    - MENTIONED_IN relationships (Entity → Chunk)
    - FROM_DOCUMENT relationships (Chunk → Document)

    Args:
        extracted_doc: ExtractedDocument from ResearchGraphTransformer
        neo4j_store: Neo4jStore instance
        session_id: Research session ID
        worker_id: Worker ID for tracking contributions

    Returns:
        Dict with counts: {documents: 1, chunks: N, entities: N, relationships: N}
    """
    from .artifacts.osint import EntityArtifact

    counts = {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}

    # Prepare worker metadata
    worker_meta = {"worker_id": worker_id} if worker_id else {}

    # 1. Create Document node (via finding artifact for now)
    from .artifacts import FindingArtifact
    doc_artifact = FindingArtifact(
        id=f"doc_{extracted_doc.doc_id}",
        content=extracted_doc.title or extracted_doc.doc_id,
        session_id=session_id,
        source_ids=[extracted_doc.url] if extracted_doc.url else [],
        metadata={"doc_type": "document", "url": extracted_doc.url, **worker_meta}
    )
    await neo4j_store._add_internal(doc_artifact)
    counts["documents"] += 1

    # 2. Create Chunk nodes and relationships
    for i, chunk in enumerate(extracted_doc.chunks):
        chunk_artifact = EntityArtifact(
            entity_name=f"chunk_{extracted_doc.doc_id}_{i}",
            entity_type="Chunk",
            content=chunk.text[:500],  # Truncate for storage
            session_id=session_id,
            metadata={
                "chunk_index": i,
                "doc_id": extracted_doc.doc_id,
                "entity_count": len(chunk.entities),
                **worker_meta
            }
        )
        await neo4j_store._add_internal(chunk_artifact)
        counts["chunks"] += 1

        # Create NEXT_CHUNK relationship (except for last chunk)
        if i < len(extracted_doc.chunks) - 1:
            await neo4j_store.add_relationship(
                from_entity=f"chunk_{extracted_doc.doc_id}_{i}",
                to_entity=f"chunk_{extracted_doc.doc_id}_{i+1}",
                relationship_type="NEXT_CHUNK",
                metadata={"order": i, **worker_meta}
            )
            counts["relationships"] += 1

        # Create FROM_DOCUMENT relationship
        await neo4j_store.add_relationship(
            from_entity=f"chunk_{extracted_doc.doc_id}_{i}",
            to_entity=extracted_doc.doc_id,
            relationship_type="FROM_DOCUMENT",
            metadata=worker_meta
        )
        counts["relationships"] += 1

    # 3. Create Entity nodes and MENTIONED_IN relationships
    for entity in extracted_doc.entities:
        # Add worker_id to entity metadata
        entity_meta = {**entity.properties, **worker_meta}

        entity_artifact = EntityArtifact(
            entity_name=entity.name,
            entity_type=entity.type,
            content=entity.description,
            session_id=session_id,
            metadata=entity_meta
        )
        await neo4j_store._add_internal(entity_artifact)
        counts["entities"] += 1

        # Find which chunks mention this entity and create relationships
        for chunk in extracted_doc.chunks:
            for chunk_entity in chunk.entities:
                if chunk_entity.name == entity.name:
                    await neo4j_store.add_relationship(
                        from_entity=entity.name,
                        to_entity=f"chunk_{extracted_doc.doc_id}_{chunk.index}",
                        relationship_type="MENTIONED_IN",
                        metadata={"confidence": 1.0, **worker_meta}
                    )
                    counts["relationships"] += 1

    # 4. Create relationships between entities
    for rel in extracted_doc.relationships:
        await neo4j_store.add_relationship(
            from_entity=rel.head,
            to_entity=rel.tail,
            relationship_type=rel.relation,
            metadata={"confidence": rel.confidence, "description": rel.description, **worker_meta}
        )
        counts["relationships"] += 1

    logger.info(f"[RESEARCH_GRAPH_TRANSFORMER] Stored in Neo4j: {counts}")
    return counts
