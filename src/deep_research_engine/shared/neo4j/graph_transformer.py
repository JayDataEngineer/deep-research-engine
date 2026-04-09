# ruff: noqa: D101
"""ResearchGraphTransformer — LLM-based Knowledge Graph Extraction.

Uses ``langchain-experimental.LLMGraphTransformer`` under the hood for
entity/relationship extraction.  Adds dynamic schema extraction and
schema enhancement with actual graph values.

Optional dependency: install with ``pip install deep-research-engine[graph-transformer]``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ...utils.logging import get_loguru_logger

logger = get_loguru_logger("shared.neo4j.graph_transformer")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PropertySchema(BaseModel):
    name: str = Field(description="Property name")
    type: str = Field(description="Property type (STRING, INTEGER, etc.)")
    example_values: list[str] = Field(default_factory=list)


class NodeType(BaseModel):
    label: str = Field(description="Node label")
    properties: list[PropertySchema] = Field(default_factory=list)
    description: str = Field(default="")
    count: int = Field(default=0)


class RelationshipType(BaseModel):
    label: str = Field(description="Relationship label")
    description: str = Field(default="")
    properties: list[PropertySchema] = Field(default_factory=list)
    count: int = Field(default=0)


class GraphPattern(BaseModel):
    source: str
    relationship: str
    target: str


class DynamicGraphSchema(BaseModel):
    node_types: list[NodeType] = Field(default_factory=list)
    relationship_types: list[RelationshipType] = Field(default_factory=list)
    patterns: list[GraphPattern] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelationship(BaseModel):
    head: str
    tail: str
    relation: str
    description: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedChunk(BaseModel):
    text: str
    index: int
    entities: list[ExtractedEntity] = Field(default_factory=list)


class ExtractedDocument(BaseModel):
    doc_id: str
    title: str = ""
    url: str = ""
    chunks: list[ExtractedChunk] = Field(default_factory=list)
    graph_schema: DynamicGraphSchema = Field(default_factory=DynamicGraphSchema)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


@dataclass
class TransformResult:
    document: ExtractedDocument
    nodes_created: int
    relationships_created: int
    chunks_created: int


# ---------------------------------------------------------------------------
# Main transformer
# ---------------------------------------------------------------------------


class ResearchGraphTransformer:
    """LLM-based knowledge graph extraction.

    Usage::

        transformer = ResearchGraphTransformer(llm=my_llm)
        result = await transformer.transform(text, metadata={"url": "..."})
    """

    def __init__(
        self,
        llm: Any,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        extract_schema: bool = True,
    ) -> None:
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_schema = extract_schema

        # Lazy init of LLMGraphTransformer (optional dependency)
        self._lc_transformer: Any | None = None

    def _get_lc_transformer(self) -> Any:
        if self._lc_transformer is None:
            try:
                from langchain_experimental.graph_transformers import LLMGraphTransformer

                self._lc_transformer = LLMGraphTransformer(llm=self.llm)
            except ImportError:
                raise ImportError(
                    "langchain-experimental is required for ResearchGraphTransformer. "
                    "Install with: pip install deep-research-engine[graph-transformer]"
                )
        return self._lc_transformer

    async def transform(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        use_case: str = "general",
    ) -> TransformResult:
        """Transform text into knowledge graph structure."""
        metadata = metadata or {}
        doc_id = metadata.get(
            "doc_id", f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        logger.info(f"Processing document: {doc_id} ({len(text)} chars)")

        # Step 1: Chunk text
        chunks = self._split_text(text)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 2: Extract schema
        schema: DynamicGraphSchema | None = None
        if self.extract_schema:
            schema = await self._extract_schema(text, use_case)

        # Step 3: Extract entities and relationships via LLMGraphTransformer
        lc_transformer = self._get_lc_transformer()
        document = Document(page_content=text, metadata=metadata)
        graph_docs = await lc_transformer.aconvert_to_graph_documents([document])

        all_entities: dict[str, ExtractedEntity] = {}
        all_relationships: list[ExtractedRelationship] = []

        if graph_docs:
            for node in graph_docs[0].nodes:
                entity = ExtractedEntity(
                    name=node.id,
                    type=node.type,
                    properties=dict(node.properties) if node.properties else {},
                )
                all_entities[entity.name] = entity

            for rel in graph_docs[0].relationships:
                head_id = rel.source.id if hasattr(rel.source, "id") else str(rel.source)
                tail_id = rel.target.id if hasattr(rel.target, "id") else str(rel.target)
                all_relationships.append(
                    ExtractedRelationship(
                        head=head_id,
                        tail=tail_id,
                        relation=rel.type,
                        confidence=0.8,
                    )
                )

        extracted_doc = ExtractedDocument(
            doc_id=doc_id,
            title=metadata.get("title", ""),
            url=metadata.get("url", ""),
            graph_schema=schema or DynamicGraphSchema(),
            entities=list(all_entities.values()),
            relationships=all_relationships,
        )

        return TransformResult(
            document=extracted_doc,
            nodes_created=len(extracted_doc.entities),
            relationships_created=len(extracted_doc.relationships),
            chunks_created=len(chunks),
        )

    def _split_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + self.chunk_size])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    async def _extract_schema(
        self, text: str, use_case: str
    ) -> DynamicGraphSchema:
        sample = text[:3000]
        prompt = (
            f"Analyze this text and extract a Neo4j graph schema for {use_case} domain.\n\n"
            f"TEXT SAMPLE:\n{sample}\n\n"
            "Extract the graph schema as JSON:\n"
            "1. NODE TYPES with properties\n"
            "2. RELATIONSHIP TYPES\n"
            "3. PATTERNS (source_label, relationship, target_label)\n"
        )

        try:
            result = await self.llm.with_structured_output(DynamicGraphSchema).ainvoke(
                [HumanMessage(content=prompt)]
            )
            return result
        except Exception:
            pass

        # Fallback: text + JSON parse
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            match = re.search(r"\{.*\}", str(response.content), re.DOTALL)
            if match:
                return DynamicGraphSchema(**json.loads(match.group()))
        except Exception as e:
            logger.warning(f"Schema extraction failed: {e}")

        return DynamicGraphSchema(
            node_types=[NodeType(label="Entity")],
            relationship_types=[RelationshipType(label="RELATED_TO")],
            patterns=[GraphPattern(source="Entity", relationship="RELATED_TO", target="Entity")],
        )

    async def enhance_schema_with_values(
        self,
        schema: DynamicGraphSchema,
        driver: Any,
        database: str = "neo4j",
    ) -> DynamicGraphSchema:
        """Enhance schema by scanning actual graph values."""
        try:
            async with driver.session(database=database) as session:
                # Node labels
                labels_result = await session.run("CALL db.labels() YIELD label RETURN label")
                labels = [r["label"] async for r in labels_result]

                node_types: list[NodeType] = []
                for label in labels:
                    count_result = await session.run(
                        f"MATCH (n:{label}) RETURN count(n) as count"
                    )
                    count_record = await count_result.single()
                    node_count = count_record["count"] if count_record else 0

                    node_types.append(NodeType(label=label, count=node_count))

                # Relationship types
                rels_result = await session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                rel_types = [r["relationshipType"] async for r in rels_result]

                rel_types_list: list[RelationshipType] = []
                for rt in rel_types:
                    rel_count_result = await session.run(
                        f"MATCH ()-[r:{rt}]->() RETURN count(r) as count"
                    )
                    rel_count_record = await rel_count_result.single()
                    rel_count = rel_count_record["count"] if rel_count_record else 0

                    rel_types_list.append(RelationshipType(label=rt, count=rel_count))

            schema.node_types = node_types
            schema.relationship_types = rel_types_list
            return schema
        except Exception as e:
            logger.warning(f"Schema enhancement failed: {e}")
            return schema


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


async def extract_graph_from_text(
    text: str,
    metadata: dict[str, Any],
    llm: Any,
    use_case: str = "general",
) -> TransformResult:
    """One-shot graph extraction from text."""
    transformer = ResearchGraphTransformer(llm=llm)
    return await transformer.transform(text, metadata, use_case)
