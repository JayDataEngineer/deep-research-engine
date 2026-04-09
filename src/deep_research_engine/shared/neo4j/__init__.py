"""Shared Neo4j integration module.

Provides:
- Neo4jConnection: Factory for Neo4jGraph + async driver from config
- Neo4jVectorStore: Entity-only vector search via langchain-neo4j
- ResearchGraphTransformer: LLM-based knowledge graph extraction
- NaturalLanguageGraphQuery: NL-to-Cypher query chain
"""

from .connection import Neo4jConnection
from .graph_transformer import (
    DynamicGraphSchema,
    ExtractedDocument,
    ExtractedEntity,
    ExtractedRelationship,
    ResearchGraphTransformer,
    TransformResult,
    extract_graph_from_text,
)
from .qa_chain import (
    GeneratedCypher,
    GraphQueryResult,
    NaturalLanguageGraphQuery,
    format_enhanced_schema_for_llm,
)
from .vector_store import (
    DEFAULT_ENTITY_RETRIEVAL_QUERY,
    Neo4jVectorStore,
)

__all__ = [
    "DEFAULT_ENTITY_RETRIEVAL_QUERY",
    "DynamicGraphSchema",
    "ExtractedDocument",
    "ExtractedEntity",
    "ExtractedRelationship",
    "GeneratedCypher",
    "GraphQueryResult",
    "NaturalLanguageGraphQuery",
    "Neo4jConnection",
    "Neo4jVectorStore",
    "ResearchGraphTransformer",
    "TransformResult",
    "extract_graph_from_text",
    "format_enhanced_schema_for_llm",
]
