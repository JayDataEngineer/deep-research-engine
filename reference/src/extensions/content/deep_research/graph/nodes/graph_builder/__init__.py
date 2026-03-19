"""Graph Builder Node - Builds knowledge graphs from research findings.

This module provides a single isolated node that:
1. Pulls findings from Context Engine (Postgres/MinIO)
2. Extracts entities and concepts using LLMGraphTransformer
3. Consolidates schema (normalizes entity types)
4. Builds Neo4j graph with session isolation
5. Creates entity embeddings
6. Detects communities using Leiden algorithm
7. Generates community summaries and embeddings

All Neo4j creation happens HERE - Flash Explorers should NOT write to Neo4j.
This node runs ONCE before synthesis.
"""

from .node import GraphBuilderNode, graph_builder_node
from .chunker import FindingChunker, EntityChunk
from .extractor import ConceptExtractor
from .consolidator import SchemaConsolidator
from .builder import Neo4jGraphBuilder
from .embedder import EntityEmbedder
from .community_detector import CommunityDetector
from .summarizer import CommunitySummarizer

__all__ = [
    "GraphBuilderNode",
    "graph_builder_node",
    "FindingChunker",
    "EntityChunk",
    "ConceptExtractor",
    "SchemaConsolidator",
    "Neo4jGraphBuilder",
    "EntityEmbedder",
    "CommunityDetector",
    "CommunitySummarizer",
]
