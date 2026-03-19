"""Graph Builder module - Optional NEO4j integration.

This module provides knowledge graph building from LangGraph Store content.

Usage:
    from context_engine.graph_builder import build_knowledge_graph

    result = await build_knowledge_graph(session_id="abc123")

CLI:
    uv run context-engine build-graph --session-id abc123
"""

from .neo4j_builder import (
    Neo4jBuilder,
    Entity,
    Relationship,
    EntityType,
    RelationType,
    build_knowledge_graph,
)

__all__ = [
    "Neo4jBuilder",
    "Entity",
    "Relationship",
    "EntityType",
    "RelationType",
    "build_knowledge_graph",
]
