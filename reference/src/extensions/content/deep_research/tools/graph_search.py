"""Graph Search Tools for Deep Research.

These tools query the Neo4j knowledge graph built by graph_builder_node.
They're available to synthesis and critic nodes for intelligent retrieval.

Tools:
- graph_entity_search: Vector search on entities
- graph_community_search: Search community summaries
- graph_traverse: Multi-hop graph traversal
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=7)


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class GraphEntitySearchInput(BaseModel):
    """Input for graph entity search."""
    query: str = Field(..., description="Search query to find similar entities")
    k: int = Field(default=10, description="Number of results to return")


class GraphCommunitySearchInput(BaseModel):
    """Input for graph community search."""
    query: str = Field(..., description="Search query to find relevant communities")
    k: int = Field(default=5, description="Number of communities to return")


class GraphTraverseInput(BaseModel):
    """Input for graph traversal."""
    entity_id: str = Field(..., description="Starting entity ID")
    depth: int = Field(default=2, description="Traversal depth (1-3)")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_neo4j_connection():
    """Get Neo4j connection from environment or context engine."""
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    return GraphDatabase.driver(uri, auth=(username, password))


def _get_embeddings_model():
    """Get embeddings model from factory."""
    try:
        from src.shared.llm.factory import get_embeddings
        return get_embeddings()
    except Exception:
        return None


def _get_session_id() -> str:
    """Get current session ID from environment."""
    return os.getenv("RESEARCH_SESSION_ID", "")


# =============================================================================
# TOOLS
# =============================================================================

@tool
async def graph_entity_search(
    query: str,
    k: int = 10,
) -> str:
    """Search for entities in the knowledge graph using vector similarity.

    This tool finds entities (concepts, findings, people, organizations) that
    are semantically similar to your query. Use it to discover relevant
    entities from the research knowledge graph.

    Returns entity details including:
    - Entity ID and type
    - Description
    - Related entities and relationships
    - Source chunks where the entity appears

    Args:
        query: Search query describing what entities to find
        k: Number of results (default: 10)

    Returns:
        JSON string with list of matching entities and their relationships
    """
    import json

    session_id = _get_session_id()
    if not session_id:
        return json.dumps({"error": "No session_id available. Set RESEARCH_SESSION_ID env var."})

    # Get embeddings
    embeddings = _get_embeddings_model()
    if not embeddings:
        return json.dumps({"error": "Could not get embeddings model"})

    try:
        # Generate query embedding
        if hasattr(embeddings, 'aembed_query'):
            query_embedding = await embeddings.aembed_query(query)
        else:
            query_embedding = embeddings.embed_query(query)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate embedding: {str(e)}"})

    driver = _get_neo4j_connection()
    results = []

    try:
        with driver.session() as session:
            # Vector search on entities
            cypher = """
                CALL db.index.vector.queryNodes('entity_vector', $k, $embedding)
                YIELD node AS entity, score
                WHERE $session_id IN entity.session_ids
                OPTIONAL MATCH (entity)-[r]-(related:Entity)
                WHERE $session_id IN related.session_ids
                WITH entity, score,
                     collect(DISTINCT {
                         id: related.id,
                         type: related.type,
                         relationship: type(r)
                     }) AS connections
                RETURN entity.id AS id,
                       entity.type AS type,
                       entity.description AS description,
                       score,
                       connections
                ORDER BY score DESC
            """

            result = session.run(cypher, {
                "embedding": query_embedding,
                "k": k * 2,  # Request more to account for session filtering
                "session_id": session_id,
            })

            for record in result:
                entity = {
                    "id": record["id"],
                    "type": record["type"],
                    "description": record["description"] or "",
                    "score": round(record["score"], 4),
                    "connections": record["connections"][:5] if record["connections"] else [],
                }
                results.append(entity)

                if len(results) >= k:
                    break

    except Exception as e:
        return json.dumps({"error": f"Neo4j query failed: {str(e)}"})
    finally:
        driver.close()

    return json.dumps({
        "query": query,
        "session_id": session_id,
        "count": len(results),
        "entities": results,
    }, indent=2)


@tool
async def graph_community_search(
    query: str,
    k: int = 5,
) -> str:
    """Search for communities in the knowledge graph.

    Communities are clusters of related entities discovered by the Leiden
    algorithm. Each community has an LLM-generated title and summary.
    Use this for high-level topic discovery and understanding research themes.

    Returns:
    - Community title and summary
    - Key entities in the community
    - Similarity score

    Args:
        query: Search query describing what topics to find
        k: Number of communities to return (default: 5)

    Returns:
        JSON string with list of matching communities
    """
    import json

    session_id = _get_session_id()
    if not session_id:
        return json.dumps({"error": "No session_id available. Set RESEARCH_SESSION_ID env var."})

    # Get embeddings
    embeddings = _get_embeddings_model()
    if not embeddings:
        return json.dumps({"error": "Could not get embeddings model"})

    try:
        if hasattr(embeddings, 'aembed_query'):
            query_embedding = await embeddings.aembed_query(query)
        else:
            query_embedding = embeddings.embed_query(query)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate embedding: {str(e)}"})

    driver = _get_neo4j_connection()
    results = []

    try:
        with driver.session() as session:
            # Vector search on communities
            cypher = """
                CALL db.index.vector.queryNodes('community_vector', $k, $embedding)
                YIELD node AS community, score
                WHERE community.session_id = $session_id
                MATCH (community)<-[:IN_COMMUNITY]-(e:Entity)
                WITH community, score, collect(DISTINCT e.id) AS entities
                RETURN community.id AS id,
                       community.title AS title,
                       community.summary AS summary,
                       community.level AS level,
                       score,
                       entities[0..10] AS key_entities
                ORDER BY score DESC
            """

            result = session.run(cypher, {
                "embedding": query_embedding,
                "k": k * 2,
                "session_id": session_id,
            })

            for record in result:
                community = {
                    "id": record["id"],
                    "title": record["title"] or "Untitled",
                    "summary": record["summary"] or "",
                    "level": record["level"],
                    "score": round(record["score"], 4),
                    "key_entities": record["key_entities"] or [],
                }
                results.append(community)

                if len(results) >= k:
                    break

    except Exception as e:
        return json.dumps({"error": f"Neo4j query failed: {str(e)}"})
    finally:
        driver.close()

    return json.dumps({
        "query": query,
        "session_id": session_id,
        "count": len(results),
        "communities": results,
    }, indent=2)


@tool
async def graph_traverse(
    entity_id: str,
    depth: int = 2,
) -> str:
    """Traverse the knowledge graph from a starting entity.

    Explores the graph neighborhood around an entity, following relationships
    to discover connected entities. Use this to understand how concepts
    relate to each other.

    Args:
        entity_id: The ID of the starting entity
        depth: How many hops to traverse (1-3, default: 2)

    Returns:
        JSON string with the subgraph: entities and their relationships
    """
    import json

    session_id = _get_session_id()
    if not session_id:
        return json.dumps({"error": "No session_id available. Set RESEARCH_SESSION_ID env var."})

    # Clamp depth
    depth = max(1, min(depth, 3))

    driver = _get_neo4j_connection()
    entities = []
    relationships = []
    seen_entities = set()
    seen_rels = set()

    try:
        with driver.session() as session:
            # Variable-length path query
            cypher = f"""
                MATCH path = (start:Entity {{id: $entity_id}})-[*1..{depth}]-(end:Entity)
                WHERE $session_id IN start.session_ids
                AND $session_id IN end.session_ids
                RETURN path
            """

            result = session.run(cypher, {
                "entity_id": entity_id,
                "session_id": session_id,
            })

            for record in result:
                path = record["path"]

                # Extract nodes
                for node in path.nodes:
                    node_id = node.element_id
                    if node_id not in seen_entities:
                        seen_entities.add(node_id)
                        entities.append({
                            "id": node.get("id", ""),
                            "type": list(node.labels),
                            "description": node.get("description", ""),
                        })

                # Extract relationships
                for rel in path.relationships:
                    rel_id = rel.element_id
                    if rel_id not in seen_rels:
                        seen_rels.add(rel_id)
                        relationships.append({
                            "type": rel.type,
                            "start": rel.start_node.get("id", ""),
                            "end": rel.end_node.get("id", ""),
                        })

    except Exception as e:
        return json.dumps({"error": f"Neo4j query failed: {str(e)}"})
    finally:
        driver.close()

    return json.dumps({
        "start_entity": entity_id,
        "depth": depth,
        "session_id": session_id,
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "entities": entities[:50],  # Limit output size
        "relationships": relationships[:50],
    }, indent=2)


@tool
async def graph_get_entity_context(
    entity_ids: List[str],
) -> str:
    """Get full context for specific entities including their relationships.

    Given a list of entity IDs, returns detailed information about each entity
    and how they relate to each other. Useful for understanding specific
    entities mentioned in synthesis.

    Args:
        entity_ids: List of entity IDs to look up

    Returns:
        JSON string with entity details and their interconnections
    """
    import json

    session_id = _get_session_id()
    if not session_id:
        return json.dumps({"error": "No session_id available. Set RESEARCH_SESSION_ID env var."})

    if not entity_ids:
        return json.dumps({"error": "No entity_ids provided"})

    driver = _get_neo4j_connection()
    entities = []
    relationships = []

    try:
        with driver.session() as session:
            # Get entities
            cypher = """
                MATCH (e:Entity)
                WHERE e.id IN $entity_ids
                AND $session_id IN e.session_ids
                OPTIONAL MATCH (e)-[r]-(related:Entity)
                WHERE $session_id IN related.session_ids
                AND related.id IN $entity_ids
                WITH e, collect(DISTINCT {
                    type: type(r),
                    direction: CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END,
                    other_entity: related.id
                }) AS rels
                RETURN e.id AS id, e.type AS type, e.description AS description, rels
            """

            result = session.run(cypher, {
                "entity_ids": entity_ids,
                "session_id": session_id,
            })

            for record in result:
                entities.append({
                    "id": record["id"],
                    "type": record["type"],
                    "description": record["description"] or "",
                    "relationships": record["rels"] if record["rels"] else [],
                })

    except Exception as e:
        return json.dumps({"error": f"Neo4j query failed: {str(e)}"})
    finally:
        driver.close()

    return json.dumps({
        "session_id": session_id,
        "requested_ids": entity_ids,
        "found_count": len(entities),
        "entities": entities,
    }, indent=2)


# =============================================================================
# TOOL REGISTRATION
# =============================================================================

# List of all graph tools for easy import
GRAPH_SEARCH_TOOLS = [
    graph_entity_search,
    graph_community_search,
    graph_traverse,
    graph_get_entity_context,
]


__all__ = [
    "graph_entity_search",
    "graph_community_search",
    "graph_traverse",
    "graph_get_entity_context",
    "GRAPH_SEARCH_TOOLS",
]
