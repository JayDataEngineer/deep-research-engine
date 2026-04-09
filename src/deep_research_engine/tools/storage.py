"""Tools for interacting with the project datastore."""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..storage.backends.neo4j import Neo4jStorage
from ..storage.backends.postgres import PostgresStorage
from ..utils.logging import get_loguru_logger

logger = get_loguru_logger("tools.storage")

# Singleton instances for storage backends
_postgres_storage: PostgresStorage | None = None
_neo4j_storage: Neo4jStorage | None = None
_storage_config: dict[str, Any] | None = None


def initialize_storage_tools(config: dict[str, Any]) -> None:
    """Initialize storage tools with configuration.

    Args:
        config: Workflow configuration dictionary.
    """
    global _postgres_storage, _neo4j_storage, _storage_config
    _storage_config = config

    pg_config = config.get("storage", {}).get("postgres", {})
    if pg_config.get("enabled", False):
        try:
            _postgres_storage = PostgresStorage(pg_config)
            logger.info("Storage tools initialized (lazy PostgreSQL)")
        except Exception as e:
            logger.error(f"Failed to setup PostgreSQL storage tools: {e}")

    neo4j_config = config.get("storage", {}).get("neo4j", {})
    if neo4j_config.get("enabled", False):
        try:
            _neo4j_storage = Neo4jStorage(neo4j_config)
            logger.info("Storage tools initialized (lazy Neo4j)")
        except Exception as e:
            logger.error(f"Failed to setup Neo4j storage tools: {e}")


async def _get_storage() -> PostgresStorage | None:
    """Get the initialized PostgreSQL storage backend."""
    global _postgres_storage
    if _postgres_storage is not None and not _postgres_storage.initialized:
        await _postgres_storage.initialize()
    return _postgres_storage


async def _get_neo4j_storage() -> Neo4jStorage | None:
    """Get the initialized Neo4j storage backend."""
    global _neo4j_storage
    if _neo4j_storage is not None and not _neo4j_storage.initialized:
        await _neo4j_storage.initialize()
    return _neo4j_storage


# ---------------------------------------------------------------------------
# Shared helper (replaces duplicated _try_init_neo4j across nodes)
# ---------------------------------------------------------------------------


async def get_neo4j_from_config(config_data: dict[str, Any]) -> Neo4jStorage | None:
    """Create, initialize, and return a ``Neo4jStorage`` from workflow config.

    Returns *None* if Neo4j is disabled or initialization fails.
    Nodes should call this instead of duplicating the init pattern.
    """
    neo4j_config = config_data.get("storage", {}).get("neo4j", {})
    if not neo4j_config.get("enabled"):
        return None
    try:
        storage = Neo4jStorage(neo4j_config)
        await storage.initialize()
        return storage
    except Exception as e:
        logger.warning(f"Neo4j init failed (non-fatal): {e}")
        return None


class VectorSearchInput(BaseModel):
    """Input for vector search tool."""

    query: str = Field(..., description="Semantic search query for the datastore")
    namespace: str = Field(
        default="default", description="Namespace to search in (e.g., 'messages', 'files')"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to return (default 5, max 20)"
    )


@tool(args_schema=VectorSearchInput)
async def vector_search_datastore(query: str, namespace: str = "default", top_k: int = 5) -> str:
    """Search the project's persistent datastore using semantic similarity.

    Retrieves relevant items from the PostgreSQL/pgvector datastore based on
    the provided query. Use this to find information previously ingested or
    stored during the research process.

    Args:
        query: The search query text.
        namespace: The namespace to search in (default: 'default').
        top_k: Maximum number of results to return.

    Returns:
        A formatted string containing the search results with scores and citations.
    """
    storage = await _get_storage()
    if storage is None:
        return "Error: PostgreSQL datastore is not configured or enabled."

    try:
        # 1. Vectorize the query using the shared embeddings factory
        from src.shared.embeddings.factory import get_embeddings

        # Get embedding config from stored workflow config
        embed_config = {}
        if _storage_config:
            embed_config = _storage_config.get("llm", {})

        model_name = embed_config.get("models", {}).get("embedding", "text-embedding-3-small")

        logger.debug(f"Vectorizing query for datastore search: '{query}' using {model_name}")
        embeddings = get_embeddings(model=model_name)
        query_vector = await embeddings.aembed_query(query)

        # 2. Perform vector query
        results = await storage.query(
            query=query_vector, namespace=namespace, limit=top_k, embedding_query=True
        )

        if not results:
            return f"No results found in datastore namespace '{namespace}' for query: '{query}'"

        # 3. Format results
        output = [f"Found {len(results)} relevant items in '{namespace}':\n"]
        for i, res in enumerate(results, 1):
            score = res.get("similarity", 0)
            content = res.get("content") or res.get("text") or str(res.get("data", res))
            # Truncate content for the agent
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"

            metadata = res.get("metadata", {})
            source = metadata.get("source", "Unknown")

            output.append(f"[{i}] Score: {score:.3f} | Source: {source}")
            output.append(content)
            output.append("")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Vector search tool failed: {e}")
        return f"Error performing vector search: {e!s}"


class GraphQueryInput(BaseModel):
    """Input for knowledge graph query tool."""

    query: str = Field(
        ...,
        description="What to find in the knowledge graph (entity name, relationship, or text search)",
    )
    entity_name: str | None = Field(
        default=None, description="Specific entity name to look up and traverse neighbors"
    )
    relationship_type: str | None = Field(
        default=None, description="Filter by relationship type (e.g., 'WORKS_AT', 'SPOKE_IN')"
    )
    depth: int = Field(
        default=2, ge=1, le=4, description="Traversal depth for neighbor queries (1-4, default 2)"
    )
    search_type: str = Field(
        default="text",
        description="Search mode: 'text' (CONTAINS search), 'vector' (semantic similarity), 'entity' (neighbor traversal), 'relationship' (filter by rel type)",
    )


@tool(args_schema=GraphQueryInput)
async def graph_query(
    query: str,
    entity_name: str | None = None,
    relationship_type: str | None = None,
    depth: int = 2,
    search_type: str = "text",
) -> str:
    """Query the knowledge graph to find entities and relationships.

    Use this tool to explore the knowledge graph built from ingested data.
    You can look up specific entities, find relationships, search by text,
    or perform semantic vector search.

    Args:
        query: What to find in the knowledge graph.
        entity_name: Specific entity to look up (traverses neighbors).
        relationship_type: Filter by relationship type.
        depth: Traversal depth (1-4).
        search_type: Search mode - 'text', 'vector', 'entity', or 'relationship'.

    Returns:
        Formatted string with graph query results.
    """
    storage = await _get_neo4j_storage()
    if storage is None:
        return "Error: Neo4j knowledge graph is not configured or enabled."

    namespace = "default"
    if _storage_config:
        namespace = _storage_config.get("storage", {}).get("neo4j", {}).get("namespace", "default")

    try:
        # Route 1: Entity name lookup with neighbor traversal
        if entity_name or search_type == "entity":
            name = entity_name or query
            neighbors = await storage.get_neighbors_by_name(
                name=name, namespace=namespace, depth=depth
            )
            if not neighbors:
                return f"No entity named '{name}' found in the knowledge graph."

            output = [f"Neighbors of '{name}' (depth={depth}):\n"]
            for i, neighbor in enumerate(neighbors, 1):
                labels = neighbor.get("labels", [])
                n_name = neighbor.get("name", "Unknown")
                rel_types = neighbor.get("relationship_types", [])
                output.append(f"[{i}] {':'.join(labels)} - {n_name}")
                if rel_types:
                    output.append(f"    via: {', '.join(rel_types)}")
                for k, v in neighbor.items():
                    if (
                        k
                        not in (
                            "labels",
                            "relationship_types",
                            "namespace",
                            "updated_at",
                            "created_at",
                        )
                        and v
                    ):
                        val_str = str(v)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        output.append(f"    {k}: {val_str}")
                output.append("")

            return "\n".join(output)

        # Route 2: Relationship type filter
        if relationship_type or search_type == "relationship":
            rel = relationship_type or query
            safe_rel = Neo4jStorage._sanitize_label(rel)
            results = await storage.execute_cypher(
                f"MATCH (n)-[r:{safe_rel}]-(m) "
                f"WHERE n.namespace = $namespace "
                f"RETURN n.name as source, labels(n) as source_labels, "
                f"type(r) as rel_type, m.name as target, labels(m) as target_labels "
                f"LIMIT 25",
                {"namespace": namespace},
            )
            if not results or (len(results) == 1 and "error" in results[0]):
                return f"No relationships of type '{rel}' found."

            output = [f"Relationships of type '{rel}':\n"]
            for i, rec in enumerate(results, 1):
                src = rec.get("source", "?")
                tgt = rec.get("target", "?")
                src_labels = rec.get("source_labels", [])
                tgt_labels = rec.get("target_labels", [])
                output.append(
                    f"[{i}] ({':'.join(src_labels)}) {src} --[{rel}]--> ({':'.join(tgt_labels)}) {tgt}"
                )

            return "\n".join(output)

        # Route 3: Vector similarity search
        if search_type == "vector":
            try:
                from ..shared.embeddings import get_embeddings

                embed_config = {}
                if _storage_config:
                    embed_config = _storage_config.get("llm", {})
                model_name = embed_config.get("models", {}).get("embedding", "text-embedding-3-small")
                embeddings = get_embeddings(model=model_name)

                results = await storage.search_similar(
                    query=query,
                    embedding_model=embeddings,
                    k=5,
                )
                if not results:
                    return f"No vector search results for: '{query}'"

                output = [f"Vector search results for '{query}':\n"]
                for i, res in enumerate(results, 1):
                    score = res.get("score", 0)
                    text = res.get("text", "")
                    metadata = res.get("metadata", {})
                    if len(text) > 500:
                        text = text[:500] + "..."
                    output.append(f"[{i}] Score: {score:.3f}")
                    output.append(f"    {text}")
                    if metadata:
                        for k, v in metadata.items():
                            if v and k not in ("namespace", "updated_at"):
                                output.append(f"    {k}: {v}")
                    output.append("")

                return "\n".join(output)
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text search: {e}")

        # Route 4: Text search across node properties (default)
        results = await storage.execute_cypher(
            "MATCH (n {namespace: $namespace}) "
            "WHERE ANY(k IN keys(n) WHERE toString(n[k]) CONTAINS $query) "
            "RETURN n, labels(n) as node_labels LIMIT 15",
            {"namespace": namespace, "query": query},
        )
        if not results or (len(results) == 1 and "error" in results[0]):
            return f"No results found in knowledge graph for: '{query}'"

        output = [f"Knowledge graph search results for '{query}':\n"]
        for i, rec in enumerate(results, 1):
            node = rec.get("n", rec)
            if hasattr(node, "items"):
                node = dict(node)
            node_labels = rec.get("node_labels", [])
            name = node.get("name", "Unknown")
            output.append(f"[{i}] {':'.join(node_labels)} - {name}")
            for k, v in node.items():
                if k not in ("namespace", "updated_at", "created_at") and v:
                    val_str = str(v)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    output.append(f"    {k}: {val_str}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Graph query tool failed: {e}")
        return f"Error querying knowledge graph: {e!s}"
