"""Tools for interacting with the project datastore."""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..storage.backends.postgres import PostgresStorage
from ..utils.logging import get_loguru_logger

logger = get_loguru_logger("tools.storage")

# Singleton instance for the storage backend
_postgres_storage: Optional[PostgresStorage] = None
_storage_config: Optional[Dict[str, Any]] = None


def initialize_storage_tools(config: Dict[str, Any]) -> None:
    """Initialize storage tools with configuration.

    Args:
        config: Workflow configuration dictionary.
    """
    global _postgres_storage, _storage_config
    _storage_config = config

    pg_config = config.get("storage", {}).get("postgres", {})
    if not pg_config.get("enabled", False):
        logger.debug("PostgreSQL storage not enabled for tools")
        return

    try:
        _postgres_storage = PostgresStorage(pg_config)
        # Note: We don't initialize here to avoid blocking; 
        # it will be initialized lazily on first use.
        logger.info("Storage tools initialized (lazy PostgreSQL)")
    except Exception as e:
        logger.error(f"Failed to setup storage tools: {e}")


async def _get_storage() -> Optional[PostgresStorage]:
    """Get the initialized PostgreSQL storage backend."""
    global _postgres_storage
    if _postgres_storage is not None and not _postgres_storage.initialized:
        await _postgres_storage.initialize()
    return _postgres_storage


class VectorSearchInput(BaseModel):
    """Input for vector search tool."""
    query: str = Field(
        ..., 
        description="Semantic search query for the datastore"
    )
    namespace: str = Field(
        default="default",
        description="Namespace to search in (e.g., 'messages', 'files')"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (default 5, max 20)"
    )


@tool(args_schema=VectorSearchInput)
async def vector_search_datastore(
    query: str, 
    namespace: str = "default", 
    top_k: int = 5
) -> str:
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
            query=query_vector,
            namespace=namespace,
            limit=top_k,
            embedding_query=True
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
        return f"Error performing vector search: {str(e)}"
