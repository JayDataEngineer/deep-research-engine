"""LangGraph Store factory for vectorization.

Provides semantic search over ingested documents and research findings.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..config import WorkflowConfig
from ..utils.logging import get_loguru_logger

logger = get_loguru_logger("graph.store")

# Default embedding dimensions for common models
EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}


@asynccontextmanager
async def get_store(config: WorkflowConfig) -> AsyncIterator:
    """Get or create a LangGraph Store for vectorization.

    Args:
        config: Workflow configuration with storage and LLM settings

    Yields:
        AsyncPostgresStore instance or None if not configured
    """
    pg_config = config.storage.get("postgres", {})

    if not pg_config.get("enabled", False) or not pg_config.get("vector_store", False):
        logger.info("Postgres vector store not enabled")
        yield None
        return

    postgres_url = pg_config.get("url")
    if not postgres_url:
        logger.warning("Postgres URL not configured for vector store")
        yield None
        return

    # Get embedding model config
    llm_config = config.llm
    embedding_model = llm_config.get("models", {}).get("embedding", "text-embedding-3-small")

    # Determine embedding dimensions
    dims = EMBEDDING_DIMS.get(embedding_model, 1536)

    try:
        from langchain.embeddings import init_embeddings
        from langgraph.store.postgres import AsyncPostgresStore

        logger.info(f"Initializing Postgres vector store with {embedding_model} ({dims} dims)")

        # Create embedding function
        embed = init_embeddings(f"openai:{embedding_model}")

        async with AsyncPostgresStore.from_conn_string(
            postgres_url,
            index={
                "dims": dims,
                "embed": embed,
                "fields": ["text", "content"],
            },
        ) as store:
            await store.setup()
            logger.info("Postgres vector store initialized successfully")
            yield store

    except ImportError as e:
        logger.warning(f"langgraph[postgres] or langchain not installed: {e}")
        yield None
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        yield None
