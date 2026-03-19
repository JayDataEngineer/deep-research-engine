"""LangGraph Store + Checkpointer setup with auto-vectorization."""

import os
from typing import Tuple, Optional

from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

from ..config import POSTGRES_URL, EMBEDDING_MODEL, EMBEDDING_DIMS
from ..logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# EMBEDDINGS (via LiteLLM)
# =============================================================================

def get_embeddings():
    """Get embeddings instance via LiteLLM.

    Uses LiteLLM for embeddings (configured via EMBEDDING_MODEL env).
    """
    from langchain_litellm import LiteLLMEmbeddings

    return LiteLLMEmbeddings(model=EMBEDDING_MODEL)


# =============================================================================
# STORE + CHECKPOINTER
# =============================================================================

_store_instance: Optional[PostgresStore] = None
_checkpointer_instance: Optional[PostgresSaver] = None


async def get_store_and_checkpointer(
    conn_string: Optional[str] = None,
    setup: bool = True,
) -> Tuple[PostgresStore, PostgresSaver]:
    """Create LangGraph Store with auto-vectorization + Checkpointer.

    This is the main entry point for getting store + checkpointer.

    Args:
        conn_string: Postgres connection string (default: from config)
        setup: Whether to run setup (create tables/indexes)

    Returns:
        Tuple of (PostgresStore, PostgresSaver)

    Usage:
        store, checkpointer = await get_store_and_checkpointer()

        # Write (auto-vectorized!)
        await store.aput(
            namespace=("session_123", "memories", "insights"),
            key="finding_001",
            value={"content": "Important research finding..."}
        )

        # Semantic search
        results = await store.asearch(
            ("session_123", "memories"),
            query="research about AI"
        )

        # Use checkpointer for graph state
        graph = builder.compile(checkpointer=checkpointer, store=store)
    """
    global _store_instance, _checkpointer_instance

    if _store_instance is not None and _checkpointer_instance is not None:
        return _store_instance, _checkpointer_instance

    conn_string = conn_string or POSTGRES_URL

    logger.info(f"Setting up LangGraph Store + Checkpointer", conn_string=conn_string[:50] + "...")

    # Get embeddings for auto-vectorization
    embeddings = get_embeddings()

    # Store with auto-vectorization config
    store = PostgresStore(
        conn_string,
        index_config={
            "dims": EMBEDDING_DIMS,
            "embed": embeddings,
            "fields": ["content"],  # Auto-vectorize 'content' field
        }
    )

    # Checkpointer for state persistence
    checkpointer = PostgresSaver.from_conn_string(conn_string)

    if setup:
        await store.setup()
        await checkpointer.setup()
        logger.info("Store + Checkpointer setup complete")

    _store_instance = store
    _checkpointer_instance = checkpointer

    return store, checkpointer


async def get_store(conn_string: Optional[str] = None, setup: bool = True) -> PostgresStore:
    """Get LangGraph Store with auto-vectorization.

    Args:
        conn_string: Postgres connection string
        setup: Whether to run setup

    Returns:
        PostgresStore instance
    """
    store, _ = await get_store_and_checkpointer(conn_string, setup)
    return store


async def get_checkpointer(conn_string: Optional[str] = None, setup: bool = True) -> PostgresSaver:
    """Get LangGraph Checkpointer.

    Args:
        conn_string: Postgres connection string
        setup: Whether to run setup

    Returns:
        PostgresSaver instance
    """
    _, checkpointer = await get_store_and_checkpointer(conn_string, setup)
    return checkpointer


async def close_connections():
    """Close store and checkpointer connections.

    Call this on shutdown.
    """
    global _store_instance, _checkpointer_instance

    if _store_instance:
        # PostgresStore doesn't have explicit close, but clear reference
        _store_instance = None
        logger.info("Store connection closed")

    if _checkpointer_instance:
        _checkpointer_instance = None
        logger.info("Checkpointer connection closed")
