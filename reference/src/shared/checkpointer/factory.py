"""Checkpointer factory for AsyncPostgresSaver.

SINGLE BEST WAY: AsyncPostgresSaver for everything.
NO FALLBACKS - crash if Postgres is not available.

FACTORY FUNCTIONS ARE ASYNC - callers must await them.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class CheckpointerError(Exception):
    """Raised when checkpointer configuration is invalid."""
    pass


async def get_checkpointer(
    postgres_url: Optional[str] = None,
) -> "BaseCheckpointer":
    """Async factory that returns the AsyncPostgresSaver.

    CRASH-ONLY: Postgres is REQUIRED. NO FALLBACKS.

    Args:
        postgres_url: Optional Postgres connection string. If not provided,
            will attempt to read from POSTGRES_URL or CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING.

    Returns:
        AsyncPostgresSaver instance (fully initialized with tables created)

    Raises:
        CheckpointerError: If Postgres is not available.

    Examples:
        >>> checkpointer = await get_checkpointer()
        >>> graph = builder.compile(checkpointer=checkpointer)
    """
    # ========================================================================
    # POSTGRES CHECKPOINTER (REQUIRED - NO FALLBACKS)
    # ========================================================================
    db_url = postgres_url or os.getenv("POSTGRES_URL") or os.getenv("CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING")

    # Add sslmode=disable for local connections if not already specified
    # Local Postgres servers typically don't have SSL configured
    if db_url and "sslmode" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{separator}sslmode=disable"

    if not db_url:
        raise CheckpointerError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           POSTGRES REQUIRED - NO FALLBACKS                         ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "Postgres checkpointer is REQUIRED.\n"
            "Set one of these environment variables:\n"
            "  - POSTGRES_URL\n"
            "  - CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING\n"
            "\n"
            "Example: postgresql://user:password@localhost:5432/ulli_db\n"
        )

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool
        import psycopg

        # Create connection pool (LangGraph uses pools internally)
        # The pool manages connections and keeps them alive
        pool = AsyncConnectionPool(
            conninfo=db_url,
            min_size=1,
            max_size=10,
            open=False,
        )

        # Open the pool
        await pool.open()

        # Create AsyncPostgresSaver with the pool (LangGraph handles connections from the pool)
        checkpointer = AsyncPostgresSaver(pool)

        # Set up database schema using the checkpointer's setup method
        try:
            await checkpointer.setup()
            logger.info("[CHECKPOINTER] Database schema setup completed")
        except Exception as setup_err:
            # Setup may fail if tables already exist - log at warning level for visibility
            logger.warning(f"[CHECKPOINTER] Setup note (may be OK if tables exist): {setup_err}")

        # Store pool reference to prevent garbage collection
        # This keeps the pool alive for the lifetime of the checkpointer
        checkpointer._pool = pool

        logger.info("[CHECKPOINTER] AsyncPostgresSaver initialized with psycopg pool")
        return checkpointer

    except ImportError as e:
        raise CheckpointerError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           POSTGRES CHECKPOINTER PACKAGE REQUIRED                   ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Error: {e}\n"
            "\n"
            "Install with: uv add langgraph-checkpoint-postgres\n"
        ) from e

    except Exception as e:
        raise CheckpointerError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           POSTGRES CHECKPOINTER INITIALIZATION FAILED              ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Error: {e}\n"
            "\n"
            "Please check:\n"
            "  1. Postgres is running and accessible\n"
            "  2. Connection string is correct\n"
            "  3. Database exists\n"
            "  4. User has proper permissions\n"
        ) from e


# ============================================================================
# POSTGRES STORE (For Long-term Memory / Research Artifacts)
# ============================================================================

async def get_store(
    postgres_url: Optional[str] = None,
    embeddings: Optional[object] = None,
) -> "BaseStore":
    """Async factory for LangGraph Store (long-term memory).

    CRASH-ONLY: PostgresStore is REQUIRED. NO FALLBACKS.

    The Store is used for:
    - Research artifacts and sources
    - Long-term memory across sessions
    - Semantic search with embeddings

    Args:
        postgres_url: Postgres connection string
        embeddings: Embeddings model for semantic search
            (default: HuggingFace sentence-transformers)

    Returns:
        PostgresStore instance (fully initialized)

    Raises:
        CheckpointerError: If Store cannot be initialized

    Examples:
        >>> store = await get_store()
        >>> await store.put(namespace, key, value)
    """
    try:
        from langgraph.store.postgres import PostgresStore
    except ImportError as e:
        raise CheckpointerError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           POSTGRES STORE PACKAGE REQUIRED                          ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Error: {e}\n"
            "\n"
            "Install with: uv add langgraph-store-postgres\n"
        ) from e

    db_url = postgres_url or os.getenv("POSTGRES_URL")
    if not db_url:
        raise CheckpointerError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           POSTGRES STORE REQUIRED - NO FALLBACKS                   ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "Set POSTGRES_URL environment variable.\n"
        )

    # Initialize embeddings if not provided (REQUIRED for semantic search)
    if embeddings is None:
        try:
            from .embeddings.factory import get_embeddings
            embeddings = get_embeddings()
        except Exception as e:
            raise CheckpointerError(
                "╔═══════════════════════════════════════════════════════════════════╗\n"
                "║           EMBEDDINGS REQUIRED FOR STORE                           ║\n"
                "╚═══════════════════════════════════════════════════════════════════╝\n"
                "\n"
                f"Error: {e}\n"
            ) from e

    # Get embedding dimensions by embedding a test string
    try:
        test_embedding = embeddings.embed_query("test")
        dims = len(test_embedding)
    except Exception as e:
        raise CheckpointerError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           EMBEDDING DIMS DETECTION FAILED                         ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Error: {e}\n"
        ) from e

    # Create store with semantic search
    # Note: from_conn_string returns a context manager, we enter it to get the store
    store_ctx = PostgresStore.from_conn_string(
        db_url,
        index={
            "dims": dims,  # Explicit dimensions required
            "embed": embeddings,
            "fields": ["content"],  # Fields to vectorize
        }
    )

    # Enter the context manager to get the actual store
    store = store_ctx.__enter__()

    # Initialize tables - CRASH if setup fails (setup is SYNC, not async)
    try:
        store.setup()
        logger.info("[STORE] PostgresStore initialized with semantic search")
    except Exception as e:
        store_ctx.__exit__(type(e), e, e.__traceback__)
        raise CheckpointerError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           POSTGRES STORE SETUP FAILED                               ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Error: {e}\n"
            "\n"
            "Please check:\n"
            "  1. Postgres is running and accessible\n"
            "  2. Database exists\n"
            "  3. User has proper permissions\n"
        ) from e

    return store


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

async def get_shared_checkpointer():
    """Convenience function for backward compatibility.

    This function maintains backward compatibility with existing code:
        from .checkpointer import get_shared_checkpointer

    Now uses the async factory pattern.

    Returns:
        AsyncPostgresSaver instance or MemorySaver for testing
    """
    # Allow memory checkpointer for testing
    if os.getenv("USE_MEMORY_CHECKPOINTER", "").lower() == "true":
        logger.info("[CHECKPOINTER] Using MemorySaver (testing mode)")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    return await get_checkpointer()


__all__ = [
    "get_checkpointer",
    "get_store",
    "get_shared_checkpointer",
    "CheckpointerError",
]
