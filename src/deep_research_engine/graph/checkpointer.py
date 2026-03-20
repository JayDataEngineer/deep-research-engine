"""AsyncPostgresSaver factory for LangGraph checkpointing.

Enables resume-from-failure capability by persisting workflow state.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..config import WorkflowConfig
from ..utils.logging import get_loguru_logger

logger = get_loguru_logger("graph.checkpointer")


@asynccontextmanager
async def get_checkpointer(config: WorkflowConfig) -> AsyncIterator:
    """Get or create an AsyncPostgresSaver for checkpointing.

    Args:
        config: Workflow configuration with storage settings

    Yields:
        AsyncPostgresSaver instance or None if not configured
    """
    pg_config = config.storage.get("postgres", {})

    if not pg_config.get("enabled", False) or not pg_config.get("checkpoint", False):
        logger.info("Postgres checkpointing not enabled")
        yield None
        return

    postgres_url = pg_config.get("url")
    if not postgres_url:
        logger.warning("Postgres URL not configured for checkpointing")
        yield None
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        logger.info(
            f"Initializing Postgres checkpointer at {postgres_url.split('@')[-1] if '@' in postgres_url else 'configured'}"
        )

        async with AsyncPostgresSaver.from_conn_string(postgres_url) as checkpointer:
            await checkpointer.setup()
            logger.info("Postgres checkpointer initialized successfully")
            yield checkpointer

    except ImportError:
        logger.warning("langgraph[postgres] not installed. Checkpointing disabled.")
        yield None
    except Exception as e:
        logger.error(f"Failed to initialize checkpointer: {e}")
        yield None
