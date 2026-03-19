"""Deep Research Extension Builder.

Builds the deep research graph for the extension.
All logic is self-contained within this extension.
"""

from pathlib import Path
import logging
import sys


from typing import Any

# Local imports - extension is self-contained
from .graph import build_v4_graph
from src.shared.checkpointer import get_shared_checkpointer
from .logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level

# Cache the graph
_graph_cache = None


async def build_research_graph(
    use_case: str = "rag",
    variant: str = "default"
) -> Any:
    """Build the deep research graph.

    This wraps the existing Spawner pipeline for the registry.

    Args:
        use_case: Research domain (rag, podcast, trading, osint, news, content)
        variant: Config variant (default, or custom)

    Returns:
        Compiled StateGraph for deep research mode
    """
    # Clear cache on build to ensure fresh graph with checkpointer
    global _graph_cache
    _graph_cache = None

    logger.info(f"[DEEP_RESEARCH] Building graph: use_case={use_case}, variant={variant}")

    # Get shared checkpointer for conversation persistence
    checkpointer = await get_shared_checkpointer()
    logger.info("[DEEP_RESEARCH] Got shared checkpointer for persistence")

    # Get LangGraph PostgresStore for memory persistence (deep_agent findings)
    # This replaces the default InMemoryStore with persistent Postgres + pgvector storage
    try:
        from src.extensions.features.tools.context_engine.storage.langgraph_store import get_langgraph_store
        store = await get_langgraph_store()
        logger.info("[DEEP_RESEARCH] Got LangGraphPostgresStore for memory persistence")
    except Exception as e:
        logger.warning(f"[DEEP_RESEARCH] Could not get LangGraphPostgresStore: {e}. Using InMemoryStore fallback.")
        store = None

    # Build graph with checkpointer and store for persistence
    graph = build_v4_graph(checkpointer=checkpointer, store=store)

    logger.info("[DEEP_RESEARCH] Graph built successfully with checkpointer and store")

    return graph
