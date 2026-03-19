"""
search_memories tool for semantic search over stored artifacts.

Uses LangGraph's Store.asearch() for semantic similarity search.
Searches across all auto-vectorized content:
- /memories/dumps/ (context overflow)
- /memories/web/ (webpage content)
- /memories/insights/ (worker findings)
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from langchain_core.tools import tool
from src.shared.logging_config import get_logger

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = get_logger(__name__)


@tool
async def search_memories(
    query: str,
    subcategory: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search ALL auto-vectorized memories using semantic similarity.

    Use this to find information from:
    - Context dumps (when agents ran out of context)
    - Webpage content (scraped by web agent)
    - Worker insights (findings from flash explorers)

    Searches by MEANING, not keywords. Returns most relevant matches.

    Args:
        query: Natural language search query (e.g., "what did we find about APIs?")
        subcategory: Optional filter - "dumps", "web", "insights"
        limit: Maximum results to return (default 10)

    Returns:
        Matching memories with relevance scores and content previews
    """
    from langchain_core.globals import get_config

    try:
        # Get store from context
        config = get_config()
        store = config.get("configurable", {}).get("__store__")

        if store is None:
            # Fallback: try to get from global context engine
            try:
                from src.api.core.extensions import get_extension_manager

                manager = get_extension_manager()
                if manager:
                    ext = manager.get_extension("context_engine")
                    if ext and hasattr(ext, 'langgraph_store'):
                        store = ext.langgraph_store
            except Exception as e:
                logger.warning(f"[search_memories] Could not get store from extension: {e}")

        if store is None:
            return (
                "Error: search_memories requires a LangGraph Store. "
                "Ensure the agent is created with a store parameter."
            )

        # Get session_id for namespace isolation (research sessions)
        # Priority: configurable.session_id > assistant_id > "default"
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id") or configurable.get("assistant_id", "default")

        # Build namespace prefix - searches ALL subcategories
        # Format: (session_id, "memories") matches:
        #   - (session_id, "memories", "dumps")
        #   - (session_id, "memories", "web")
        #   - (session_id, "memories", "insights")
        namespace_prefix = (session_id, "memories")

        logger.info(f"[search_memories] Searching namespace={namespace_prefix}, query='{query[:50]}...', subcategory={subcategory}")

        # Semantic search using store.asearch()
        # This uses the AUTO-VECTORIZED embeddings
        results = await store.asearch(
            namespace_prefix,
            query=query,
            limit=limit * 2 if subcategory else limit,  # Get more if filtering
        )

        if not results:
            return f"No memories found matching: '{query}'"

        # Format results
        output = [f"Found {len(results)} memories matching '{query}':\n"]

        count = 0
        for item in results:
            if count >= limit:
                break

            # Get content from value
            value = item.value if hasattr(item, 'value') else item
            content = value.get("content", "")

            # Handle content as list (StoreBackend format)
            if isinstance(content, list):
                content_preview = " ".join(str(c) for c in content)[:400]
            else:
                content_preview = str(content)[:400]

            # Get relevance score
            score = getattr(item, 'score', 0.0)
            score_pct = int(score * 100) if score else 0

            # Get subcategory from value or namespace
            item_subcategory = value.get("subcategory", "unknown")
            if item_subcategory == "unknown" and hasattr(item, 'namespace'):
                # Extract from namespace tuple
                ns = item.namespace
                if len(ns) >= 3:
                    item_subcategory = ns[2]

            # Filter by subcategory if specified
            if subcategory and item_subcategory != subcategory:
                continue

            # Get source info
            source = value.get("source", value.get("worker_id", "unknown"))
            url = value.get("url", "")
            title = value.get("title", "")

            # Format output
            count += 1
            output.append(
                f"{count}. [{item_subcategory}] {item.key} (relevance: {score_pct}%)\n"
                f"   Source: {source}\n"
            )
            if url:
                output.append(f"   URL: {url}\n")
            if title:
                output.append(f"   Title: {title}\n")
            output.append(f"   Preview: {content_preview}...\n")

        if count == 0:
            return f"No memories found in subcategory '{subcategory}' matching: '{query}'"

        return "\n".join(output)

    except Exception as e:
        logger.error(f"[search_memories] Error: {e}", exc_info=True)
        return f"Error searching memories: {str(e)}"
