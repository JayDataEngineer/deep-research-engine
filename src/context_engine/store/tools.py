"""Custom tools for store access - primarily search_vectors."""

from typing import Optional
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

from ..logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# SEARCH VECTORS TOOL
# =============================================================================

def create_search_vectors_tool(
    store: BaseStore,
    session_id: str,
    default_namespace: str = "memories",
):
    """Create search_vectors tool for semantic search.

    This tool enables agents to semantically search the context store.
    All content written via write_file is auto-vectorized, so this
    works immediately without any manual embedding.

    Args:
        store: LangGraph BaseStore instance
        session_id: Current session ID
        default_namespace: Default sub-namespace to search

    Returns:
        search_vectors tool function

    Usage:
        store = await get_store()
        search_vectors = create_search_vectors_tool(store, "session_123")

        agent = create_deep_agent(
            tools=[search_vectors],
            store=store,  # write_file will auto-vectorize
        )
    """

    @tool
    async def search_vectors(
        query: str,
        namespace: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Search context semantically.

        Searches through all stored context using semantic similarity.
        The query is automatically embedded and matched against
        auto-vectorized content from write_file operations.

        Args:
            query: Natural language search query
            namespace: Sub-namespace to search (default: "memories").
                      Options: "memories", "input", "web", "insights"
            limit: Maximum results to return (default: 10)

        Returns:
            Formatted search results with relevance scores

        Example:
            search_vectors("research about AI safety", namespace="memories")
        """
        ns = namespace or default_namespace
        full_namespace = (session_id, ns)

        try:
            results = await store.asearch(
                namespace_prefix=full_namespace,
                query=query,
                limit=limit,
            )

            if not results:
                return f"No results found for query: '{query}' in namespace: {ns}"

            output_lines = [
                f"Found {len(results)} results for: '{query}'",
                f"Namespace: {ns}",
                "",
                "---",
                "",
            ]

            for r in results:
                content = r.value.get("content", "")
                score = getattr(r, "score", 0.0)

                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "..."

                output_lines.extend([
                    f"### [{r.key}] (relevance: {score:.2f})",
                    content,
                    "",
                    "---",
                    "",
                ])

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"search_vectors failed", error=str(e), query=query)
            return f"Search failed: {str(e)}"

    return search_vectors


# =============================================================================
# READ ALL MEMORIES TOOL (for Graph Builder)
# =============================================================================

def create_read_all_memories_tool(
    store: BaseStore,
    session_id: str,
):
    """Create read_all_memories tool for reading ALL context.

    Used by Graph Builder to extract entities from all context.

    Args:
        store: LangGraph BaseStore instance
        session_id: Current session ID

    Returns:
        read_all_memories tool function
    """

    @tool
    async def read_all_memories(
        namespace: Optional[str] = None,
        limit: int = 1000,
    ) -> str:
        """Read ALL memories from context store.

        Unlike search_vectors, this returns all items without
        semantic filtering. Used for entity extraction.

        Args:
            namespace: Sub-namespace to read (default: all under session)
            limit: Maximum items to return (default: 1000)

        Returns:
            All stored items with content and metadata
        """
        if namespace:
            ns_prefix = (session_id, namespace)
        else:
            ns_prefix = (session_id,)

        try:
            # Use search without query to get all items
            results = await store.asearch(
                namespace_prefix=ns_prefix,
                query=None,  # No semantic filter
                limit=limit,
            )

            if not results:
                return f"No memories found in session: {session_id}"

            output_lines = [
                f"Total memories: {len(results)}",
                "",
            ]

            for r in results:
                content = r.value.get("content", "")
                metadata = {k: v for k, v in r.value.items() if k != "content"}

                output_lines.extend([
                    f"## [{r.namespace}/{r.key}]",
                    f"Content: {content[:300]}..." if len(content) > 300 else f"Content: {content}",
                    f"Metadata: {metadata}",
                    "",
                ])

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"read_all_memories failed", error=str(e))
            return f"Read failed: {str(e)}"

    return read_all_memories


# =============================================================================
# LIST NAMESPACES TOOL
# =============================================================================

def create_list_namespaces_tool(
    store: BaseStore,
    session_id: str,
):
    """Create list_namespaces tool for discovering namespaces.

    Args:
        store: LangGraph BaseStore instance
        session_id: Current session ID

    Returns:
        list_namespaces tool function
    """

    @tool
    async def list_namespaces() -> str:
        """List all namespaces in the current session.

        Returns:
            Formatted list of namespaces with item counts
        """
        try:
            # Get all namespaces under session
            namespaces = await store.alist_namespaces(
                prefix=(session_id,),
                limit=100,
            )

            if not namespaces:
                return f"No namespaces found for session: {session_id}"

            output_lines = [
                f"Namespaces for session: {session_id}",
                "",
            ]

            for ns in namespaces:
                # Count items in namespace
                items = await store.asearch(
                    namespace_prefix=ns,
                    query=None,
                    limit=1000,
                )
                output_lines.append(f"  {'/'.join(ns)}: {len(items)} items")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"list_namespaces failed", error=str(e))
            return f"List failed: {str(e)}"

    return list_namespaces
