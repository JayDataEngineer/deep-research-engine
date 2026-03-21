"""Semantic search tool with tier filtering.

This module provides LangChain tools for searching the research database
using semantic (vector) search with optional tier filtering.

The integration with the Virtual Filesystem allows retrieval of full content
by reading the VFS path stored in document metadata.
"""

from .logging_config import get_logger
from typing import Literal, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool

logger = get_logger(__name__)


# =============================================================================
# SEMANTIC SEARCH TOOL
# =============================================================================

@tool
def lookup_knowledge(
    query: str,
    filter_tier: Optional[Literal["A", "B", "C"]] = None,
    k: int = 5
) -> str:
    """Search research database with semantic search + tier filtering.

    This tool performs semantic (vector-based) search across all ingested
    research documents, with optional filtering by tier (A/B/C).

    Args:
        query: Semantic concept to search for (e.g., "nietzsche diet", "machine learning")
        filter_tier: Only search specified tier ("A", "B", or "C"). None searches all tiers.
        k: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with:
            - Source URL
            - VFS path (for reading full content)
            - Tier
            - Relevant content chunk

    Example:
        >>> results = lookup_knowledge.invoke({
        ...     "query": "nietzsche's diet",
        ...     "filter_tier": "A",
        ...     "k": 3
        ... })
    """
    from .ingestion import get_vector_store

    logger.info(f"[RETRIEVE] Searching for '{query}' (tier={filter_tier}, k={k})")

    vector_store = get_vector_store()

    # Build metadata filter
    filter_dict = {"tier": filter_tier} if filter_tier else None

    # Perform semantic search
    results = vector_store.similarity_search(
        query,
        k=k,
        filter=filter_dict
    )

    if not results:
        return f"No results found for query: '{query}'"

    logger.debug(f"[RETRIEVE] Found {len(results)} results")

    # Format output with VFS paths
    output_parts = []
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        source_url = meta.get("source_url", "Unknown")
        vfs_path = meta.get("vfs_path", "N/A")
        tier = meta.get("tier", "unknown")

        output_parts.append(
            f"**[{i}] {source_url}**\n"
            f"   [VFS: {vfs_path}]\n"
            f"   [Tier: {tier}]\n"
            f"   \n"
            f"   {doc.page_content}\n"
        )

    return "\n".join(output_parts)


# =============================================================================
# SEMANTIC SEARCH WITH FULL CONTENT (VFS READ)
# =============================================================================

@tool
def lookup_knowledge_full(
    query: str,
    filter_tier: Optional[Literal["A", "B", "C"]] = None,
    k: int = 3,
    max_content_length: int = 2000
) -> str:
    """Search research database and return full content from VFS.

    This tool performs semantic search and then reads the full content
    from the Virtual Filesystem for each result.

    Args:
        query: Semantic concept to search for
        filter_tier: Only search specified tier
        k: Maximum number of results
        max_content_length: Max characters to return per document

    Returns:
        Formatted search results with full content from VFS

    Example:
        >>> results = lookup_knowledge_full.invoke({
        ...     "query": "nietzsche",
        ...     "k": 2
        ... })
    """
    from .ingestion import get_vector_store

    logger.info(f"[RETRIEVE] Full content search for '{query}' (k={k})")

    vector_store = get_vector_store()
    filter_dict = {"tier": filter_tier} if filter_tier else None

    # Perform semantic search
    results = vector_store.similarity_search(
        query,
        k=k,
        filter=filter_dict
    )

    if not results:
        return f"No results found for query: '{query}'"

    output_parts = []

    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        source_url = meta.get("source_url", "Unknown")
        vfs_path = meta.get("vfs_path")
        tier = meta.get("tier", "unknown")

        output_parts.append(f"**[{i}] {source_url}** [Tier: {tier}]")

        # Try to read full content from VFS
        if vfs_path:
            try:
                from deepagents.storage import get_filesystem_from_context

                filesystem = get_filesystem_from_context()
                if filesystem is not None:
                    full_content = filesystem.read(vfs_path)
                    # Truncate if needed
                    if len(full_content) > max_content_length:
                        full_content = full_content[:max_content_length] + "\n\n... (truncated)"
                    output_parts.append(f"\n{full_content}\n")
                else:
                    output_parts.append(f"\n(Chunk preview: {doc.page_content[:500]}...)\n")
            except ImportError:
                output_parts.append(f"\n(Chunk preview: {doc.page_content[:500]}...)\n")
            except Exception as e:
                logger.warning(f"[RETRIEVE] Failed to read VFS {vfs_path}: {e}")
                output_parts.append(f"\n(Chunk preview: {doc.page_content[:500]}...)\n")
        else:
            output_parts.append(f"\n{doc.page_content}\n")

    return "\n".join(output_parts)


# =============================================================================
# TIER-SPECIFIC SEARCH HELPERS
# =============================================================================

@tool
def lookup_tier_a(query: str, k: int = 5) -> str:
    """Search only Tier A (highest quality) research documents.

    Tier A documents are explicitly requested sources and have the highest
    relevance and quality.

    Args:
        query: Semantic concept to search for
        k: Maximum number of results

    Returns:
        Formatted search results from Tier A only

    Example:
        >>> results = lookup_tier_a.invoke({"query": "machine learning", "k": 3})
    """
    return lookup_knowledge.invoke({"query": query, "filter_tier": "A", "k": k})


@tool
def lookup_tier_b(query: str, k: int = 5) -> str:
    """Search only Tier B research documents.

    Tier B documents are from search results and have medium relevance.

    Args:
        query: Semantic concept to search for
        k: Maximum number of results

    Returns:
        Formatted search results from Tier B only

    Example:
        >>> results = lookup_tier_b.invoke({"query": "nlp", "k": 3})
    """
    return lookup_knowledge.invoke({"query": query, "filter_tier": "B", "k": k})


@tool
def lookup_tier_c(query: str, k: int = 5) -> str:
    """Search only Tier C (lowest priority) research documents.

    Tier C documents are from exploratory search and may have lower relevance.

    Args:
        query: Semantic concept to search for
        k: Maximum number of results

    Returns:
        Formatted search results from Tier C only

    Example:
        >>> results = lookup_tier_c.invoke({"query": "exploratory", "k": 3})
    """
    return lookup_knowledge.invoke({"query": query, "filter_tier": "C", "k": k})


# =============================================================================
# LIST ALL INGESTED DOCUMENTS
# =============================================================================

@tool
def list_ingested_documents(
    filter_tier: Optional[Literal["A", "B", "C"]] = None,
    limit: int = 50
) -> str:
    """List all ingested documents in the research database.

    This tool returns a summary of all documents that have been ingested,
    optionally filtered by tier.

    Args:
        filter_tier: Only list documents from specified tier
        limit: Maximum number of documents to list

    Returns:
        Formatted list of ingested documents with metadata

    Example:
        >>> docs = list_ingested_documents.invoke({"filter_tier": "A"})
    """
    from .ingestion import get_vector_store

    logger.info(f"[RETRIEVE] Listing documents (tier={filter_tier}, limit={limit})")

    vector_store = get_vector_store()

    # The vector store doesn't have a simple "list all" method
    # We need to use a workaround - search with a very broad query
    # or store metadata separately

    # For now, return a placeholder message
    # In production, we would implement a proper document registry
    return (
        "Document listing not yet implemented. "
        "Use lookup_knowledge with a relevant query to find documents. "
        f"Vector store is initialized with {len(vector_store.store)} embeddings."
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "lookup_knowledge",
    "lookup_knowledge_full",
    "lookup_tier_a",
    "lookup_tier_b",
    "lookup_tier_c",
    "list_ingested_documents",
]
