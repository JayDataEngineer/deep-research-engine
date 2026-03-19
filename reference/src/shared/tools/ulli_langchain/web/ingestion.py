"""Smart URL ingestion using our scraper + LangChain downstream components.

This module combines our CAPTCHA-bypass web scraper with LangChain's
battle-tested text splitting, embeddings, and vector storage components.

Hybrid Architecture:
- Scraping: Our custom web_get tool (CAPTCHA bypass, markdown conversion)
- Chunking: LangChain RecursiveCharacterTextSplitter
- Embeddings: Factory-based embeddings (OpenAI, local, ZAI, etc.)
- Vector Storage: LangChain InMemoryVectorStore (or Chroma/Pinecone)
"""

import hashlib
from src.shared.logging_config import get_logger
from typing import Dict, Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)

# =============================================================================
# SINGLETON: Vector Store (embeddings now from factory)
# =============================================================================

_vector_store: Optional[InMemoryVectorStore] = None


def get_vector_store() -> InMemoryVectorStore:
    """Get or initialize vector store singleton.

    Returns:
        InMemoryVectorStore instance

    Example:
        >>> store = get_vector_store()
        >>> results = store.similarity_search("query", k=5)
    """
    global _vector_store
    if _vector_store is None:
        logger.info("[INGEST] Initializing InMemoryVectorStore")
        from src.shared.embeddings.factory import get_embeddings
        _vector_store = InMemoryVectorStore(get_embeddings())
    return _vector_store


def _generate_source_id(url: str) -> str:
    """Generate a unique source ID from URL.

    Args:
        url: URL to hash

    Returns:
        12-character hex string

    Example:
        >>> _generate_source_id("https://example.com")
        '5a105e8b9d40e132'
    """
    return hashlib.md5(url.encode()).hexdigest()[:12]


# =============================================================================
# MAIN INGESTION FUNCTION
# =============================================================================

async def ingest_url(
    url: str,
    tier: Literal["A", "B", "C"] = "B",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_to_vfs: bool = True,
    vfs_path_override: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest URL using our scraper + LangChain downstream processing.

    This function:
    1. Scrapes the URL using our custom web_get tool (CAPTCHA bypass)
    2. Chunks the content using LangChain's RecursiveCharacterTextSplitter
    3. Embeds chunks using LangChain's OpenAIEmbeddings
    4. Stores embeddings in vector store for semantic search
    5. Optionally saves full content to VFS for large file retrieval

    Args:
        url: URL to ingest
        tier: Information Funnel tier ("A", "B", or "C")
        chunk_size: Text splitter chunk size in characters
        chunk_overlap: Text splitter overlap in characters
        save_to_vfs: Whether to save full content to virtual filesystem
        vfs_path_override: Optional custom VFS path (auto-generated if None)

    Returns:
        Dict with ingestion results:
            - url: Original URL
            - tier: Assigned tier
            - chunks: Number of chunks created
            - vfs_path: Path to full content in VFS
            - source_id: Unique source identifier

    Raises:
        Exception: If scraping fails or URL is invalid

    Example:
        >>> result = await ingest_url("https://example.com", tier="A")
        >>> print(f"Ingested {result['chunks']} chunks to {result['vfs_path']}")
    """
    from . import web_get as web_get_tool

    logger.info(f"[INGEST] Starting ingestion for {url} (tier={tier})")

    # -------------------------------------------------------------------------
    # A. Use OUR custom scraper (handles CAPTCHA, converts to markdown)
    # -------------------------------------------------------------------------
    logger.debug(f"[INGEST] Scraping {url} with web_get...")
    scrape_result = await web_get_tool.ainvoke({"url": url})

    # Extract content from result (format: "Content: ...\\n\\nSource: ...")
    if isinstance(scrape_result, dict):
        content = scrape_result.get("content", scrape_result)
    else:
        content = scrape_result

    # If content has "Source:" marker, extract just the content part
    if "\n\nSource:" in str(content):
        content = str(content).split("\n\nSource:")[0].strip()
    elif "Content:" in str(content):
        content = str(content).split("Content:")[1].split("\n\nSource:")[0].strip()

    if not content or len(content) < 50:
        raise ValueError(f"Scraped content too short or empty for {url}")

    logger.debug(f"[INGEST] Scraped {len(content)} characters from {url}")

    # -------------------------------------------------------------------------
    # B. Use LangChain TextSplitter (smart paragraph/newline handling)
    # -------------------------------------------------------------------------
    logger.debug("[INGEST] Splitting content with RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        length_function=len,
    )

    # Create Document from our scraped content
    doc = Document(page_content=content, metadata={"source_url": url})
    splits = text_splitter.split_documents([doc])

    logger.debug(f"[INGEST] Created {len(splits)} chunks")

    # -------------------------------------------------------------------------
    # C. Generate IDs and paths
    # -------------------------------------------------------------------------
    source_id = _generate_source_id(url)

    if vfs_path_override:
        vfs_path = vfs_path_override
    else:
        vfs_path = f"/research/tier_{tier.lower()}/{source_id}.md"

    # -------------------------------------------------------------------------
    # D. Inject metadata with tiering logic
    # -------------------------------------------------------------------------
    for split in splits:
        split.metadata.update({
            "tier": tier,
            "source_url": url,
            "vfs_path": vfs_path,
            "source_id": source_id,
        })

        # For Tier C, tag as low priority
        if tier == "C":
            split.metadata["priority"] = "low"

    # -------------------------------------------------------------------------
    # E. Add to vector store (handles embedding & indexing)
    # -------------------------------------------------------------------------
    logger.debug("[INGEST] Adding chunks to vector store...")
    get_vector_store().add_documents(splits)
    logger.info(f"[INGEST] Added {len(splits)} chunks to vector store")

    # -------------------------------------------------------------------------
    # F. Optionally save full content to VFS
    # -------------------------------------------------------------------------
    if save_to_vfs:
        from deepagents.storage import get_filesystem_from_context

        filesystem = get_filesystem_from_context()
        if filesystem is None:
            raise RuntimeError(
                "[INGEST] VFS save requested but no filesystem context available. "
                "Ensure deepagents.storage is properly initialized."
            )

        filesystem.write(vfs_path, content)
        logger.debug(f"[INGEST] Saved full content to VFS: {vfs_path}")

    return {
        "url": url,
        "tier": tier,
        "chunks": len(splits),
        "vfs_path": vfs_path,
        "source_id": source_id,
        "content_length": len(content),
        "content": content,  # Return actual content for database storage
    }


# =============================================================================
# BATCH INGESTION
# =============================================================================

async def ingest_urls(
    urls: list[str],
    tier: Literal["A", "B", "C"] = "B",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_concurrency: int = 5
) -> list[Dict[str, Any]]:
    """Ingest multiple URLs concurrently.

    Args:
        urls: List of URLs to ingest
        tier: Information Funnel tier for all URLs
        chunk_size: Text splitter chunk size
        chunk_overlap: Text splitter overlap
        max_concurrency: Max concurrent ingestions

    Returns:
        List of ingestion result dicts

    Example:
        >>> results = await ingest_urls(
        ...     ["https://example.com", "https://example.org"],
        ...     tier="A"
        ... )
        >>> print(f"Ingested {len(results)} URLs")
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrency)

    async def ingest_with_limit(url: str) -> Dict[str, Any]:
        async with semaphore:
            return await ingest_url(url, tier=tier, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info(f"[INGEST] Batch ingesting {len(urls)} URLs (concurrency={max_concurrency})")
    results = await asyncio.gather(*[ingest_with_limit(url) for url in urls])

    successful = sum(1 for r in results if r.get("chunks", 0) > 0)
    logger.info(f"[INGEST] Batch complete: {successful}/{len(urls)} successful")

    return results


# =============================================================================
# CLEAR FUNCTIONS (for testing)
# =============================================================================

def clear_vector_store() -> None:
    """Clear the vector store singleton.

    Useful for testing to reset state.

    Example:
        >>> clear_vector_store()
        >>> store = get_vector_store()  # Fresh instance
    """
    global _vector_store
    _vector_store = None
    logger.info("[INGEST] Vector store cleared")


def clear_embeddings_cache() -> None:
    """Clear the embeddings factory cache.

    This clears the embedding model cache in the embeddings factory,
    not the vector store.

    Example:
        >>> from src.shared.embeddings.factory import clear_embeddings_cache
        >>> clear_embeddings_cache()
        >>> embeddings = get_embeddings()  # Fresh instance
    """
    from src.shared.embeddings.factory import clear_embeddings_cache as _clear
    _clear()
    logger.info("[INGEST] Embeddings factory cache cleared")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "get_vector_store",
    "ingest_url",
    "ingest_urls",
    "clear_vector_store",
    "clear_embeddings_cache",
]
