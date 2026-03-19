"""Document Deduplication Utilities

Provides exact and semantic deduplication for documents before embedding/indexing.

Why deduplicate?
- Saves embedding costs (API calls)
- Improves retrieval quality (no duplicate results)
- Reduces storage requirements

Usage:
    from src.shared.utils.deduplication import (
        deduplicate_exact,
        deduplicate_semantic,
        deduplicate_hybrid,
    )

    # Fast exact dedup (no embeddings needed)
    result = deduplicate_exact(documents)

    # Semantic dedup (catches paraphrases)
    result = await deduplicate_semantic(documents, embeddings, threshold=0.95)

    # Best of both (exact first, then semantic)
    result = deduplicate_hybrid(documents, embeddings)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for consistent hashing.

    Strips whitespace, collapses multiple spaces, lowercases.
    """
    return " ".join(text.split()).lower().strip()


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash of normalized content.

    MD5 is fast and sufficient for deduplication (not security).
    """
    normalized = normalize_text(content)
    return hashlib.md5(normalized.encode()).hexdigest()


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""

    unique_documents: List[Any]
    duplicate_count: int = 0
    total_count: int = 0
    duplicate_hashes: Set[str] = field(default_factory=set)

    @property
    def deduplication_rate(self) -> float:
        """Percentage of documents that were duplicates."""
        if self.total_count == 0:
            return 0.0
        return self.duplicate_count / self.total_count


def deduplicate_exact(
    documents: List[Any],
    content_field: str = "page_content",
    metadata_field: str = "metadata",
) -> DeduplicationResult:
    """Deduplicate documents using exact hash matching.

    Fast, no embeddings needed. Catches exact and near-exact duplicates
    (after normalization).

    Args:
        documents: List of Document objects or dicts
        content_field: Field name for document content
        metadata_field: Field name for document metadata

    Returns:
        DeduplicationResult with unique documents and stats

    Example:
        >>> from langchain_core.documents import Document
        >>> docs = [
        ...     Document(page_content="Hello world"),
        ...     Document(page_content="  hello   world  "),  # Duplicate after normalization
        ...     Document(page_content="Different content"),
        ... ]
        >>> result = deduplicate_exact(docs)
        >>> print(f"Kept {len(result.unique_documents)} of {result.total_count}")
        Kept 2 of 3
    """
    seen_hashes: Set[str] = set()
    unique_documents: List[Any] = []
    duplicate_hashes: Set[str] = set()

    for doc in documents:
        # Extract content
        if hasattr(doc, content_field):
            content = getattr(doc, content_field)
        elif isinstance(doc, dict):
            content = doc.get(content_field, "")
        else:
            content = str(doc)

        # Compute hash
        content_hash = compute_content_hash(content)

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_documents.append(doc)
        else:
            duplicate_hashes.add(content_hash)

    duplicate_count = len(documents) - len(unique_documents)

    result = DeduplicationResult(
        unique_documents=unique_documents,
        duplicate_count=duplicate_count,
        total_count=len(documents),
        duplicate_hashes=duplicate_hashes,
    )

    if duplicate_count > 0:
        logger.info(
            f"[Deduplication] Exact: {len(unique_documents)}/{len(documents)} unique "
            f"({result.deduplication_rate:.1%} duplicates removed)"
        )

    return result


async def deduplicate_semantic(
    documents: List[Any],
    embeddings: "Embeddings",
    threshold: float = 0.95,
    content_field: str = "page_content",
    batch_size: int = 100,
) -> DeduplicationResult:
    """Deduplicate documents using semantic similarity.

    Catches near-duplicates that exact matching misses (paraphrases, etc.).
    Requires embeddings (costs API calls for cloud providers).

    Algorithm:
    1. Embed all documents
    2. For each document, check similarity to all prior unique documents
    3. Keep if max similarity < threshold

    Args:
        documents: List of Document objects or dicts
        embeddings: Embeddings instance for computing vectors
        threshold: Similarity threshold (0.95 = very strict, 0.8 = loose)
        content_field: Field name for document content
        batch_size: Batch size for embedding API calls

    Returns:
        DeduplicationResult with unique documents and stats

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> docs = [
        ...     Document(page_content="The cat sat on the mat"),
        ...     Document(page_content="A cat was sitting on a mat"),  # Near-duplicate
        ...     Document(page_content="Dogs are great pets"),
        ... ]
        >>> result = await deduplicate_semantic(docs, OpenAIEmbeddings(), threshold=0.9)
        >>> print(f"Kept {len(result.unique_documents)} of {result.total_count}")
        Kept 2 of 3
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if not documents:
        return DeduplicationResult(
            unique_documents=[],
            duplicate_count=0,
            total_count=0,
        )

    # Extract contents
    contents = []
    for doc in documents:
        if hasattr(doc, content_field):
            contents.append(getattr(doc, content_field))
        elif isinstance(doc, dict):
            contents.append(doc.get(content_field, ""))
        else:
            contents.append(str(doc))

    # Embed all documents in batches
    all_embeddings_list = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        batch_embeddings = await embeddings.aembed_documents(batch)
        all_embeddings_list.extend(batch_embeddings)

    all_embeddings = np.array(all_embeddings_list)

    # Find unique documents using similarity threshold
    unique_indices = [0]  # First document is always unique

    for i in range(1, len(documents)):
        # Compare to all unique documents so far
        unique_embeddings = all_embeddings[unique_indices]
        current_embedding = all_embeddings[i:i+1]

        # Compute cosine similarities
        similarities = cosine_similarity(current_embedding, unique_embeddings)[0]
        max_similarity = np.max(similarities)

        if max_similarity < threshold:
            unique_indices.append(i)

    unique_documents = [documents[i] for i in unique_indices]
    duplicate_count = len(documents) - len(unique_documents)

    result = DeduplicationResult(
        unique_documents=unique_documents,
        duplicate_count=duplicate_count,
        total_count=len(documents),
    )

    if duplicate_count > 0:
        logger.info(
            f"[Deduplication] Semantic (threshold={threshold}): "
            f"{len(unique_documents)}/{len(documents)} unique "
            f"({result.deduplication_rate:.1%} duplicates removed)"
        )

    return result


def deduplicate_hybrid(
    documents: List[Any],
    embeddings: Optional["Embeddings"] = None,
    semantic_threshold: float = 0.95,
    content_field: str = "page_content",
) -> DeduplicationResult:
    """Two-stage deduplication: exact first, then semantic.

    Most efficient approach:
    1. Remove exact duplicates (fast, free)
    2. Remove semantic duplicates (slower, costs API calls)

    Args:
        documents: List of documents to deduplicate
        embeddings: Optional embeddings for semantic deduplication
        semantic_threshold: Similarity threshold for semantic dedup
        content_field: Field name for document content

    Returns:
        DeduplicationResult with final unique documents

    Example:
        >>> result = deduplicate_hybrid(docs, embeddings)
    """
    import asyncio

    # Stage 1: Exact deduplication
    exact_result = deduplicate_exact(documents, content_field)

    if embeddings is None:
        return exact_result

    # Stage 2: Semantic deduplication
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - need to run in executor or use create_task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                deduplicate_semantic(
                    exact_result.unique_documents,
                    embeddings,
                    threshold=semantic_threshold,
                    content_field=content_field,
                )
            )
            semantic_result = future.result()
    except RuntimeError:
        # No running loop, can use asyncio.run directly
        semantic_result = asyncio.run(
            deduplicate_semantic(
                exact_result.unique_documents,
                embeddings,
                threshold=semantic_threshold,
                content_field=content_field,
            )
        )

    # Combine stats
    return DeduplicationResult(
        unique_documents=semantic_result.unique_documents,
        duplicate_count=exact_result.duplicate_count + semantic_result.duplicate_count,
        total_count=len(documents),
    )


# =============================================================================
# Store Integration Helpers
# =============================================================================

class DeduplicatingStore:
    """Mixin for stores that want automatic deduplication.

    Usage:
        class MyStore(DeduplicatingStore, BaseStore):
            def put(self, namespace, key, value):
                # Check for duplicates before storing
                if self._is_duplicate(value):
                    return  # Skip duplicate
                # ... store logic
    """

    def __init__(self, dedup_enabled: bool = True, **kwargs):
        self._dedup_enabled = dedup_enabled
        self._content_hashes: Dict[str, str] = {}  # hash -> key mapping
        super().__init__(**kwargs)

    def _compute_hash(self, value: Dict[str, Any], content_field: str = "content") -> str:
        """Compute hash for a value dict."""
        content = value.get(content_field, "")
        if not content:
            content = value.get("page_content", "")
        return compute_content_hash(str(content))

    def _is_duplicate(self, value: Dict[str, Any], namespace: str = None) -> bool:
        """Check if value is a duplicate based on content hash."""
        if not self._dedup_enabled:
            return False

        content_hash = self._compute_hash(value)
        return content_hash in self._content_hashes

    def _record_hash(self, key: str, value: Dict[str, Any]) -> None:
        """Record hash after storing."""
        content_hash = self._compute_hash(value)
        self._content_hashes[content_hash] = key
