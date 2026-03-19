"""Deduplication - Content hashing for duplicate detection.

This module provides a single source of truth for content hashing
and duplicate detection across all Context Engine storage operations.

Usage:
    from deep_research.storage.deduplication import (
        compute_content_hash,
        is_duplicate,
        check_duplicates,
    )
    
    # Compute hash of content
    hash = compute_content_hash("Important research finding...")
    
    # Check if content already exists
    is_dup = await is_duplicate(store, namespace, content)
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from langgraph.store.base import BaseStore

from .namespace_factory import namespace_to_str

logger = logging.getLogger(__name__)


# =============================================================================
# CONTENT HASHING
# =============================================================================

def compute_content_hash(
    content: str,
    normalize: bool = True,
    algorithm: str = "md5",
) -> str:
    """Compute hash of content for deduplication.
    
    Normalizes text before hashing to detect semantically identical content
    even with formatting differences.
    
    Args:
        content: Content string to hash
        normalize: Whether to normalize text (lowercase, collapse whitespace)
        algorithm: Hash algorithm to use ("md5", "sha256", etc.)
    
    Returns:
        Hexadecimal hash string
        
    Examples:
        >>> compute_content_hash("Hello World")
        'b10a8db164e0754105b7a99be72e3fe5'
        
        >>> compute_content_hash("  Hello   World  ", normalize=True)
        'b10a8db164e0754105b7a99be72e3fe5'  # Same as above
    """
    if normalize:
        # Normalize: lowercase, collapse whitespace, strip
        content = " ".join(content.split()).lower().strip()
    
    if algorithm == "md5":
        return hashlib.md5(content.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(content.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def compute_dict_hash(
    data: Dict[str, Any],
    keys_to_hash: Optional[List[str]] = None,
    normalize: bool = True,
) -> str:
    """Compute hash of dictionary content.
    
    Useful for hashing structured data like findings, citations, etc.
    
    Args:
        data: Dictionary to hash
        keys_to_hash: Specific keys to include in hash (None = all string values)
        normalize: Whether to normalize text values
    
    Returns:
        Hexadecimal hash string
        
    Examples:
        >>> data = {"content": "Research finding", "source": "example.com"}
        >>> compute_dict_hash(data, keys_to_hash=["content"])
        'abc123...'
    """
    if keys_to_hash:
        # Hash only specified keys
        values_to_hash = []
        for key in keys_to_hash:
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    values_to_hash.append(value)
                elif isinstance(value, list):
                    values_to_hash.append(" ".join(str(v) for v in value))
                else:
                    values_to_hash.append(str(value))
        combined = " ".join(values_to_hash)
    else:
        # Hash all string values
        values_to_hash = []
        for value in data.values():
            if isinstance(value, str):
                values_to_hash.append(value)
            elif isinstance(value, list):
                values_to_hash.append(" ".join(str(v) for v in value))
            elif isinstance(value, dict):
                # Recursively hash nested dicts
                values_to_hash.append(compute_dict_hash(value, normalize=normalize))
            else:
                values_to_hash.append(str(value))
        combined = " ".join(values_to_hash)
    
    return compute_content_hash(combined, normalize=normalize)


# =============================================================================
# DUPLICATE DETECTION
# =============================================================================

async def is_duplicate(
    store: BaseStore,
    namespace: Tuple[str, ...],
    content: str,
    content_field: str = "content",
) -> bool:
    """Check if content already exists in namespace.
    
    Uses exact hash matching on normalized content for deduplication.
    
    Args:
        store: LangGraph Store instance
        namespace: Namespace tuple to search
        content: Content string to check for duplicates
        content_field: Field name containing content in stored values
    
    Returns:
        True if duplicate found, False otherwise
        
    Raises:
        StorageError: If store operation fails
        
    Examples:
        >>> from langgraph.store.postgres import PostgresStore
        >>> store = PostgresStore(...)
        >>> ns = ("session-123", "memories", "findings")
        >>> await is_duplicate(store, ns, "Research finding...")
        False
    """
    from ..exceptions import StorageError
    
    content_hash = compute_content_hash(content)
    namespace_str = namespace_to_str(namespace)
    
    try:
        # Search within namespace
        items = await store.asearch(namespace)
        
        for item in items:
            if isinstance(item.value, dict):
                existing_content = item.value.get(content_field, "")
                if isinstance(existing_content, str):
                    existing_hash = compute_content_hash(existing_content)
                    if existing_hash == content_hash:
                        logger.debug(
                            f"[DEDUP] Duplicate content detected in {namespace_str}"
                        )
                        return True
        
        return False
        
    except Exception as e:
        raise StorageError(
            f"Failed to check for duplicates in {namespace_str}: {e}"
        ) from e


async def find_similar_content(
    store: BaseStore,
    namespace: Tuple[str, ...],
    content: str,
    threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """Find similar content in namespace using semantic search.
    
    Uses store's built-in semantic search to find content with similar meaning
    (not just exact hash matches).
    
    Args:
        store: LangGraph Store instance with semantic search
        namespace: Namespace tuple to search
        content: Query content
        threshold: Similarity threshold (0-1)
    
    Returns:
        List of similar items with scores
        
    Examples:
        >>> similar = await find_similar_content(
        ...     store,
        ...     ("session-123", "memories", "findings"),
        ...     "AI research findings",
        ...     threshold=0.7
        ... )
    """
    from ..exceptions import StorageError
    
    namespace_str = namespace_to_str(namespace)
    
    try:
        # Use semantic search
        items = await store.asearch(namespace, query=content, limit=10)
        
        # Filter by similarity threshold
        similar_items = []
        for item in items:
            if hasattr(item, 'score') and item.score >= threshold:
                similar_items.append({
                    "key": item.key,
                    "value": item.value,
                    "score": item.score,
                })
        
        return similar_items
        
    except Exception as e:
        raise StorageError(
            f"Failed to find similar content in {namespace_str}: {e}"
        ) from e


async def check_duplicates(
    store: BaseStore,
    namespace: Tuple[str, ...],
    items: List[Dict[str, Any]],
    content_field: str = "content",
) -> List[Dict[str, Any]]:
    """Check multiple items for duplicates and return only unique ones.
    
    Args:
        store: LangGraph Store instance
        namespace: Namespace tuple to search
        items: List of items to check (each should have content_field)
        content_field: Field name containing content
    
    Returns:
        List of unique items (duplicates removed)
        
    Examples:
        >>> items = [
        ...     {"content": "Finding 1", "source": "example.com"},
        ...     {"content": "Finding 2", "source": "test.com"},
        ... ]
        >>> unique = await check_duplicates(store, namespace, items)
    """
    unique_items = []
    
    for item in items:
        content = item.get(content_field, "")
        if not content:
            continue
            
        if not await is_duplicate(store, namespace, content, content_field):
            unique_items.append(item)
        else:
            logger.debug(f"[DEDUP] Skipping duplicate: {content[:50]}...")
    
    return unique_items


# =============================================================================
# BATCH DEDUPLICATION
# =============================================================================

async def deduplicate_batch(
    store: BaseStore,
    namespace: Tuple[str, ...],
    items: List[Dict[str, Any]],
    content_field: str = "content",
    check_similarity: bool = False,
    similarity_threshold: float = 0.9,
) -> Dict[str, Any]:
    """Deduplicate a batch of items against existing store data.
    
    Args:
        store: LangGraph Store instance
        namespace: Namespace tuple
        items: Items to deduplicate
        content_field: Field containing content
        check_similarity: Whether to check for similar (not just duplicate) content
        similarity_threshold: Threshold for similarity detection
    
    Returns:
        Dictionary with:
            - unique_items: List of unique items to store
            - duplicates_skipped: Number of duplicates skipped
            - similar_found: Number of similar items found (if check_similarity)
            
    Examples:
        >>> result = await deduplicate_batch(
        ...     store,
        ...     ("session-123", "memories", "findings"),
        ...     items,
        ... )
        >>> unique_items = result["unique_items"]
    """
    from ..exceptions import StorageError
    
    namespace_str = namespace_to_str(namespace)
    
    try:
        unique_items = []
        duplicates_skipped = 0
        similar_count = 0
        
        for item in items:
            content = item.get(content_field, "")
            if not content:
                continue
            
            # Check for exact duplicate
            if await is_duplicate(store, namespace, content, content_field):
                duplicates_skipped += 1
                continue
            
            # Optionally check for similar content
            if check_similarity:
                similar = await find_similar_content(
                    store, namespace, content, similarity_threshold
                )
                if similar:
                    similar_count += 1
                    logger.debug(
                        f"[DEDUP] Found {len(similar)} similar items for: {content[:50]}..."
                    )
            
            unique_items.append(item)
        
        logger.info(
            f"[DEDUP] Batch deduplication in {namespace_str}: "
            f"{len(items)} items → {len(unique_items)} unique "
            f"({duplicates_skipped} duplicates, {similar_count} similar)"
        )
        
        return {
            "unique_items": unique_items,
            "duplicates_skipped": duplicates_skipped,
            "similar_found": similar_count,
        }
        
    except Exception as e:
        raise StorageError(
            f"Failed to deduplicate batch in {namespace_str}: {e}"
        ) from e


# =============================================================================
# CONTENT NORMALIZATION
# =============================================================================

def normalize_content(content: str) -> str:
    """Normalize content for consistent comparison.
    
    Applied before hashing and duplicate detection.
    
    Args:
        content: Raw content string
    
    Returns:
        Normalized content string
        
    Examples:
        >>> normalize_content("  Hello   World  ")
        'hello world'
    """
    # Lowercase
    content = content.lower()
    # Collapse whitespace
    content = " ".join(content.split())
    # Strip leading/trailing whitespace
    content = content.strip()
    return content


def normalize_for_comparison(content: str, ignore_urls: bool = True) -> str:
    """Normalize content for comparison (more aggressive than hashing).
    
    Removes elements that shouldn't affect duplicate detection:
    - URLs (if ignore_urls=True)
    - Extra whitespace
    - Case differences
    - Punctuation
    
    Args:
        content: Raw content string
        ignore_urls: Whether to remove URLs from comparison
    
    Returns:
        Normalized content for comparison
    """
    import re
    
    # Normalize basic content
    content = normalize_content(content)
    
    # Optionally remove URLs
    if ignore_urls:
        content = re.sub(r'https?://\S+', '', content)
    
    # Remove punctuation (keep alphanumeric and spaces)
    content = re.sub(r'[^\w\s]', '', content)
    
    # Collapse whitespace again
    content = " ".join(content.split())
    
    return content


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Hashing
    "compute_content_hash",
    "compute_dict_hash",
    # Duplicate detection
    "is_duplicate",
    "find_similar_content",
    "check_duplicates",
    "deduplicate_batch",
    # Normalization
    "normalize_content",
    "normalize_for_comparison",
]
