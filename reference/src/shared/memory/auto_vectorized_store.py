"""Auto-Vectorized Memory Store - Single source of truth for research content.

This module ensures ALL research CONTENT goes through the LangGraph Store
for automatic vectorization. The Graph Builder reads from this store to
extract entities for Neo4j.

Entry Points for AUTO-VECTORIZATION:
    A) Context dumps (context overflow) → /memories/dumps/
    B) Web agent (webpage content) → /memories/web/
    C) Worker insights (findings) → /memories/insights/

Entry Point D (Graph entities) → Goes DIRECTLY to Neo4j, NOT here.
    Neo4j stores entity relationships, not semantic vectors.

ALL content paths route through LangGraph Store for auto-vectorization.
Graph Builder queries: store.asearch((session_id, "memories"))
Then extracts entities → writes to Neo4j separately.

Usage:
    from .memory.auto_vectorized_store import (
        save_context_dump,
        save_webpage,
        save_insight,
    )

    # Entry Point A: Context overflow
    await save_context_dump(store, session_id, "state_step_5", state_json)

    # Entry Point B: Web agent
    await save_webpage(store, session_id, url, content, metadata)

    # Entry Point C: Worker insight
    await save_insight(store, session_id, "worker_0", findings, confidence)

    # Entry Point D: Graph entities → Use Neo4j directly, NOT this module
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import hashlib
import json

from .logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# SCHEMA CONSTANTS
# =============================================================================

# Namespace structure: (session_id, "memories", subcategory)
# Graph Builder queries: (session_id, "memories") to get ALL subcategories

NAMESPACE_DUMPS = "dumps"      # Entry Point A: Context overflow
NAMESPACE_WEB = "web"          # Entry Point B: Webpage content
NAMESPACE_INSIGHTS = "insights"  # Entry Point C: Worker findings
# NOTE: Entry Point D (entities) goes to Neo4j, NOT to LangGraph Store


# =============================================================================
# CORE SAVE FUNCTION
# =============================================================================

async def _save_to_store(
    store: Any,
    session_id: str,
    subcategory: str,
    key: str,
    content: str,
    metadata: Dict[str, Any],
) -> str:
    """Core function to save content to LangGraph Store with auto-vectorization.

    The LangGraph Store automatically vectorizes the "content" field
    when index_fields=["content"] is configured.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        subcategory: Subcategory (dumps, web, insights, entities)
        key: Unique key for this item
        content: Text content to store and vectorize
        metadata: Additional metadata

    Returns:
        The key used to store the item
    """
    if not store:
        logger.warning(f"[AUTO_VECTORIZE] No store available, skipping save")
        return ""

    if not session_id:
        logger.warning(f"[AUTO_VECTORIZE] No session_id, skipping save")
        return ""

    if not content or len(content) < 10:
        logger.debug(f"[AUTO_VECTORIZE] Content too short, skipping save")
        return ""

    try:
        # Namespace: (session_id, "memories", subcategory)
        # Graph Builder queries (session_id, "memories") to get ALL
        namespace = (session_id, "memories", subcategory)

        # Build value with content field (auto-vectorized by LangGraph Store)
        value = {
            "content": content[:50000],  # Truncate for storage (increased from 10k)
            "subcategory": subcategory,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "processed_by_graph_builder": False,  # For ETL tracking
            **metadata
        }

        # Write to store - AUTO-VECTORIZATION HAPPENS HERE
        await store.aput(namespace, key, value)

        logger.debug(
            f"[AUTO_VECTORIZE] Saved to ({session_id}, memories, {subcategory})/{key} "
            f"- content_len={len(content)}, auto-vectorized"
        )

        return key

    except Exception as e:
        logger.error(f"[AUTO_VECTORIZE] Failed to save: {e}")
        raise RuntimeError(f"Auto-vectorized save failed: {e}") from e


# =============================================================================
# ENTRY POINT A: Context Dumps
# =============================================================================

async def save_context_dump(
    store: Any,
    session_id: str,
    dump_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save context overflow dump with auto-vectorization.

    Entry Point A: When deep agents run out of context, they dump state here.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        dump_type: Type of dump (state, artifact, checkpoint, summary)
        content: Dump content (JSON or text)
        metadata: Additional metadata

    Returns:
        Storage key
    """
    # Generate key from dump type and timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    key = f"dump_{dump_type}_{timestamp}"

    full_metadata = {
        "dump_type": dump_type,
        "source": "context_overflow",
        **(metadata or {})
    }

    return await _save_to_store(
        store=store,
        session_id=session_id,
        subcategory=NAMESPACE_DUMPS,
        key=key,
        content=content,
        metadata=full_metadata,
    )


# =============================================================================
# ENTRY POINT B: Webpage Content
# =============================================================================

async def save_webpage(
    store: Any,
    session_id: str,
    url: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save webpage content with auto-vectorization.

    Entry Point B: Web agent scrapes pages, saves after critic approval.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        url: Source URL
        content: Page content (markdown or text)
        metadata: Additional metadata (title, snippet, etc.)

    Returns:
        Storage key
    """
    # Generate key from URL hash
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    key = f"web_{url_hash}"

    full_metadata = {
        "url": url,
        "source": "web_agent",
        "type": "webpage",
        **(metadata or {})
    }

    return await _save_to_store(
        store=store,
        session_id=session_id,
        subcategory=NAMESPACE_WEB,
        key=key,
        content=content,
        metadata=full_metadata,
    )


async def save_webpages_batch(
    store: Any,
    session_id: str,
    webpages: Dict[str, Dict[str, Any]],
    worker_id: str = "web_agent",
) -> List[str]:
    """Save multiple webpages at once.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        webpages: Dict of url -> {content, title, snippet, ...}
        worker_id: Worker ID for tracking

    Returns:
        List of saved keys
    """
    saved_keys = []

    for url, data in webpages.items():
        content = data.get("content", "")
        if not content or len(content) < 50:
            continue

        key = await save_webpage(
            store=store,
            session_id=session_id,
            url=url,
            content=content,
            metadata={
                "title": data.get("title", ""),
                "snippet": data.get("snippet", "")[:500],
                "relevance_score": data.get("relevance_score"),
                "worker_id": worker_id,
            }
        )
        if key:
            saved_keys.append(key)

    logger.info(f"[AUTO_VECTORIZE] Saved {len(saved_keys)} webpages for {worker_id}")
    return saved_keys


# =============================================================================
# ENTRY POINT C: Worker Insights
# =============================================================================

async def save_insight(
    store: Any,
    session_id: str,
    worker_id: str,
    content: str,
    artifact_type: str = "finding",
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save worker insight/finding with auto-vectorization.

    Entry Point C: Flash explorers and workers save their findings.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        worker_id: ID of the worker that produced this insight
        content: The insight/finding content
        artifact_type: Type (finding, claim, citation, decision)
        confidence: Confidence score (0-1)
        metadata: Additional metadata

    Returns:
        Storage key
    """
    # Generate key from worker ID and content hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    key = f"insight_{worker_id}_{content_hash}"

    full_metadata = {
        "worker_id": worker_id,
        "artifact_type": artifact_type,
        "confidence": confidence,
        "source": "worker_insight",
        **(metadata or {})
    }

    return await _save_to_store(
        store=store,
        session_id=session_id,
        subcategory=NAMESPACE_INSIGHTS,
        key=key,
        content=content,
        metadata=full_metadata,
    )


# NOTE: Entry Point D (Graph Entities) goes DIRECTLY to Neo4j
# Neo4j stores entity relationships, not semantic vectors
# See: graph_builder/node.py for Neo4j entity creation


# =============================================================================
# QUERY FUNCTIONS (for Graph Builder)
# =============================================================================

async def get_all_memories(
    store: Any,
    session_id: str,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Get ALL memories for a session (used by Graph Builder).

    Queries the (session_id, "memories") namespace prefix to get
    all subcategories: dumps, web, insights.

    Graph Builder uses this to read ALL content, extract entities,
    and write them to Neo4j (Entry Point D).

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        limit: Maximum items to return

    Returns:
        List of memory items with content and metadata
    """
    if not store:
        raise RuntimeError("Store not available")

    namespace_prefix = (session_id, "memories")

    # Query without filter to get ALL items
    results = await store.asearch(
        namespace_prefix,
        query=None,  # No semantic filter
        limit=limit,
    )

    memories = []
    for item in results:
        memories.append({
            "key": item.key,
            "namespace": item.namespace,
            "content": item.value.get("content", ""),
            "subcategory": item.value.get("subcategory", "unknown"),
            "metadata": item.value,
            "created_at": str(item.created_at) if hasattr(item, 'created_at') else "",
        })

    logger.info(
        f"[AUTO_VECTORIZE] Retrieved {len(memories)} memories for session {session_id}"
    )
    return memories


async def search_memories_semantic(
    store: Any,
    session_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Semantic search across ALL memories for a session.

    Uses the auto-vectorized embeddings for similarity search.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        query: Search query
        limit: Maximum results

    Returns:
        List of matching memories with scores
    """
    if not store:
        raise RuntimeError("Store not available")

    namespace_prefix = (session_id, "memories")

    # Semantic search using auto-vectorized embeddings
    results = await store.asearch(
        namespace_prefix,
        query=query,
        limit=limit,
    )

    memories = []
    for item in results:
        memories.append({
            "key": item.key,
            "namespace": item.namespace,
            "content": item.value.get("content", ""),
            "score": getattr(item, 'score', 0.0),
            "metadata": item.value,
        })

    return memories


# =============================================================================
# MARK AS PROCESSED (for ETL pipeline)
# =============================================================================

async def mark_memories_processed(
    store: Any,
    session_id: str,
    keys: List[str],
    subcategory: str,
) -> int:
    """Mark memories as processed by Graph Builder.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        keys: List of keys to mark
        subcategory: Subcategory namespace

    Returns:
        Number of items marked
    """
    if not store or not keys:
        return 0

    namespace = (session_id, "memories", subcategory)
    count = 0

    for key in keys:
        try:
            existing = await store.aget(namespace, key)
            if existing:
                existing.value["processed_by_graph_builder"] = True
                await store.aput(namespace, key, existing.value)
                count += 1
        except Exception as e:
            logger.debug(f"[AUTO_VECTORIZE] Failed to mark {key}: {e}")

    return count


__all__ = [
    # Save functions (Entry Points A, B, C - auto-vectorized)
    "save_context_dump",    # Entry Point A: Context overflow
    "save_webpage",         # Entry Point B: Web agent content
    "save_webpages_batch",  # Entry Point B: Batch webpages
    "save_insight",         # Entry Point C: Worker findings
    # NOTE: Entry Point D (entities) goes to Neo4j, not here
    # Query functions (for Graph Builder)
    "get_all_memories",
    "search_memories_semantic",
    "mark_memories_processed",
    # Namespace constants
    "NAMESPACE_DUMPS",
    "NAMESPACE_WEB",
    "NAMESPACE_INSIGHTS",
]
