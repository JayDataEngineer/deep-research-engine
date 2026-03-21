"""Shared memory storage utilities for deep research.

DEPRECATED: Use auto_vectorized_store instead.

This module is kept for backwards compatibility. New code should use:
    from .memory.auto_vectorized_store import (
        save_context_dump,
        save_webpage,
        save_insight,
    )

The auto_vectorized_store module ensures ALL content goes through
LangGraph Store for automatic vectorization.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .logging_config import get_logger

logger = get_logger(__name__)

# Re-export from auto_vectorized_store for convenience
from .auto_vectorized_store import (
    save_context_dump,
    save_webpage,
    save_webpages_batch,
    save_insight,
    get_all_memories,
    search_memories_semantic,
    mark_memories_processed,
    NAMESPACE_DUMPS,
    NAMESPACE_WEB,
    NAMESPACE_INSIGHTS,
)


async def persist_research_artifact(
    store: Any,
    session_id: str,
    content: str,
    artifact_type: str,
    metadata: Dict[str, Any],
    producer_id: str,
) -> str:
    """The Single Source of Truth for saving research artifacts.

    Handles:
    - Namespace formatting (hierarchical for easy querying)
    - Timestamping
    - Schema consistency
    - Deduplication (via source_id)

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        content: The text content to store
        artifact_type: Type of artifact (affects how GRAPH_BUILDER processes it)
            - "web_page": Scraped web content
            - "agent_insight": Agent-generated insights
            - "pdf_chunk": PDF document chunks
            - "community_post": Community sentiment data
        metadata: Additional data (url, title, confidence, etc.)
        producer_id: Which agent produced this (e.g., "web_agent", "flash_explorer")

    Returns:
        The key used to store the artifact (empty string if save failed)
    """
    if not store or not session_id:
        logger.warning(f"[PERSISTENCE] Missing store or session_id, skipping save")
        return ""

    if not content or len(content) < 10:
        logger.debug(f"[PERSISTENCE] Content too short or empty, skipping save")
        return ""

    try:
        # 1. Namespace: hierarchical for prefix queries
        # Query all: store.asearch((session_id, "memories"))
        # Query specific: store.asearch((session_id, "memories", "web_agent"))
        namespace = (session_id, "memories", producer_id)

        # 2. Key: deterministic if source_id exists, otherwise hash content
        source_id = metadata.get("source_id")
        if source_id:
            key = f"source_{source_id[:8]}"
        else:
            # Create stable hash from content
            import hashlib
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
            key = f"art_{content_hash}"

        # 3. The Schema (Single Source of Truth)
        value = {
            "content": content[:10000],  # Truncate for storage
            "type": artifact_type,
            "created_at": datetime.utcnow().isoformat(),
            "processed_by_graph_builder": False,  # For ETL pipeline
            "producer_id": producer_id,
            **metadata  # Merge in url, title, confidence, etc.
        }

        # 4. The Write
        await store.aput(namespace, key, value)

        logger.debug(
            f"[PERSISTENCE] Saved artifact: key={key}, type={artifact_type}, "
            f"producer={producer_id}, content_len={len(content)}"
        )

        return key

    except Exception as e:
        logger.warning(f"[PERSISTENCE] Failed to save artifact: {e}")
        return ""


async def save_findings_to_memories(
    store: Any,
    session_id: str,
    findings: Dict[str, Any],
    worker_id: str = "unknown",
) -> List[str]:
    """Save multiple findings to LangGraph Store in memories namespace.

    AUTO-VECTORIZED: All content is automatically vectorized by LangGraph Store.

    Routes to appropriate subcategory based on worker_id:
    - web_agent / web_research_agent → /memories/web/ (webpages)
    - other workers → /memories/insights/ (findings)

    Uses hierarchical namespace for targeted retrieval:
    - Save: (session_id, "memories", subcategory)
    - Query all: store.asearch((session_id, "memories"))

    Includes deduplication: skips sources already processed by graph builder.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        findings: Dict of source_id -> source_data
        worker_id: Identifier for the producer (e.g., "web_agent", "flash_explorer")

    Returns:
        List of saved memory keys
    """
    if not store or not session_id:
        logger.warning(f"[PERSISTENCE] Missing store or session_id, skipping save")
        return []

    # Determine subcategory based on worker_id
    if "web" in worker_id.lower():
        subcategory = NAMESPACE_WEB
    else:
        subcategory = NAMESPACE_INSIGHTS

    saved_keys = []

    for source_id, source_data in findings.items():
        try:
            # Extract content
            content = source_data.get("content", "")
            if not content or len(content) < 50:
                continue

            # Create namespace with subcategory for hierarchical queries
            namespace = (session_id, "memories", subcategory)

            # Create key from source_id
            key = f"source_{source_id[:8]}"

            # Deduplication: Check if already processed
            try:
                existing = await store.aget(namespace, key)
                if existing and existing.value.get("processed_by_graph_builder"):
                    logger.debug(f"[PERSISTENCE] Skipping {key} - already processed by graph builder")
                    continue
            except Exception:
                # Item doesn't exist, proceed with save
                pass

            # Build value dict - content field is AUTO-VECTORIZED
            value = {
                "content": content[:50000],  # Increased limit, auto-vectorized
                "subcategory": subcategory,
                "type": source_data.get("type", "web_research" if subcategory == NAMESPACE_WEB else "insight"),
                "url": source_data.get("url", ""),
                "title": source_data.get("title", ""),
                "source_id": source_id,
                "worker_id": worker_id,
                "processed_by_graph_builder": False,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add any additional metadata
            if "relevance_score" in source_data:
                value["relevance_score"] = source_data["relevance_score"]
            if "snippet" in source_data:
                value["snippet"] = source_data["snippet"][:500]

            # The Write - AUTO-VECTORIZATION HAPPENS HERE
            # LangGraph Store embeds value["content"] automatically
            await store.aput(namespace, key, value)
            saved_keys.append(key)

        except Exception as e:
            logger.warning(f"[PERSISTENCE] Failed to save source {source_id}: {e}")

    if saved_keys:
        logger.info(f"[PERSISTENCE] Saved {len(saved_keys)} sources to /memories/{subcategory}/ for {worker_id} (auto-vectorized)")

    return saved_keys


async def mark_artifacts_processed(
    store: Any,
    session_id: str,
    keys: List[str],
    producer_id: str
) -> int:
    """Mark artifacts as processed by GRAPH_BUILDER.

    Called after successfully adding to Neo4j knowledge graph.
    This is for idempotency in the ETL pipeline.

    Args:
        store: LangGraph BaseStore instance
        session_id: Research session ID
        keys: List of artifact keys to mark
        producer_id: Producer ID used when saving

    Returns:
        Number of artifacts successfully marked
    """
    if not store or not session_id or not keys:
        return 0

    namespace = (session_id, "memories", producer_id)
    count = 0

    for key in keys:
        try:
            existing = await store.aget(namespace, key)
            if existing:
                existing.value["processed_by_graph_builder"] = True
                await store.aput(namespace, key, existing.value)
                count += 1
        except Exception as e:
            logger.debug(f"[PERSISTENCE] Failed to mark {key} as processed: {e}")

    if count > 0:
        logger.info(f"[PERSISTENCE] Marked {count} artifacts as processed for {producer_id}")

    return count
