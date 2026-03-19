"""Storage Operations - Unified storage operations for Context Engine.

This module provides a SINGLE SOURCE OF TRUTH for all storage operations
in the deep research system. All save/load operations MUST go through
these functions to ensure:

1. Consistent namespace construction (via namespace_factory)
2. Automatic deduplication (via deduplication)
3. Proper error handling (via exceptions)
4. Observability logging

Usage:
    from deep_research.storage.operations import (
        save_worker_finding,
        save_citation,
        save_artifact,
        load_findings,
    )
    
    # Save a worker finding
    await save_worker_finding(
        store=store,
        session_id="uuid-123",
        worker_id="philosopher-1",
        content="Research finding...",
        metadata={"source": "example.com"}
    )
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from langgraph.store.base import BaseStore

from .namespace_factory import (
    findings_namespace,
    citations_namespace,
    artifacts_namespace,
    research_brief_namespace,
    synthesis_namespace,
    critique_namespace,
    boss_decision_namespace,
    architect_findings_namespace,
    session_metadata_namespace,
    namespace_to_str,
)
from .deduplication import (
    compute_content_hash,
    is_duplicate,
    deduplicate_batch,
)

logger = logging.getLogger(__name__)


# =============================================================================
# WORKER FINDINGS
# =============================================================================

async def save_worker_finding(
    store: BaseStore,
    session_id: str,
    worker_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    check_duplicate: bool = True,
) -> bool:
    """Save a worker finding to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID (single source of truth)
        worker_id: Worker identifier (e.g., "philosopher-1")
        content: Finding content
        metadata: Optional metadata (source, confidence, etc.)
        check_duplicate: Whether to check for duplicates before saving
    
    Returns:
        True if saved, False if skipped (duplicate)
        
    Raises:
        StorageError: If save operation fails
        
    Examples:
        >>> await save_worker_finding(
        ...     store,
        ...     session_id="uuid-123",
        ...     worker_id="philosopher-1",
        ...     content="Key research finding...",
        ...     metadata={"source": "example.com", "confidence": 0.9}
        ... )
        True
    """
    from ..exceptions import StorageError, DuplicateContentError
    
    namespace = findings_namespace(session_id, worker_id)
    namespace_str = namespace_to_str(namespace)
    
    try:
        # Build value object
        value = {
            "content": content,
            "worker_id": worker_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": compute_content_hash(content),
            **(metadata or {}),
        }
        
        # Generate key from content hash (for deduplication)
        key = f"finding_{value['content_hash']}"
        
        # Check for duplicate
        if check_duplicate:
            if await is_duplicate(store, namespace, content, "content"):
                logger.debug(
                    f"[STORAGE] Skipping duplicate finding in {namespace_str}"
                )
                return False
        
        # Save to store
        await store.aput(namespace, key, value)
        
        logger.info(
            f"[STORAGE] Saved worker finding: {worker_id} in {namespace_str}"
        )
        return True
        
    except DuplicateContentError:
        raise
    except Exception as e:
        raise StorageError(
            f"Failed to save worker finding ({worker_id}) in {namespace_str}: {e}"
        ) from e


async def save_worker_findings_batch(
    store: BaseStore,
    session_id: str,
    worker_id: str,
    findings: List[Dict[str, Any]],
    content_field: str = "content",
) -> Dict[str, Any]:
    """Save multiple worker findings in a batch.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        worker_id: Worker identifier
        findings: List of finding dicts (each with content_field)
        content_field: Field name containing content
    
    Returns:
        Dictionary with save statistics
        
    Examples:
        >>> findings = [
        ...     {"content": "Finding 1", "source": "example.com"},
        ...     {"content": "Finding 2", "source": "test.com"},
        ... ]
        >>> result = await save_worker_findings_batch(
        ...     store, "uuid-123", "philosopher-1", findings
        ... )
    """
    from ..exceptions import StorageError
    
    namespace = findings_namespace(session_id, worker_id)
    namespace_str = namespace_to_str(namespace)
    
    try:
        # Deduplicate batch
        dedup_result = await deduplicate_batch(
            store, namespace, findings, content_field
        )
        
        unique_items = dedup_result["unique_items"]
        
        # Save each unique finding
        saved_count = 0
        for item in unique_items:
            content = item.get(content_field, "")
            metadata = {k: v for k, v in item.items() if k != content_field}
            
            if await save_worker_finding(
                store, session_id, worker_id, content, metadata, check_duplicate=False
            ):
                saved_count += 1
        
        logger.info(
            f"[STORAGE] Batch save in {namespace_str}: "
            f"{len(findings)} items → {saved_count} saved "
            f"({dedup_result['duplicates_skipped']} duplicates)"
        )
        
        return {
            "total_input": len(findings),
            "saved": saved_count,
            "duplicates_skipped": dedup_result["duplicates_skipped"],
            "similar_found": dedup_result["similar_found"],
        }
        
    except Exception as e:
        raise StorageError(
            f"Failed to batch save findings in {namespace_str}: {e}"
        ) from e


async def load_worker_findings(
    store: BaseStore,
    session_id: str,
    worker_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Load worker findings from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        worker_id: Optional worker ID filter (None = all workers)
        limit: Maximum results to return
    
    Returns:
        List of findings with metadata
        
    Examples:
        >>> findings = await load_worker_findings(
        ...     store, "uuid-123", worker_id="philosopher-1"
        ... )
    """
    from ..exceptions import StorageError
    
    namespace = findings_namespace(session_id, worker_id)
    namespace_str = namespace_to_str(namespace)
    
    try:
        items = await store.asearch(namespace, limit=limit)
        
        findings = []
        for item in items:
            if isinstance(item.value, dict):
                findings.append({
                    "key": item.key,
                    **item.value,
                })
        
        logger.debug(
            f"[STORAGE] Loaded {len(findings)} findings from {namespace_str}"
        )
        return findings
        
    except Exception as e:
        raise StorageError(
            f"Failed to load findings from {namespace_str}: {e}"
        ) from e


# =============================================================================
# CITATIONS
# =============================================================================

async def save_citation(
    store: BaseStore,
    session_id: str,
    citation: Dict[str, Any],
    check_duplicate: bool = True,
) -> bool:
    """Save a citation to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        citation: Citation dict with url, title, content, etc.
        check_duplicate: Whether to check for duplicates
    
    Returns:
        True if saved, False if skipped (duplicate)
        
    Raises:
        StorageError: If save operation fails
    """
    from ..exceptions import StorageError
    
    namespace = citations_namespace(session_id)
    namespace_str = namespace_to_str(namespace)
    
    try:
        # Extract content for deduplication
        content = citation.get("content", citation.get("excerpt", ""))
        
        # Check for duplicate
        if check_duplicate and content:
            if await is_duplicate(store, namespace, content, "content"):
                logger.debug(
                    f"[STORAGE] Skipping duplicate citation in {namespace_str}"
                )
                return False
        
        # Build value with metadata
        value = {
            **citation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Generate key from URL or content hash
        if "url" in citation:
            key = f"citation_{compute_content_hash(citation['url'])}"
        elif content:
            key = f"citation_{compute_content_hash(content)}"
        else:
            key = f"citation_{datetime.now(timezone.utc).timestamp()}"
        
        # Save to store
        await store.aput(namespace, key, value)
        
        logger.info(
            f"[STORAGE] Saved citation in {namespace_str}: {citation.get('title', 'Untitled')}"
        )
        return True
        
    except Exception as e:
        raise StorageError(
            f"Failed to save citation in {namespace_str}: {e}"
        ) from e


async def load_citations(
    store: BaseStore,
    session_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Load citations from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        limit: Maximum results to return
    
    Returns:
        List of citations with metadata
    """
    from ..exceptions import StorageError
    
    namespace = citations_namespace(session_id)
    namespace_str = namespace_to_str(namespace)
    
    try:
        items = await store.asearch(namespace, limit=limit)
        
        citations = []
        for item in items:
            if isinstance(item.value, dict):
                citations.append({
                    "key": item.key,
                    **item.value,
                })
        
        logger.debug(
            f"[STORAGE] Loaded {len(citations)} citations from {namespace_str}"
        )
        return citations
        
    except Exception as e:
        raise StorageError(
            f"Failed to load citations from {namespace_str}: {e}"
        ) from e


# =============================================================================
# ARTIFACTS (Generic)
# =============================================================================

async def save_artifact(
    store: BaseStore,
    session_id: str,
    artifact_type: str,
    content: Dict[str, Any],
    key: Optional[str] = None,
    check_duplicate: bool = False,
) -> bool:
    """Save a generic artifact to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        artifact_type: Type of artifact (e.g., "worker_finding", "synthesis")
        content: Artifact content dict
        key: Optional custom key (auto-generated if None)
        check_duplicate: Whether to check for duplicates
    
    Returns:
        True if saved, False if skipped
    
    Raises:
        StorageError: If save operation fails
    """
    from ..exceptions import StorageError
    
    namespace = artifacts_namespace(session_id, artifact_type)
    namespace_str = namespace_to_str(namespace)
    
    try:
        # Build value with metadata
        value = {
            **content,
            "artifact_type": artifact_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Generate key if not provided
        if key is None:
            content_str = str(content)
            key = f"{artifact_type}_{compute_content_hash(content_str)}"
        
        # Save to store
        await store.aput(namespace, key, value, check_duplicate=check_duplicate)
        
        logger.info(
            f"[STORAGE] Saved artifact {key} in {namespace_str}"
        )
        return True
        
    except Exception as e:
        raise StorageError(
            f"Failed to save artifact ({artifact_type}) in {namespace_str}: {e}"
        ) from e


# =============================================================================
# RESEARCH BRIEF
# =============================================================================

async def save_research_brief(
    store: BaseStore,
    session_id: str,
    brief: Dict[str, Any],
) -> bool:
    """Save research brief to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        brief: Research brief dict
    
    Returns:
        True if saved
        
    Raises:
        StorageError: If save operation fails
    """
    return await save_artifact(
        store, session_id, "research_brief", brief, key="research_brief"
    )


async def load_research_brief(
    store: BaseStore,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Load research brief from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
    
    Returns:
        Research brief dict or None if not found
    """
    from ..exceptions import StorageError
    
    namespace = research_brief_namespace(session_id)
    
    try:
        item = await store.aget(namespace, "research_brief")
        if item and isinstance(item.value, dict):
            return item.value
        return None
    except Exception as e:
        raise StorageError(f"Failed to load research brief: {e}") from e


# =============================================================================
# SYNTHESIS
# =============================================================================

async def save_synthesis(
    store: BaseStore,
    session_id: str,
    synthesis: Dict[str, Any],
) -> bool:
    """Save final synthesis to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        synthesis: Synthesis dict with content, citations, etc.
    
    Returns:
        True if saved
    """
    return await save_artifact(
        store, session_id, "synthesis", synthesis, key="final_synthesis"
    )


async def load_synthesis(
    store: BaseStore,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Load final synthesis from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
    
    Returns:
        Synthesis dict or None if not found
    """
    from ..exceptions import StorageError
    
    namespace = synthesis_namespace(session_id)
    
    try:
        item = await store.aget(namespace, "final_synthesis")
        if item and isinstance(item.value, dict):
            return item.value
        return None
    except Exception as e:
        raise StorageError(f"Failed to load synthesis: {e}") from e


# =============================================================================
# CRITIQUE
# =============================================================================

async def save_critique(
    store: BaseStore,
    session_id: str,
    critique: Dict[str, Any],
) -> bool:
    """Save critique/evaluation to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        critique: Critique dict with evaluation, confidence, gaps
    
    Returns:
        True if saved
    """
    return await save_artifact(
        store, session_id, "critique", critique, key="latest_critique"
    )


async def load_critique(
    store: BaseStore,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Load latest critique from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
    
    Returns:
        Critique dict or None if not found
    """
    from ..exceptions import StorageError
    
    namespace = critique_namespace(session_id)
    
    try:
        item = await store.aget(namespace, "latest_critique")
        if item and isinstance(item.value, dict):
            return item.value
        return None
    except Exception as e:
        raise StorageError(f"Failed to load critique: {e}") from e


# =============================================================================
# BOSS DECISION
# =============================================================================

async def save_boss_decision(
    store: BaseStore,
    session_id: str,
    decision: Dict[str, Any],
) -> bool:
    """Save boss decision to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        decision: Decision dict with action, workers, reasoning
    
    Returns:
        True if saved
    """
    return await save_artifact(
        store, session_id, "boss_decision", decision, key="latest_decision"
    )


async def load_boss_decision(
    store: BaseStore,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Load latest boss decision from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
    
    Returns:
        Decision dict or None if not found
    """
    from ..exceptions import StorageError
    
    namespace = boss_decision_namespace(session_id)
    
    try:
        item = await store.aget(namespace, "latest_decision")
        if item and isinstance(item.value, dict):
            return item.value
        return None
    except Exception as e:
        raise StorageError(f"Failed to load boss decision: {e}") from e


# =============================================================================
# ARCHITECT FINDINGS
# =============================================================================

async def save_architect_finding(
    store: BaseStore,
    session_id: str,
    architect_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Save architect finding to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        architect_id: Architect identifier
        content: Finding content
        metadata: Optional metadata
    
    Returns:
        True if saved
    """
    return await save_worker_finding(
        store, session_id, architect_id, content, metadata
    )


async def load_architect_findings(
    store: BaseStore,
    session_id: str,
    architect_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Load architect findings from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        architect_id: Optional architect ID filter
        limit: Maximum results
    
    Returns:
        List of findings
    """
    return await load_worker_findings(
        store, session_id, architect_id, limit
    )


# =============================================================================
# SESSION METADATA
# =============================================================================

async def save_session_metadata(
    store: BaseStore,
    session_id: str,
    metadata: Dict[str, Any],
) -> bool:
    """Save session metadata to Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
        metadata: Metadata dict
    
    Returns:
        True if saved
    """
    from ..exceptions import StorageError
    
    namespace = session_metadata_namespace(session_id)
    
    try:
        await store.aput(namespace, "metadata", metadata)
        logger.info(f"[STORAGE] Saved session metadata for {session_id}")
        return True
    except Exception as e:
        raise StorageError(f"Failed to save session metadata: {e}") from e


async def load_session_metadata(
    store: BaseStore,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Load session metadata from Context Engine.
    
    Args:
        store: LangGraph Store instance
        session_id: Session UUID
    
    Returns:
        Metadata dict or None if not found
    """
    from ..exceptions import StorageError
    
    namespace = session_metadata_namespace(session_id)
    
    try:
        item = await store.aget(namespace, "metadata")
        if item and isinstance(item.value, dict):
            return item.value
        return None
    except Exception as e:
        raise StorageError(f"Failed to load session metadata: {e}") from e


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Worker findings
    "save_worker_finding",
    "save_worker_findings_batch",
    "load_worker_findings",
    # Citations
    "save_citation",
    "load_citations",
    # Generic artifacts
    "save_artifact",
    # Specific artifacts
    "save_research_brief",
    "load_research_brief",
    "save_synthesis",
    "load_synthesis",
    "save_critique",
    "load_critique",
    "save_boss_decision",
    "load_boss_decision",
    "save_architect_finding",
    "load_architect_findings",
    # Session metadata
    "save_session_metadata",
    "load_session_metadata",
]
