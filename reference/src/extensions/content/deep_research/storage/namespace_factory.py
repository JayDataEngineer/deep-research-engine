"""Namespace Factory - Consistent namespace construction for Context Engine storage.

This module provides a single source of truth for building namespace tuples
used across all Context Engine storage operations (Postgres, Neo4j, MinIO).

Usage:
    from deep_research.storage.namespace_factory import (
        findings_namespace,
        citations_namespace,
        artifacts_namespace,
    )
    
    # Build namespace for worker findings
    ns = findings_namespace(session_id="uuid-123", worker_id="philosopher-1")
    # Returns: ("uuid-123", "memories", "findings", "philosopher-1")
    
    # Build namespace for citations
    ns = citations_namespace(session_id="uuid-123")
    # Returns: ("uuid-123", "memories", "citations")
"""

from typing import Tuple, Optional, Literal


# =============================================================================
# NAMESPACE FACTORY FUNCTIONS
# =============================================================================

def findings_namespace(
    session_id: str,
    worker_id: Optional[str] = None,
) -> Tuple[str, ...]:
    """Build namespace for worker findings.
    
    Args:
        session_id: Session UUID (single source of truth across all storage)
        worker_id: Optional worker ID for filtering (e.g., "philosopher-1")
    
    Returns:
        Namespace tuple for LangGraph Store
        
    Examples:
        >>> findings_namespace("uuid-123")
        ('uuid-123', 'memories', 'findings')
        
        >>> findings_namespace("uuid-123", "philosopher-1")
        ('uuid-123', 'memories', 'findings', 'philosopher-1')
    """
    if worker_id:
        return (session_id, "memories", "findings", worker_id)
    return (session_id, "memories", "findings")


def citations_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for citations.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for citations
        
    Examples:
        >>> citations_namespace("uuid-123")
        ('uuid-123', 'memories', 'citations')
    """
    return (session_id, "memories", "citations")


def artifacts_namespace(
    session_id: str,
    artifact_type: str,
) -> Tuple[str, ...]:
    """Build namespace for artifacts.
    
    Args:
        session_id: Session UUID
        artifact_type: Type of artifact (e.g., "worker_finding", "synthesis", "brief")
    
    Returns:
        Namespace tuple for artifacts
        
    Examples:
        >>> artifacts_namespace("uuid-123", "worker_finding")
        ('uuid-123', 'artifacts', 'worker_finding')
    """
    return (session_id, "artifacts", artifact_type)


def research_brief_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for research brief.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for research brief
        
    Examples:
        >>> research_brief_namespace("uuid-123")
        ('uuid-123', 'artifacts', 'research_brief')
    """
    return (session_id, "artifacts", "research_brief")


def synthesis_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for final synthesis.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for synthesis
        
    Examples:
        >>> synthesis_namespace("uuid-123")
        ('uuid-123', 'artifacts', 'synthesis')
    """
    return (session_id, "artifacts", "synthesis")


def critique_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for critique/evaluation.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for critique
        
    Examples:
        >>> critique_namespace("uuid-123")
        ('uuid-123', 'artifacts', 'critique')
    """
    return (session_id, "artifacts", "critique")


def boss_decision_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for boss decisions.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for boss decisions
        
    Examples:
        >>> boss_decision_namespace("uuid-123")
        ('uuid-123', 'artifacts', 'boss_decision')
    """
    return (session_id, "artifacts", "boss_decision")


def architect_findings_namespace(
    session_id: str,
    architect_id: Optional[str] = None,
) -> Tuple[str, ...]:
    """Build namespace for architect findings.
    
    Args:
        session_id: Session UUID
        architect_id: Optional architect ID for filtering
    
    Returns:
        Namespace tuple for architect findings
        
    Examples:
        >>> architect_findings_namespace("uuid-123")
        ('uuid-123', 'memories', 'architect_findings')
        
        >>> architect_findings_namespace("uuid-123", "scout-1")
        ('uuid-123', 'memories', 'architect_findings', 'scout-1')
    """
    if architect_id:
        return (session_id, "memories", "architect_findings", architect_id)
    return (session_id, "memories", "architect_findings")


def session_metadata_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for session metadata.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for session metadata
        
    Examples:
        >>> session_metadata_namespace("uuid-123")
        ('uuid-123', 'metadata', 'session')
    """
    return (session_id, "metadata", "session")


def graph_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for knowledge graph data.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for knowledge graph
        
    Examples:
        >>> graph_namespace("uuid-123")
        ('uuid-123', 'graph', 'knowledge')
    """
    return (session_id, "graph", "knowledge")


def logs_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for execution logs.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for logs
        
    Examples:
        >>> logs_namespace("uuid-123")
        ('uuid-123', 'logs', 'execution')
    """
    return (session_id, "logs", "execution")


def metrics_namespace(session_id: str) -> Tuple[str, ...]:
    """Build namespace for performance metrics.
    
    Args:
        session_id: Session UUID
        
    Returns:
        Namespace tuple for metrics
        
    Examples:
        >>> metrics_namespace("uuid-123")
        ('uuid-123', 'metrics', 'performance')
    """
    return (session_id, "metrics", "performance")


# =============================================================================
# NAMESPACE UTILITIES
# =============================================================================

def namespace_to_str(namespace: Tuple[str, ...]) -> str:
    """Convert namespace tuple to string for display/logging.
    
    Args:
        namespace: Namespace tuple
        
    Returns:
        String representation (e.g., "session-123/memories/findings")
        
    Examples:
        >>> namespace_to_str(("uuid-123", "memories", "findings"))
        'uuid-123/memories/findings'
    """
    return "/".join(namespace)


def str_to_namespace(namespace_str: str) -> Tuple[str, ...]:
    """Convert string back to namespace tuple.
    
    Args:
        namespace_str: String representation
        
    Returns:
        Namespace tuple
        
    Examples:
        >>> str_to_namespace("uuid-123/memories/findings")
        ('uuid-123', 'memories', 'findings')
    """
    return tuple(namespace_str.split("/"))


def namespace_prefix(session_id: str, category: str) -> Tuple[str, ...]:
    """Build namespace prefix for searching within a category.
    
    Used for semantic search across all items in a category.
    
    Args:
        session_id: Session UUID
        category: Category name (e.g., "memories", "artifacts", "graph")
    
    Returns:
        Namespace prefix tuple for search
        
    Examples:
        >>> namespace_prefix("uuid-123", "memories")
        ('uuid-123', 'memories')
    """
    return (session_id, category)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core namespaces
    "findings_namespace",
    "citations_namespace",
    "artifacts_namespace",
    "research_brief_namespace",
    "synthesis_namespace",
    "critique_namespace",
    "boss_decision_namespace",
    "architect_findings_namespace",
    "session_metadata_namespace",
    "graph_namespace",
    "logs_namespace",
    "metrics_namespace",
    # Utilities
    "namespace_to_str",
    "str_to_namespace",
    "namespace_prefix",
]
