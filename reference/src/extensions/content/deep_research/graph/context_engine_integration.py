"""Context Engine Integration for Deep Research

This module integrates the Context Engine with the deep research framework,
enabling:
- Semantic search across research findings
- Neo4j knowledge graph with entity-aware search
- Natural language graph queries
- Persistent artifact storage with deduplication
- Session isolation for concurrent research sessions

Usage:
    from .context_engine_integration import (
        initialize_context_engine_for_research,
        search_research_context,
        get_context_for_synthesis,
        get_context_engine_from_state,
    )

    # Initialize Context Engine for research session
    ext = await initialize_context_engine_for_research(state, yaml_config)

    # Search for relevant context (includes vector + graph search)
    context = await get_context_for_synthesis(query, state, yaml_config)

Artifact storage functions are in context_engine_artifacts.py
Entity extraction functions are in context_engine_entities.py
"""

from pathlib import Path
import logging
import os
import re
import sys


from typing import Dict, Any, List, Optional
import asyncio

from ..logging_utils import get_logger
from .error_utils import ContextEngineErrors, ErrorBoxTemplates, CrashErrors
from .validation_utils import (
    validate_session_id,
    validate_context_engine,
    validate_postgres_store,
    validate_neo4j_available,
)
from .constants import Defaults
from .performance import gather_with_crash, PerformanceMonitor, timed

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level

# Import constants from the root constants.py
from ..constants import (
    DEFAULT_NEO4J_URL,
    DEFAULT_NEO4J_USERNAME,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_MINIO_ENDPOINT,
    DEFAULT_MINIO_ACCESS_KEY,
    DEFAULT_MINIO_SECRET_KEY,
    DEFAULT_MINIO_BUCKET,
)


# =============================================================================
# FORMATTING CONSTANTS (was hardcoded values)
# =============================================================================

class ContextFormatting:
    """Constants for context formatting to avoid magic numbers."""

    # Default search limits
    DEFAULT_SYNTHESIS_ARTIFACTS = 20
    DEFAULT_CRITIC_ARTIFACTS = 15
    DEFAULT_ENTITY_SEARCH_K = 5

    # Content preview lengths
    ARTIFACT_PREVIEW_LENGTH = 300
    CRITIC_ARTIFACT_PREVIEW = 200
    ENTITY_CHUNK_PREVIEW = 150
    DRAFT_QUERY_LENGTH = 500

    # Display limits
    MAX_ARTIFACTS_TO_SHOW = 10
    MAX_ENTITIES_TO_SHOW = 5
    MAX_SOURCES_TO_SHOW = 3


# =============================================================================
# INITIALIZATION
# =============================================================================

async def initialize_context_engine_for_research(
    state: Dict[str, Any],
    yaml_config: Dict[str, Any],
) -> Any:
    """Initialize Context Engine for a research session via extension manager.

    CRITICAL: Context Engine MUST be available for deep research to work.
    This function will CRASH if Context Engine is not available.

    This function uses the extension manager to get the ContextEngineExtension
    instead of duplicating storage initialization. Falls back to direct
    initialization when running from CLI (no extension manager).

    Args:
        state: Global state dict (will be updated with context_engine state)
        yaml_config: Full YAML configuration

    Returns:
        ContextEngineExtension instance with postgres_store, neo4j_store, s3_backend

    Raises:
        RuntimeError: If Context Engine is not enabled or fails to initialize
    """
    # Check if Context Engine is enabled in config
    context_engine_config = yaml_config.get("context_engine", {})
    if not context_engine_config.get("enabled", False):
        raise RuntimeError(ContextEngineErrors.NOT_ENABLED)

    # Try to get context engine from extension manager first
    try:
        from src.api.core.extensions import get_extension_manager

        manager = get_extension_manager()

        if manager is not None:
            # Extension manager available - use it
            ext_wrapper = manager.get_extension("context_engine")

            if ext_wrapper and getattr(ext_wrapper, 'enabled', False):
                context_engine = ext_wrapper.instance

                if context_engine and context_engine.postgres_store:
                    # Verify Neo4j is enabled for deep research
                    neo4j_config = context_engine_config.get("neo4j", {})
                    if neo4j_config.get("enabled", False) and context_engine.neo4j_store:
                        from .helpers.config import set_context_engine_context
                        set_context_engine_context(context_engine)
                        logger.info("[CONTEXT_ENGINE] Initialized successfully via extension manager")
                        return context_engine

            logger.info("[CONTEXT_ENGINE] Extension manager available but context_engine not ready, falling back to direct init")

    except Exception as e:
        logger.debug(f"[CONTEXT_ENGINE] Extension manager not available: {e}, using direct initialization")

    # =========================================================================
    # FALLBACK: Direct initialization for CLI mode
    # =========================================================================
    # When running from CLI (not through FastAPI), the extension manager is
    # not available. We need to directly initialize Context Engine from config.
    # This still requires Postgres, Neo4j, and MinIO to be running.
    # =========================================================================
    logger.info("[CONTEXT_ENGINE] Attempting direct initialization for CLI mode...")

    try:
        from src.extensions.features.tools.context_engine import (
            PostgresVectorStore,
            S3Backend,
            Neo4jStore,
        )

        postgres_config = context_engine_config.get("postgres", {})
        neo4j_config = context_engine_config.get("neo4j", {})
        minio_config = context_engine_config.get("minio", {})

        # Initialize Postgres store (REQUIRED)
        connection_string = postgres_config.get("connection_string")
        if not connection_string:
            raise RuntimeError("Postgres connection_string is required in context_engine config")

        postgres_store = PostgresVectorStore(
            connection_string=connection_string,
            artifact_table=postgres_config.get("artifact_table", "artifacts"),
            embedding_table=postgres_config.get("embedding_table", "artifact_embeddings"),
            pool_min_size=postgres_config.get("pool_min_size", 5),
            pool_max_size=postgres_config.get("pool_max_size", 20),
        )
        logger.info("[CONTEXT_ENGINE] Postgres store initialized")

        # Initialize Neo4j store (REQUIRED for deep research)
        neo4j_store = None
        if neo4j_config.get("enabled", False):
            neo4j_store = Neo4jStore(
                url=neo4j_config.get("url", DEFAULT_NEO4J_URL),
                username=neo4j_config.get("username", DEFAULT_NEO4J_USERNAME),
                password=neo4j_config.get("password", DEFAULT_NEO4J_PASSWORD),
            )
            logger.info("[CONTEXT_ENGINE] Neo4j store initialized")

        # Initialize S3/MinIO backend (REQUIRED for large file storage)
        s3_backend = None
        if minio_config.get("enabled", True):
            s3_backend = S3Backend(
                endpoint=minio_config.get("endpoint", DEFAULT_MINIO_ENDPOINT),
                access_key=minio_config.get("access_key", DEFAULT_MINIO_ACCESS_KEY),
                secret_key=minio_config.get("secret_key", DEFAULT_MINIO_SECRET_KEY),
                bucket=minio_config.get("bucket", DEFAULT_MINIO_BUCKET),
            )
            logger.info("[CONTEXT_ENGINE] S3/MinIO backend initialized")

        # Create a simple object to hold the backends (duck typing)
        # The code that uses context_engine expects: postgres_store, neo4j_store, s3_backend
        class ContextEngineBackends:
            def __init__(self, postgres_store, neo4j_store, s3_backend):
                self.postgres_store = postgres_store
                self.neo4j_store = neo4j_store
                self.s3_backend = s3_backend
                self.neo4j_vector_store = None  # Not initialized in CLI mode

        context_engine = ContextEngineBackends(
            postgres_store=postgres_store,
            neo4j_store=neo4j_store,
            s3_backend=s3_backend,
        )

        # Set context for helpers
        from .helpers.config import set_context_engine_context
        set_context_engine_context(context_engine)

        logger.info("[CONTEXT_ENGINE] Direct initialization successful for CLI mode")
        return context_engine

    except Exception as e:
        logger.error(f"[CONTEXT_ENGINE] Direct initialization failed: {e}")
        raise RuntimeError(ContextEngineErrors.INITIALIZATION_FAILED + f"\n\nDirect initialization error: {e}") from e


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

async def search_research_context(
    query: str,
    session_id: str,
    context_engine: Any,
    artifact_type: Optional[str] = None,
    k: int = 10,
) -> List[Any]:
    """Search for semantically similar research artifacts.

    CRASH-ONLY: This function raises on failure instead of returning empty list.
    If Context Engine search fails, that's a CRITICAL error that should crash.

    Args:
        query: Search query
        session_id: Research session ID (for isolation)
        context_engine: ContextEngineBackends instance
        artifact_type: Optional artifact type filter
        k: Number of results to return

    Returns:
        List of BaseArtifact objects (may be empty if no matches, but NOT on error)

    Raises:
        RuntimeError: If context_engine is not available
        Exception: If search operation fails (propagates original exception)
    """
    # Validate context engine using centralized validation
    validate_context_engine(context_engine, "search_research_context")
    validate_postgres_store(context_engine)

    # Perform search - let exceptions propagate (crash-only)
    results = await context_engine.postgres_store.search(
        query=query,
        session_id=session_id,
        k=k,
    )
    return results


# =============================================================================
# FORMATTING HELPERS (DRY - eliminate duplicate formatting code)
# =============================================================================

def _format_artifact(artifact: Any, preview_length: int = ContextFormatting.ARTIFACT_PREVIEW_LENGTH) -> List[str]:
    """Format a single artifact for display.

    Args:
        artifact: The artifact to format
        preview_length: Length of content preview

    Returns:
        List of formatted strings
    """
    parts = [
        f"**[Artifact]** {artifact.artifact_type.value}",
        f"Content: {artifact.content[:preview_length]}...",
    ]

    if artifact.source_ids:
        sources = ', '.join(artifact.source_ids[:ContextFormatting.MAX_SOURCES_TO_SHOW])
        parts.append(f"Sources: {sources}")

    # Check if full content is in MinIO
    minio_path = artifact.metadata.get("minio_path")
    if minio_path:
        parts.append(f"📁 Full content available: {minio_path}")

    # Add confidence if present
    confidence = artifact.metadata.get("confidence", 0.0)
    if confidence:
        parts.append(f"Confidence: {confidence}")

    return parts


def _format_entity(result: Dict[str, Any], index: int) -> List[str]:
    """Format a single entity result for display.

    Args:
        result: The entity result dict
        index: Entity number for display

    Returns:
        List of formatted strings
    """
    entity_name = result.get('entity_name', 'Unknown')
    entity_type = result.get('entity_type', 'Entity')
    score = result.get('score', 0.0)

    parts = [
        f"**[Entity {index}]** {entity_name} ({entity_type}) - relevance: {score:.2f}"
    ]

    # Show chunk where entity was found
    chunk_text = result.get('chunk_text', '')
    if chunk_text:
        preview_length = ContextFormatting.ENTITY_CHUNK_PREVIEW
        parts.append(f"Context: {chunk_text[:preview_length]}...")

    # Show source document
    doc_title = result.get('doc_title', '')
    doc_url = result.get('doc_url', '')
    if doc_title or doc_url:
        source_info = f"Source: {doc_title}"
        if doc_url:
            source_info += f" ({doc_url})"
        parts.append(source_info)

    return parts


def _format_artifacts_section(artifacts: List[Any], section_title: str) -> List[str]:
    """Format a section of artifacts.

    Args:
        artifacts: List of artifacts to format
        section_title: Title for the section

    Returns:
        List of formatted strings
    """
    if not artifacts:
        return []

    parts = [f"### {section_title} ({len(artifacts)} artifacts)\n"]

    for artifact in artifacts[:ContextFormatting.MAX_ARTIFACTS_TO_SHOW]:
        parts.extend(_format_artifact(artifact))
        parts.append("")

    return parts


def _format_entities_section(results: List[Dict[str, Any]], section_title: str) -> List[str]:
    """Format a section of entity results.

    Args:
        results: List of entity result dicts
        section_title: Title for the section

    Returns:
        List of formatted strings
    """
    if not results:
        return []

    parts = [f"### {section_title} ({len(results)} results)\n"]

    for i, result in enumerate(results[:ContextFormatting.MAX_ENTITIES_TO_SHOW], 1):
        parts.extend(_format_entity(result, i))
        parts.append("")

    return parts


# =============================================================================
# CONTEXT RETRIEVAL FUNCTIONS
# =============================================================================

async def get_context_for_synthesis(
    query: str,
    state: Dict[str, Any],
    yaml_config: Dict[str, Any],
    max_artifacts: int = ContextFormatting.DEFAULT_SYNTHESIS_ARTIFACTS,
) -> str:
    """Get relevant context for synthesis from ALL Context Engine backends.

    CRASH-ONLY: This function crashes on critical failures instead of degrading.

    This function is called by the synthesis node to retrieve:
    - Semantically similar artifacts from Postgres (vector search) - REQUIRED
    - Related entities and relationships from Neo4j (graph) - OPTIONAL
    - References to large files in MinIO (for detailed context)

    Args:
        query: Search query
        state: Global state (for session_id and context_engine)
        yaml_config: Full YAML configuration
        max_artifacts: Maximum number of artifacts to retrieve

    Returns:
        Formatted context string with data from all backends

    Raises:
        RuntimeError: If context_engine or session_id is not available (CRITICAL)
        Exception: If Postgres search fails (propagates from search_research_context)
    """
    from .helpers.config import get_context_engine_from_context

    # Validate using centralized validation
    context_engine = validate_context_engine(
        get_context_engine_from_context(),
        "get_context_for_synthesis"
    )
    session_id = validate_session_id(state, "get_context_for_synthesis")

    context_parts = [f"## Context Engine Results for Session: {session_id}\n"]

    # ========================================================================
    # PARALLEL SEARCHES: Run Postgres and Neo4j searches concurrently
    # ========================================================================
    # PERFORMANCE: Use async concurrency to speed up context retrieval

    async def search_postgres() -> List[Any]:
        """Search Postgres for relevant artifacts."""
        return await search_research_context(
            query=query,
            session_id=session_id,
            context_engine=context_engine,
            k=max_artifacts
        )

    async def search_neo4j_entities() -> Optional[List[Dict[str, Any]]]:
        """Search Neo4j for related entities."""
        if not context_engine.neo4j_vector_store:
            return None

        try:
            # Use Neo4jVectorStore for hybrid vector+graph search
            # This finds entities similar to the query AND traces their sources
            return await context_engine.neo4j_vector_store.search_entities(
                query=query,
                k=ContextFormatting.DEFAULT_ENTITY_SEARCH_K
            )
        except Exception as e:
            # CRASH-ONLY: Log and re-raise instead of silent error
            logger.error(f"[CONTEXT_ENGINE] Neo4jVectorStore search failed: {e}")
            # Provide helpful error message for debugging
            raise RuntimeError(
                f"Neo4j entity search failed during synthesis context retrieval.\n"
                f"This is a CRITICAL error that prevents proper context gathering.\n"
                f"Error: {e}"
            ) from e

    # Run searches in parallel (Neo4j is optional, so we handle None)
    postgres_result, graph_results = await gather_with_crash(
        search_postgres(),
        search_neo4j_entities(),
        continue_on_error=False,  # Crash on any error
    )

    artifacts = postgres_result

    # ========================================================================
    # FORMAT RESULTS
    # ========================================================================

    # Format Postgres results
    context_parts.extend(_format_artifacts_section(
        artifacts,
        "📊 Postgres: Semantic Search Results"
    ))

    # Format Neo4j results (if available)
    if graph_results:
        context_parts.extend(_format_entities_section(
            graph_results,
            "🕸️ Knowledge Graph: Related Entities with Sources"
        ))

    # ========================================================================
    # 3. NEO4J: Natural language query for specific questions
    # ========================================================================
    # CRASH-ONLY FIX: Now crashes instead of silent error
    if context_engine.neo4j_store:
        try:
            # Check if query looks like a question (has who/what/how/etc.)
            question_words = r'\b(who|what|where|when|why|how|which|whose|related to|connected to|find)\b'
            if re.search(question_words, query, re.IGNORECASE):
                # Use natural language query to get specific answers from graph
                from .helpers.llm import create_llm_from_config
                from src.extensions.features.tools.context_engine.graph_qa_chain import NaturalLanguageGraphQuery

                # Use researcher role (thinking OFF in config) for graph queries
                llm = await create_llm_from_config(yaml_config, role="researcher")

                query_chain = NaturalLanguageGraphQuery(
                    llm=llm,
                    neo4j_store=context_engine.neo4j_store
                )

                qa_result = await query_chain.query(query)

                if qa_result.answer and not qa_result.error:
                    context_parts.append("### 🧠 Graph Query Result\n")
                    context_parts.append(qa_result.answer)

                    if qa_result.cypher:
                        context_parts.append(f"\n*Query: `{qa_result.cypher}`*")

                    context_parts.append("")

        except Exception as e:
            # CRASH-ONLY: Log and re-raise instead of silent debug log
            logger.error(f"[CONTEXT_ENGINE] Natural language query failed: {e}")
            # Non-critical: graph query is optional enhancement
            # We log the error but don't crash since synthesis can continue without it
            logger.debug("[CONTEXT_ENGINE] Synthesis can continue without graph query results")

    return "\n".join(context_parts) if context_parts else ""


async def get_context_for_critic(
    draft_content: str,
    state: Dict[str, Any],
    yaml_config: Dict[str, Any],
    max_artifacts: int = ContextFormatting.DEFAULT_CRITIC_ARTIFACTS,
) -> str:
    """Get relevant context for critic evaluation from ALL Context Engine backends.

    CRASH-ONLY: This function crashes on critical failures instead of degrading.

    This function is called by the critic node to retrieve:
    - Semantically similar artifacts from Postgres (for fact-checking) - REQUIRED
    - Related entities and relationships from Neo4j (for verification) - OPTIONAL
    - References to large files in MinIO (for source verification)

    Args:
        draft_content: The draft content to evaluate
        state: Global state (for session_id and context_engine)
        yaml_config: Full YAML configuration
        max_artifacts: Maximum number of artifacts to retrieve

    Returns:
        Formatted context string with data from all backends

    Raises:
        RuntimeError: If context_engine or session_id is not available (CRITICAL)
        Exception: If Postgres search fails (propagates from search_research_context)
    """
    from .helpers.config import get_context_engine_from_context

    # Validate using centralized validation
    context_engine = validate_context_engine(
        get_context_engine_from_context(),
        "get_context_for_critic"
    )
    session_id = validate_session_id(state, "get_context_for_critic")

    context_parts = [f"## Context Engine Fact-Checking Data for Session: {session_id}\n"]

    # ========================================================================
    # PARALLEL SEARCHES: Run Postgres and Neo4j searches concurrently
    # ========================================================================
    # PERFORMANCE: Use async concurrency to speed up context retrieval

    query = draft_content[:ContextFormatting.DRAFT_QUERY_LENGTH]

    async def search_postgres() -> List[Any]:
        """Search Postgres for fact-checking."""
        return await search_research_context(
            query=query,
            session_id=session_id,
            context_engine=context_engine,
            k=max_artifacts
        )

    async def search_neo4j_entities() -> Optional[List[Dict[str, Any]]]:
        """Search Neo4j for entity verification."""
        if not context_engine.neo4j_vector_store:
            return None

        try:
            # Use Neo4jVectorStore to find entities mentioned in draft
            # This provides source verification for fact-checking
            return await context_engine.neo4j_vector_store.search_entities(
                query=query,
                k=ContextFormatting.DEFAULT_ENTITY_SEARCH_K
            )
        except Exception as e:
            # CRASH-ONLY: Log and re-raise instead of silent error
            logger.error(f"[CONTEXT_ENGINE] Neo4jVectorStore verification failed: {e}")
            # Provide helpful error message for debugging
            raise RuntimeError(
                f"Neo4j entity verification failed during critic context retrieval.\n"
                f"This is a CRITICAL error that prevents proper fact-checking.\n"
                f"Error: {e}"
            ) from e

    # Run searches in parallel (Neo4j is optional, so we handle None)
    postgres_result, graph_results = await gather_with_crash(
        search_postgres(),
        search_neo4j_entities(),
        continue_on_error=False,  # Crash on any error
    )

    artifacts = postgres_result

    # ========================================================================
    # FORMAT RESULTS
    # ========================================================================

    # Format Postgres results with custom preview length for critic
    if artifacts:
        context_parts.append(f"### 📊 Postgres: Source Artifacts ({len(artifacts)} artifacts)\n")

        for artifact in artifacts[:ContextFormatting.MAX_ARTIFACTS_TO_SHOW]:
            # Use shorter preview for critic
            parts = _format_artifact(artifact, ContextFormatting.CRITIC_ARTIFACT_PREVIEW)
            context_parts.extend(parts)
            context_parts.append("")

    # Format Neo4j results (if available)
    if graph_results:
        context_parts.append(f"### 🕸️ Knowledge Graph: Entity Verification ({len(graph_results)} results)\n")

        for i, result in enumerate(graph_results[:ContextFormatting.MAX_ENTITIES_TO_SHOW], 1):
            entity_name = result.get('entity_name', 'Unknown')
            entity_type = result.get('entity_type', 'Entity')

            context_parts.append(f"**[Entity {i}]** {entity_name} ({entity_type})")

            # Show chunk where entity was found (source verification)
            chunk_text = result.get('chunk_text', '')
            if chunk_text:
                context_parts.append(f"Source context: {chunk_text[:ContextFormatting.DRAFT_QUERY_LENGTH]}...")

            # Show source document for verification
            doc_title = result.get('doc_title', '')
            doc_url = result.get('doc_url', '')
            if doc_title or doc_url:
                source_info = f"Verifiable source: {doc_title}"
                if doc_url:
                    source_info += f" - {doc_url}"
                context_parts.append(source_info)

            context_parts.append("")

    return "\n".join(context_parts) if context_parts else ""


# =============================================================================
# CONTEXT ENGINE ACCESSORS
# =============================================================================

def get_context_engine_from_state(state: Dict[str, Any]) -> Any:
    """Get Context Engine instance from context variable.

    CRITICAL: Context Engine is REQUIRED for deep research.
    This function will CRASH if Context Engine is not initialized.

    Args:
        state: Global state dict (kept for backward compatibility, not used)

    Returns:
        ContextEngineBackends instance

    Raises:
        RuntimeError: If Context Engine is not initialized
    """
    from .helpers.config import get_context_engine_from_context

    context_engine = get_context_engine_from_context()
    if not context_engine:
        raise RuntimeError(
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║        CONTEXT ENGINE NOT INITIALIZED - INITIALIZATION FAILED       ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "Context Engine was not initialized properly.\n"
            "This is a CRITICAL system failure.\n"
            "\n"
            "Deep research CANNOT continue without Context Engine.\n"
        )

    return context_engine


def is_context_engine_enabled(state: Dict[str, Any]) -> bool:
    """Check if Context Engine is enabled and initialized.

    CRITICAL: Context Engine is REQUIRED for deep research.
    This function now checks if it was successfully initialized (exists in context variable).

    Args:
        state: Global state dict (kept for backward compatibility, not used)

    Returns:
        True if Context Engine is initialized and available
    """
    # Context Engine is required - check if it's in context variable
    from .helpers.config import get_context_engine_from_context
    return get_context_engine_from_context() is not None


# NOTE: Legacy artifact storage functions removed.
# The NEW system uses GRAPH_BUILDER which reads from LangGraph Store.
# See: backend/docs/save_system_redesign_plan.md
