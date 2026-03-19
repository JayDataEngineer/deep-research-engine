"""Graph Factory - LangGraph V4 graph builder with retry and cache policies.

This module contains the build_v4_graph function and associated policies
for building the deep research graph with modern LangGraph V4 patterns.

Separated from graph.py to reduce file size and improve modularity.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Any, Dict

import pydantic

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy, CachePolicy
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.cache.memory import InMemoryCache

from .state import GlobalState, ResearchInput, ResearchOutput, ResearchRuntimeConfig
from .nodes import (
    lab_architect_node,
    formatter_node,
)
# Unified nodes (current architecture - same code for Boss, Critic, Synthesis)
from .nodes.unified import (
    unified_boss_node,
    unified_critic_node,
    unified_synthesis_node,
)
# Graph Builder node - builds knowledge graph from research findings
from .nodes.graph_builder import graph_builder_node
# Note: No longer using conditional edges - all routing via Command API
from ..logging_utils import get_logger
# Graph execution tracer for observability
from ..observability.graph_tracer import wrap_node_with_tracing, get_tracer, reset_tracer


logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


# =============================================================================
# CHECKPOINTER VALIDATION - Prevent Data Loss
# =============================================================================

class InMemoryCheckpointerError(Exception):
    """Raised when an in-memory checkpointer would be used, causing data loss.

    In-memory checkpointers (MemorySaver, InMemoryStore) do not persist data
    to disk. This means:
    - All research runs are lost on restart
    - Observability UI cannot show runs
    - No time-travel debugging across sessions
    - Cannot resume interrupted research

    Always use a persistent checkpointer (SQLiteSaver) for production.
    """
    pass


def _validate_checkpointer(checkpointer) -> None:
    """Validate that AsyncPostgresSaver is being used.

    Args:
        checkpointer: The checkpointer to validate

    Raises:
        InMemoryCheckpointerError: If checkpointer is None or MemorySaver
    """
    if checkpointer is None:
        raise InMemoryCheckpointerError(
            "❌ DATA LOSS RISK: No checkpointer provided!\n\n"
            "Using no checkpointer means:\n"
            "  • All research runs are lost when the process restarts\n"
            "  • Observability UI cannot display your runs\n"
            "  • No checkpoint/time-travel debugging\n"
            "  • Cannot resume interrupted research\n\n"
            "Solution: Use the factory to get AsyncPostgresSaver:\n"
            "  from src.shared.checkpointer.factory import get_checkpointer\n"
            "  checkpointer = get_checkpointer()\n"
            "  # Set POSTGRES_URL environment variable\n"
        )

    # NO FALLBACKS: Reject MemorySaver entirely
    if isinstance(checkpointer, MemorySaver):
        raise InMemoryCheckpointerError(
            "❌ DATA LOSS RISK: MemorySaver is not allowed!\n\n"
            "The research framework requires AsyncPostgresSaver for:\n"
            "  • Persistent storage across runs\n"
            "  • Observability UI access\n"
            "  • Checkpoint recovery and debugging\n"
            "  • Context Engine integration\n\n"
            "Solution: Use the factory to get AsyncPostgresSaver:\n"
            "  from src.shared.checkpointer.factory import get_checkpointer\n"
            "  checkpointer = get_checkpointer()\n"
            "  # Set POSTGRES_URL environment variable\n"
        )


# =============================================================================
# RETRY POLICIES (Declarative retry logic per node type)
# =============================================================================

# LLM nodes: Retry on rate limits, timeouts, and connection errors
LLM_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(Exception,),  # Fallback to catching all exceptions
)

# Critic node: Retry on JSON parsing and Pydantic validation errors
# Uses with_structured_output() which raises pydantic.ValidationError on parse failure
# Also uses parse_critique_json which raises ValueError on JSON parse failure
CRITIC_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(pydantic.ValidationError, ValueError),  # Retry on validation and JSON parse errors
)

# Synthesis node: Conservative retries (already expensive)
# Also uses structured output, so retry on Pydantic validation and JSON parse errors
SYNTHESIS_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(pydantic.ValidationError, ValueError),  # Retry on validation and JSON parse errors
)

# Lab Architect node: Retry on Pydantic validation and JSON parse errors (structured output)
LAB_ARCHITECT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(pydantic.ValidationError, ValueError),  # Retry on validation and JSON parse errors
)

# Graph Builder node: Retry on Neo4j/GDS connection errors
GRAPH_BUILDER_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(ConnectionError, TimeoutError, Exception),  # Retry on connection/timeout issues
)


# =============================================================================
# CACHE POLICIES (Node-level caching)
# =============================================================================

# Boss initial planning cache: Cache brief generation for similar queries (1 hour)
# NOTE: key_func must return str or bytes, not int (hash() returns int)
BOSS_INITIAL_CACHE_POLICY = CachePolicy(
    ttl=3600,
    key_func=lambda s: s.get("query", "") or s.get("raw_query", ""),
)

# Boss routing cache: Cache routing decisions (30 minutes)
# NOTE: key_func must return str or bytes, not int (hash() returns int)
BOSS_ROUTING_CACHE_POLICY = CachePolicy(
    ttl=1800,
    key_func=lambda s: f"{s.get('query', '')}:{s.get('raw_query', '')}:{s.get('research_brief', '')}",
)

# Lab Architect cache: Cache Flash Explorer results (20 minutes)
# NOTE: key_func must return str or bytes, not int (hash() returns int)
LAB_CACHE_POLICY = CachePolicy(
    ttl=1200,
    key_func=lambda s: f"{s.get('query', '')}:{s.get('workers_to_spawn', '')}",
)


# =============================================================================
# GRAPH BUILDER WITH MODERN LANGGRAPH PATTERNS
# =============================================================================

def build_v4_graph(checkpointer=None, cache=None, store=None) -> StateGraph:
    """Build the V4 Modern LangGraph Architecture graph.

    NEW PATTERNS IMPLEMENTED:
    1. Command API: Combined state updates + control flow
    2. Send API: Map-reduce pattern for parallel workers
    3. Runtime Configuration: Dynamic parameter injection
    4. Input/Output Schemas: Clean API boundaries
    5. Retry Policies: Declarative retry logic
    6. Cache Policies: Node-level caching
    7. Store: Cross-thread persistence with semantic search
    8. Checkpointer: REQUIRED for observability and data persistence

    Args:
        checkpointer: **REQUIRED** Persistent checkpointer (SQLiteSaver recommended).
            MemorySaver and None will raise InMemoryCheckpointerError to prevent data loss.
        cache: Optional cache backend for node-level caching
        store: Optional BaseStore for cross-thread persistence (default: InMemoryStore)

    Returns:
        Compiled StateGraph ready for invoke()

    Raises:
        InMemoryCheckpointerError: If checkpointer is None or MemorySaver (data loss risk)
    """
    # ========================================================================
    # VALIDATE CHECKPOINTER - Prevent Data Loss
    # ========================================================================
    _validate_checkpointer(checkpointer)

    # ========================================================================
    # BUILD STATEGRAPH WITH INPUT/OUTPUT SCHEMAS
    # ========================================================================
    # NOTE: We use GlobalState + context_schema.
    # We don't use input_schema/output_schema to avoid conflicts with GlobalState's
    # Annotated reducers. The context_schema is needed for the Runtime[ResearchRuntimeConfig]
    # parameter in node functions.
    builder = StateGraph(GlobalState, context_schema=ResearchRuntimeConfig)

    # ========================================================================
    # ADD NODES WITH RETRY AND CACHE POLICIES
    # ========================================================================
    # Boss handles both initial planning and iterative routing
    # All inference nodes use unified draft_extract pattern
    #
    # OBSERVABILITY: Nodes are wrapped with tracing when DEEP_RESEARCH_TRACE=1
    # This provides real-time visibility into graph execution

    # Reset tracer for new graph build
    reset_tracer()

    # Boss node: Routes to lab_architect or synthesis
    builder.add_node(
        "boss",
        wrap_node_with_tracing("boss", unified_boss_node),
        retry_policy=LLM_RETRY_POLICY,
        cache_policy=BOSS_ROUTING_CACHE_POLICY,
    )

    # Lab architect: Pro-Flash hybrid orchestration for deep research
    builder.add_node(
        "lab_architect",
        wrap_node_with_tracing("lab_architect", lab_architect_node),
        retry_policy=LAB_ARCHITECT_RETRY_POLICY,
        cache_policy=LAB_CACHE_POLICY,
    )

    # Graph Builder: Builds knowledge graph from research findings
    # Runs AFTER lab_architect, BEFORE synthesis
    # Creates Neo4j entities, communities, and embeddings
    builder.add_node(
        "graph_builder",
        wrap_node_with_tracing("graph_builder", graph_builder_node),
        retry_policy=GRAPH_BUILDER_RETRY_POLICY,
    )

    # Critic node: Quality evaluation
    builder.add_node(
        "critic",
        wrap_node_with_tracing("critic", unified_critic_node),
        retry_policy=CRITIC_RETRY_POLICY,
    )

    # Synthesis node: Final output generation
    builder.add_node(
        "synthesis",
        wrap_node_with_tracing("synthesis", unified_synthesis_node),
        retry_policy=SYNTHESIS_RETRY_POLICY,
    )

    # Formatter node: Post-synthesis content formatting
    # NOTE: With unified_synthesis_node, formatting is built-in.
    # This node is kept for backwards compatibility and can be skipped.
    builder.add_node(
        "formatter",
        wrap_node_with_tracing("formatter", formatter_node),
        retry_policy=RetryPolicy(
            max_attempts=2,
            backoff_factor=2.0,
            jitter=True,
            retry_on=(Exception,),  # Retry on any formatting error
        ),
    )

    # ========================================================================
    # DEFINE EDGES
    # ========================================================================
    # Simplified flow: START -> boss -> (lab_architect OR synthesis) -> critic -> boss/END
    # Boss handles both initial planning and routing decisions
    #
    # NOTE: All nodes use Command API for routing with goto parameter.
    # Do NOT add static or conditional edges - they interfere with Command API.
    # Command(goto="node_name") takes precedence over edges.

    # Start: -> boss (no more worker)
    builder.add_edge(START, "boss")

    # Boss routes via Command API: goto="lab_architect" or goto="synthesis"
    # NO EDGE HERE - Command API handles routing

    # lab_architect -> graph_builder (builds knowledge graph from findings)
    builder.add_edge("lab_architect", "graph_builder")

    # graph_builder -> synthesis (graph_builder uses Command API to goto synthesis)
    # NO EDGE HERE - Command API handles routing

    # Synthesis routes via Command API: goto="critic"
    # NO EDGE HERE - Command API handles routing

    # Critic routes via Command API: goto="boss" or goto="formatter"
    # NO EDGE HERE - Command API handles routing

    # Formatter -> END (formatter is the terminal node)
    builder.add_edge("formatter", END)

    # ========================================================================
    # COMPILE GRAPH WITH CHECKPOINTER, CACHE, AND STORE
    # ========================================================================
    compile_kwargs = {}

    # Checkpointer: Use factory for environment-aware selection
    if checkpointer:
        actual_checkpointer = checkpointer
        if hasattr(checkpointer, 'checkpointer'):
            actual_checkpointer = checkpointer.checkpointer
        logger.info(f"[DEEP_RESEARCH] Compiling graph with custom checkpointer: {type(actual_checkpointer).__name__}")
        compile_kwargs["checkpointer"] = actual_checkpointer
    else:
        # Checkpointer is required - should always be passed from build_research_graph()
        raise InMemoryCheckpointerError(
            "checkpointer is required for Deep Research. "
            "Use build_research_graph() which awaits get_shared_checkpointer()."
        )

    # Cache: Enable InMemoryCache by default for node-level caching
    if cache:
        compile_kwargs["cache"] = cache
        logger.info(f"[DEEP_RESEARCH] Compiling graph with custom cache backend: {type(cache).__name__}")
    else:
        # Check for Valkey/Redis URL for distributed caching (multi-instance deployments)
        valkey_url = os.getenv("VALKEY_URL") or os.getenv("REDIS_URL")
        if valkey_url:
            try:
                from langgraph.cache.redis import RedisCache
                import redis

                # Use RedisCache (works with both Valkey and Redis)
                redis_client = redis.from_url(
                    valkey_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                default_cache = RedisCache(redis_client)
                logger.info(f"[DEEP_RESEARCH] Compiling graph with RedisCache (distributed caching via VALKEY_URL)")
            except ImportError:
                logger.warning("[DEEP_RESEARCH] Redis client not available, falling back to InMemoryCache")
                default_cache = InMemoryCache()
            except Exception as e:
                logger.warning(f"[DEEP_RESEARCH] Failed to connect to Valkey/Redis: {e}")
                logger.info("[DEEP_RESEARCH] Falling back to InMemoryCache")
                default_cache = InMemoryCache()
        else:
            # Use InMemoryCache by default to enable node-level caching
            # This caches expensive operations like LLM calls and web searches
            default_cache = InMemoryCache()
            logger.info("[DEEP_RESEARCH] Compiling graph with InMemoryCache (node-level caching enabled)")

        compile_kwargs["cache"] = default_cache

    # Store: Use InMemoryStore as default (can be overridden via parameter)
    if store:
        compile_kwargs["store"] = store
        logger.info(f"[DEEP_RESEARCH] Compiling graph with custom Store: {type(store).__name__}")
    else:
        # Use InMemoryStore for cross-thread persistence
        logger.info("[DEEP_RESEARCH] Using InMemoryStore for cross-thread persistence")
        default_store = InMemoryStore()

        compile_kwargs["store"] = default_store

    return builder.compile(**compile_kwargs)
