"""Deep Research Graph - Main entry point with run_v4_spawner.

This module provides the main entry point for the deep research system.
The graph building logic has been split into separate modules:

- graph_factory.py: Graph builder (build_v4_graph) with retry/cache policies

Modern LangGraph Patterns Implemented:
- Command API: Combined state updates + control flow
- Runtime Configuration: Dynamic parameter injection
- Input/Output Schemas: Clean API boundaries

SESSION MANAGEMENT:
All research runs use context_engine session tools for tracking.
This provides:
- Single source of truth for run metadata
- Cross-system data lineage (Neo4j, Postgres, MinIO all tagged with session_id)
- Rollback and cleanup capabilities
- Audit trails for reproducibility
- NO direct database connections - all via context_engine
"""

import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

# Add backend to path for absolute imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=5)

# Import state and types
from src.extensions.content.deep_research.graph.state import (
    GlobalState,
    ResearchInput,
    ResearchOutput,
    ResearchRuntimeConfig,
)

# Import nodes
from src.extensions.content.deep_research.graph.nodes import (
    boss_node,
    lab_architect_node,
    critic_node,
    formatter_node,
    unified_synthesis_node,
)

from src.extensions.content.deep_research.profiles import load_profile
from src.extensions.content.deep_research.storage import (
    create_research_filesystem,
    set_filesystem_context,
    reset_filesystem_context,
    get_default_config,
)
from src.extensions.content.deep_research.graph.helpers.config import (
    _load_yaml_config,
    _apply_config_overrides,
    _apply_overdrive_mode,
    get_config_from_runnable_config,
    get_run_logger_from_config,
    set_run_logger_context,
    reset_run_logger_context,
    set_yaml_config_context,
    reset_yaml_config_context,
    get_yaml_config_from_context,
    set_context_engine_context,
    reset_context_engine_context,
)

# Import routing functions
from src.extensions.content.deep_research.graph.graph_routing import (
    route_from_synthesis,
    route_from_critic,
)

# Import utility functions
from src.extensions.content.deep_research.graph.graph_utilities import (
    get_execution_history,
    format_checkpoint_summary,
    get_checkpoint_at_step,
    resume_from_checkpoint,
    debug_alternative_path,
    print_execution_tree,
)

# Import session tools from context_engine (via registry)
from src.shared.tools.registry import get_tool

# Set up logging
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

async def run_v4_spawner(
    query: str,
    use_case: str = "rag",
    variant: str = "default",
    chat_history: list = None,
    run_logger = None,
    config_overrides: dict = None,
    overdrive: bool = False,
    no_preset: bool = False,
    runtime_config: ResearchRuntimeConfig = None,
    store=None,
    checkpointer=None,
    durability: str = "async",
    max_checkpoint_retries: int = 3,
) -> Dict[str, Any]:
    """Run the V4 Modern LangGraph spawner with durable execution.

    CRASH-ONLY: Store and Checkpointer are REQUIRED. NO FALLBACKS.

    Args:
        query: User query
        use_case: Use case name
        variant: Config variant
        chat_history: Optional conversation history
        run_logger: **REQUIRED** RunLogger instance for metrics and observability
        config_overrides: Optional dict of config overrides for grid search testing
        overdrive: Enable overdrive mode (max quality, max context, deeper research)
        no_preset: Disable preset graphs, use LLM-generated dynamic graph instead
        runtime_config: Optional ResearchRuntimeConfig for dynamic parameter injection
        store: **REQUIRED** BaseStore for cross-thread persistence and semantic search
        checkpointer: **REQUIRED** AsyncPostgresSaver for checkpoint recovery
        durability: Durability mode - "exit", "async", or "sync" (default: "async")
        max_checkpoint_retries: Max retries with checkpoint restoration on failure (default: 3)

    Returns:
        Final state with response
    """
    # Import build_v4_graph from graph_factory
    from .graph_factory import build_v4_graph

    # ==============================================================================
    # CRITICAL VALIDATION - RunLogger is REQUIRED
    # ==============================================================================
    if run_logger is None:
        raise ValueError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║                    CRITICAL ERROR: RunLogger Missing               ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "run_v4_spawner() REQUIRES a RunLogger instance to be explicitly provided.\n"
            "The system cannot run without proper observability and metrics tracking.\n"
            "\n"
            "REQUIRED USAGE:\n"
            "----------------\n"
            "from src.extensions.content.deep_research.observability import RunLogger\n"
            "\n"
            "run_logger = RunLogger(\n"
            "    run_id='my_run',\n"
            "    use_case='rag',\n"
            "    output_dir='/path/to/output',\n"
            ")\n"
            "\n"
            "result = await run_v4_spawner(\n"
            "    query='Your query here',\n"
            "    run_logger=run_logger,  # REQUIRED\n"
            ")\n"
        )

    # ==============================================================================
    # CRITICAL VALIDATION - Store is REQUIRED
    # ==============================================================================
    if store is None:
        raise ValueError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║                    CRITICAL ERROR: Store Missing                    ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "run_v4_spawner() REQUIRES a BaseStore instance for:\n"
            "  - Cross-thread persistence\n"
            "  - Information Funnel artifact storage\n"
            "  - Semantic search and retrieval\n"
            "  - Context Engine operations\n"
            "\n"
            "REQUIRED USAGE:\n"
            "----------------\n"
            "from src.shared.checkpointer.factory import get_checkpointer, get_store\n"
            "\n"
            "# Set POSTGRES_URL environment variable\n"
            "os.environ['POSTGRES_URL'] = 'postgresql://user:pass@localhost:5433/dbname'\n"
            "\n"
            "checkpointer = get_checkpointer()\n"
            "store = get_store()\n"
            "\n"
            "result = await run_v4_spawner(\n"
            "    query='Your query here',\n"
            "    run_logger=run_logger,  # REQUIRED\n"
            "    checkpointer=checkpointer,  # REQUIRED\n"
            "    store=store,  # REQUIRED\n"
            ")\n"
        )

    # ==============================================================================
    # CRITICAL VALIDATION - Infrastructure Dependencies
    # ==============================================================================
    async def _validate_infrastructure():
        """Validate that required infrastructure (Postgres, MCP) is available."""
        errors = []

        # Check Postgres connectivity
        try:
            import asyncpg
            postgres_url = os.environ.get('POSTGRES_URL', '')
            if not postgres_url:
                errors.append("POSTGRES_URL environment variable not set")
            else:
                # Quick connection test with 5 second timeout
                conn = await asyncio.wait_for(
                    asyncpg.connect(postgres_url),
                    timeout=5.0
                )
                await conn.close()
        except asyncio.TimeoutError:
            errors.append("Postgres connection timed out (5s) - is the database running?")
        except Exception as e:
            errors.append(f"Postgres connection failed: {str(e)[:100]}")

        # Check MCP Web Server availability via Service Registry
        try:
            import httpx
            from src.shared.services.registry import get_registry

            registry = get_registry()
            mcp_url = registry.get_url("mcp_web")

            async with httpx.AsyncClient(timeout=10.0) as client:
                # MCP server uses /health endpoint for health check
                response = await client.get(f"{mcp_url}/../health", timeout=10.0, follow_redirects=True)
                # Health endpoint should return 200
        except asyncio.TimeoutError:
            errors.append(f"MCP server at {mcp_url} timed out (10s) - is the server running?")
        except httpx.ConnectError:
            errors.append(f"MCP server at {mcp_url} not reachable - is the server running?")
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            errors.append(f"MCP server check failed: {error_msg}")

        if errors:
            error_list = "\n".join(f"  ❌ {e}" for e in errors)
            raise ValueError(
                "\n"
                "╔═══════════════════════════════════════════════════════════════════╗\n"
                "║              INFRASTRUCTURE VALIDATION FAILED                      ║\n"
                "╚═══════════════════════════════════════════════════════════════════╝\n"
                "\n"
                "Deep Research requires the following services to be running:\n"
                "\n"
                f"{error_list}\n"
                "\n"
                "QUICK FIX:\n"
                "----------\n"
                "# Ensure MCP Web Server is running\n"
                "# Check MCP_WEB_SERVER_URL env var or Service Registry\n"
                "\n"
                "# Ensure Postgres is running:\n"
                "docker compose up -d postgres\n"
                "# Or set POSTGRES_URL to your database\n"
            )

    await _validate_infrastructure()

    # ==============================================================================
    # STEP 1: CONFIGURATION LOADING
    # ==============================================================================
    logger.info("[DEEP_RESEARCH] Starting V4 spawner")
    logger.info(f"[DEEP_RESEARCH] Query: {query[:100]}...")
    logger.info(f"[DEEP_RESEARCH] Use case: {use_case}, Variant: {variant}")

    # Load the YAML configuration
    yaml_config = _load_yaml_config(use_case, variant)
    logger.info(f"[CONFIG] Loaded config: {yaml_config.get('name', 'unknown')}")

    # Apply config overrides if provided
    if config_overrides:
        yaml_config = _apply_config_overrides(yaml_config, config_overrides)
        logger.info(f"[CONFIG] Applied overrides: {config_overrides}")

    # Apply overdrive mode if enabled
    if overdrive:
        yaml_config = _apply_overdrive_mode(yaml_config)
        logger.info("[CONFIG] Applied overdrive mode")

    # Store use_case and variant in config for observability
    yaml_config["_use_case"] = use_case
    yaml_config["_variant"] = variant

    # ==============================================================================
    # STEP 2: SET UP CONTEXT
    # ==============================================================================
    # Set the run_logger in context (store token for cleanup)
    run_logger_token = set_run_logger_context(run_logger)

    # Set yaml_config in context (store token for cleanup)
    yaml_config_token = set_yaml_config_context(yaml_config)

    # Create and set the virtual filesystem
    # Note: runtime is None here since we're outside graph execution
    # The filesystem will be set up for later use by nodes
    from deepagents.backends.state import StateBackend
    from deepagents.backends.composite import CompositeBackend
    fs = CompositeBackend(default=StateBackend(None), routes={})
    fs_token = set_filesystem_context(fs)

    # ==============================================================================
    # STEP 3: LOAD PROFILE
    # ==============================================================================
    profile = load_profile(use_case, variant, config=yaml_config)
    logger.info(f"[PROFILES] Loaded profile from extension: {use_case} -> {profile.name}")

    # ==============================================================================
    # STEP 4: BUILD THE GRAPH
    # ==============================================================================
    graph = build_v4_graph(checkpointer=checkpointer, store=store)

    # ==============================================================================
    # STEP 4.5: RESOLVE TOOL MODEL PROVIDER
    # ==============================================================================
    # Bind the tool model provider to state for sub-agents/tools to use
    # This allows tools like web_research_agent and pdf_deep_agent to use
    # fast local models instead of the main cloud model
    tool_model_provider = None
    try:
        from src.extensions.content.deep_research.graph.helpers.llm import _resolve_model_from_slot

        tool_model_info = _resolve_model_from_slot(yaml_config, "tool", "tool_binding")
        tool_model_provider = tool_model_info.get("provider")
        logger.info(f"[DEEP_RESEARCH] Bound tool model provider: {tool_model_provider}")
    except Exception as e:
        logger.warning(f"[DEEP_RESEARCH] Failed to resolve tool model provider: {e}")
        # Continue without tool model provider - tools will use default

    # ==============================================================================
    # STEP 5: CREATE INITIAL STATE
    # ==============================================================================
    from src.extensions.content.deep_research.graph.state import create_global_state

    initial_state = create_global_state(
        query=query,
        chat_history=chat_history,
        use_case=use_case,
        variant=variant,
        tool_model_provider=tool_model_provider,
    )

    # ==============================================================================
    # STEP 5.5: INITIALIZE CONTEXT ENGINE
    # ==============================================================================
    # Context Engine provides semantic search and persistent artifact storage
    # CRITICAL: This will CRASH if Context Engine or Neo4j is not available
    from .context_engine_integration import initialize_context_engine_for_research

    context_engine = await initialize_context_engine_for_research(
        state=initial_state,
        yaml_config=yaml_config
    )

    logger.info("[DEEP_RESEARCH] Context Engine initialized successfully with Neo4j")

    # Store context_engine in context variable (NOT in state - causes serialization issues)
    # LangGraph checkpoints use msgpack which can't serialize database connections
    context_engine_token = set_context_engine_context(context_engine)

    # ==============================================================================
    # STEP 6: CREATE SESSION AND PREPARE GRAPH INVOCATION
    # ==============================================================================
    # Create research session using context_engine tools
    # This creates a UUID-based session_id that is the PRIMARY KEY across all storage
    # NO direct database connections - all via context_engine

    # Get session tools from registry
    create_research_session = get_tool("create_research_session")
    update_session_status = get_tool("update_session_status")

    # Build config for session creation (exclude circular refs)
    session_config = dict(yaml_config)
    session_config.pop("_run_logger", None)
    session_config.pop("_yaml_config", None)

    # Create session via context_engine
    session_result = await create_research_session.ainvoke({
        "query": query,
        "use_case": use_case,
        "variant": variant,
        "run_id": run_logger.run_id,
        "config": session_config,
        "runtime": None,  # No runtime available at this level
    })

    # Parse session result
    session_data = json.loads(session_result)
    session_id = session_data.get("session_id")
    thread_id = session_data.get("thread_id")

    if not session_id:
        raise RuntimeError("Failed to create session: no session_id returned")

    # Log session creation
    logger.info(f"[SESSION] Created session {session_id} for run {run_logger.run_id}")
    logger.info(f"[SESSION] thread_id: {thread_id}")

    # Set RESEARCH_SESSION_ID for Context Engine tools
    # This is the "Golden Thread" that connects all storage systems
    os.environ["RESEARCH_SESSION_ID"] = session_id
    logger.info(f"[SESSION] Set RESEARCH_SESSION_ID={session_id}")

    # Store session_id in state for tools to access
    initial_state["session_id"] = session_id
    initial_state["run_id"] = run_logger.run_id

    invoke_config = {
        "configurable": {
            "thread_id": thread_id,
            # Store session_id for observability and recovery
            "session_id": session_id,
            # Store use_case and variant in config for observability (config IS persisted)
            "_use_case": use_case,
            "_variant": variant,
        }
    }

    # NOTE: runtime_config is accessed via langgraph's Runtime[ResearchRuntimeConfig]
    # dependency injection in node functions, NOT via invoke_config
    # This prevents serialization issues with int values in the config

    logger.info(f"[DEEP_RESEARCH] Invoking graph with thread_id: {thread_id}, session_id: {session_id}")

    # Invoke the graph (async) with automatic status updates
    try:
        final_state = await graph.ainvoke(initial_state, config=invoke_config)
        logger.info(f"[SESSION] Session {session_id} completed successfully")

        # Mark session as completed
        await update_session_status.ainvoke({
            "session_id": session_id,
            "status": "completed",
            "runtime": None
        })

    except Exception as e:
        # Mark session as failed
        logger.error(f"[SESSION] Session {session_id} failed: {e}")
        await update_session_status.ainvoke({
            "session_id": session_id,
            "status": "failed",
            "error": str(e),
            "runtime": None
        })
        raise
    finally:
        # Clear environment variable (good hygiene)
        os.environ.pop("RESEARCH_SESSION_ID", None)

    # ==============================================================================
    # STEP 7: CLEANUP CONTEXT
    # ==============================================================================
    reset_run_logger_context(run_logger_token)
    reset_yaml_config_context(yaml_config_token)
    reset_filesystem_context(fs_token)
    reset_context_engine_context(context_engine_token)

    # ==============================================================================
    # STEP 8: RETURN RESULTS
    # ==============================================================================
    logger.info("[DEEP_RESEARCH] Spawner completed successfully")

    return final_state


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main entry point
    "run_v4_spawner",
    # Graph builder (re-exported from graph_factory)
    "build_v4_graph",
]
