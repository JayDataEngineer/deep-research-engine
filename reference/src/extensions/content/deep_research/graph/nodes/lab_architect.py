"""Lab Architect - Pro-Flash hybrid orchestration for deep research.

Uses unified_flash_explorer_spawner() pattern:
- Plans research avenues from worker tasks
- Spawns Flash Explorers in parallel
- Synthesizes results into LabArchitectFindings (Pydantic validated)
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

# Import logger from deep_research package (absolute import for consistency)
from src.extensions.content.deep_research.logging_utils import get_logger
logger = get_logger(__name__)

# Import shared types and helpers from common module
from .common import (
    GlobalState,
    ResearchRuntimeConfig,
    get_run_logger_from_config,
    get_yaml_config_from_context,
    unified_flash_explorer_spawner,
)

from ..helpers.config import get_context_engine_from_context

# Import streaming helpers for SSE events
from ..helpers.streaming import (
    get_writer,
    emit_phase_change,
    emit_lab_summary,
)

# Import logging decorators for standardized phase logging
from ..decorators import research_node, Phase


@research_node(Phase.LAB)
async def lab_architect_node(
    state: GlobalState,
    writer: Optional[Any] = None,
    *,
    runtime: Runtime[ResearchRuntimeConfig],
    store: Optional[Any] = None,
    config: RunnableConfig | None = None,
) -> Dict[str, Any]:
    """Lab Architect - Pro-Flash hybrid orchestration for deep research.

    Orchestrates parallel Flash Explorers for iterative research:
    1. Plans research avenues from worker tasks
    2. Spawns Flash Explorers in parallel
    3. Synthesizes results into LabArchitectFindings (Pydantic validated)

    Args:
        state: Global state containing query, worker_tasks, etc.
        writer: LangGraph StreamWriter (passed as second arg for streaming)
        runtime: Runtime context
        store: Optional BaseStore
        config: LangGraph RunnableConfig (contains stream writer as fallback)

    Returns:
        Dict with architect_findings and current_phase="critic"
    """
    start_time = datetime.now()

    # DEBUG: Log all injection sources for store
    logger.info(f"[LAB_ARCHITECT] store kwarg: {type(store).__name__ if store else 'None'}")
    logger.info(f"[LAB_ARCHITECT] runtime type: {type(runtime).__name__ if runtime else 'None'}")

    # Get store from runtime if available (LangGraph Runtime pattern)
    # The Runtime object contains the store when graph is compiled with store=
    runtime_store = None
    if runtime and hasattr(runtime, 'store'):
        runtime_store = runtime.store
        logger.info(f"[LAB_ARCHITECT] runtime.store type: {type(runtime_store).__name__ if runtime_store else 'None'}")

    # Priority: kwarg > runtime.store > config.store
    store_from_config = config.get("store") if config else None
    effective_store = store or runtime_store or store_from_config

    if effective_store:
        logger.info(f"[LAB_ARCHITECT] Using effective store: {type(effective_store).__name__}")
    else:
        logger.warning(f"[LAB_ARCHITECT] No store available - sources will NOT be persisted/vectorized")

    yaml_config = get_yaml_config_from_context()
    run_logger = get_run_logger_from_config(config)

    # Get stream writer for SSE events (from positional arg or config)
    stream_writer = get_writer(writer, config)

    # Emit phase change event at start
    emit_phase_change(stream_writer, "lab", "Research", "Executing research tasks...", "active")

    # Get context_engine from context (for validation only, not stored in yaml_config)
    # NOTE: context_engine is NOT stored in yaml_config because it's not msgpack serializable
    # unified_flash_explorer_spawner retrieves it from context directly
    context_engine = get_context_engine_from_context()

    # Use node context manager for hierarchical tracking
    # Pass effective_store to ensure sources get persisted
    if run_logger and hasattr(run_logger, 'node'):
        async with run_logger.node("Lab Architect", "spawner"):
            return await _lab_architect_impl(state, yaml_config, run_logger, start_time, stream_writer, effective_store)
    else:
        return await _lab_architect_impl(state, yaml_config, run_logger, start_time, stream_writer, effective_store)


async def _lab_architect_impl(
    state: Dict[str, Any],
    yaml_config: Dict[str, Any],
    run_logger: Any,
    start_time: datetime,
    stream_writer: Any = None,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """Implementation of Lab Architect (wrapped by node tracking)."""
    session_id = state.get("session_id")

    if run_logger:
        run_logger._write_to_file("[LAB_ARCHITECT] Starting Lab Architect node")

    # Get worker tasks from Boss
    worker_tasks = state.get("worker_tasks", {})

    if not worker_tasks:
        # CRASH-ONLY: Lab Architect requires worker tasks from Boss
        raise RuntimeError(
            "[LAB_ARCHITECT] No worker tasks from Boss! "
            "This indicates Boss node failed or was bypassed. "
            "Lab Architect cannot proceed without worker tasks."
        )

    # Import LabArchitectFindings model for synthesis
    from ..state_models import LabArchitectFindings

    # Build input for planning (worker_tasks as JSON string)
    import json
    input_for_planning = json.dumps(worker_tasks, indent=2)

    if run_logger:
        run_logger._write_to_file(f"[LAB_ARCHITECT] Processing {len(worker_tasks)} worker tasks")

    # Call unified Flash Explorer spawner
    # Passes worker_tasks as input and LabArchitectFindings as synthesis model
    result = await unified_flash_explorer_spawner(
        state=state,
        next_phase="critic",
        output_key="architect_findings",
        synthesis_model=LabArchitectFindings,
        input_for_planning=input_for_planning,
        run_logger=run_logger,
        store=store,  # Pass LangGraph Store for memory persistence
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    if run_logger:
        run_logger._write_to_file(
            f"[LAB_ARCHITECT] Completed in {elapsed:.1f}s"
        )

    # Emit phase change complete event
    emit_phase_change(stream_writer, "lab", "Research", "Research tasks complete", "complete")

    # Emit lab summary event for frontend
    architect_findings = result.get("architect_findings", {})
    worker_summaries = []
    if architect_findings and hasattr(architect_findings, 'get'):
        # Extract summaries from findings if available
        worker_summaries = architect_findings.get("worker_summaries", [])
    elif isinstance(architect_findings, dict):
        worker_summaries = architect_findings.get("worker_summaries", [])

    emit_lab_summary(
        writer=stream_writer,
        workers_completed=len(worker_summaries) if worker_summaries else 1,
        worker_summaries=worker_summaries
    )

    # Return state compatible with rest of flow
    result["worker_outputs"] = []  # Empty list - Lab Architect handles research internally
    return result
