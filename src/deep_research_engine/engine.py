"""Main engine entry point."""

import uuid
from collections.abc import AsyncIterator
from typing import Any

from .config import WorkflowConfig
from .graph import compile_workflow_graph, get_checkpointer, get_store
from .nodes import (
    critic_node,
    drafter_node,
    ingestion_node,
    neo4j_builder_node,
    route_after_critic,
    route_after_drafter,
    route_after_ingestion,
    route_after_neo4j_builder,
    route_after_synthesizer,
    synthesizer_node,
)
from .state import EngineState, create_initial_state
from .storage import StorageManager
from .utils.logging import get_loguru_logger, log_phase_complete, log_phase_start, setup_logging

logger = get_loguru_logger("engine")


async def run_workflow(
    input_data: str | dict[str, Any],
    config: WorkflowConfig,
    session_id: str | None = None,
    stream: bool = True,
    storage_backends: list | None = None,
    resume_from: str | None = None,
    use_stategraph: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Run a workflow with the given configuration.

    Args:
        input_data: Query string or dict with workflow input
        config: Workflow configuration
        session_id: Optional session ID for tracing
        stream: If True, yield events as they happen
        storage_backends: List of backends to enable
        resume_from: Optional session ID to resume from checkpoint
        use_stategraph: If True, use StateGraph; otherwise use legacy manual loop

    Yields:
        Events (if streaming) or final result
    """
    # Setup logging
    setup_logging(level=config.execution.get("logging", "info"))
    logger = get_loguru_logger("engine")

    # Generate IDs
    run_id = str(uuid.uuid4())
    session_id = session_id or run_id

    # For resume, use the provided session_id
    if resume_from:
        session_id = resume_from

    logger.info(
        f"Starting workflow: {config.workflow['name']} | Session: {session_id} | Run: {run_id}"
        f"{'(resuming)' if resume_from else ''}"
    )

    # Initialize storage
    storage = StorageManager()
    backends_to_enable = storage_backends or list(config.storage.keys())

    for backend_name in backends_to_enable:
        backend_config = config.storage.get(backend_name, {})
        if backend_config.get("enabled", False):
            try:
                await storage.initialize(backend_name, backend_config)
                logger.info(f"Initialized storage: {backend_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {backend_name}: {e}")

    # Create initial state
    phases_config = config.phases
    boss_config = {}
    if isinstance(phases_config, list):
        for phase in phases_config:
            if phase.get("name") == "boss":
                boss_config = phase.get("config", {})
                break
    elif isinstance(phases_config, dict):
        boss_config = phases_config.get("boss", {}).get("config", {})

    max_iterations = boss_config.get("workers", {}).get("max", 3)

    state = create_initial_state(
        workflow_name=config.workflow["name"],
        workflow_type=config.workflow["type"],
        input_data={"query": input_data} if isinstance(input_data, str) else input_data,
        session_id=session_id,
        run_id=run_id,
        max_iterations=max_iterations,
    )
    # Store config in context for nodes to access
    state["context"] = state.get("context", {})
    state["context"]["config"] = config.model_dump()

    # Build and run graph
    try:
        # Choose execution path
        if use_stategraph and _should_use_stategraph(config):
            async for event in _run_stategraph(
                state, config, session_id, stream=stream, resume_from=resume_from
            ):
                yield event
        else:
            async for event in _run_legacy_loop(state, config):
                yield event

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        yield {
            "type": "error",
            "error": str(e),
            "phase": "execution",
        }
    finally:
        await storage.cleanup()


def _should_use_stategraph(config: WorkflowConfig) -> bool:
    """Check if StateGraph should be used based on config."""
    execution_config = config.execution or {}

    # Explicit flag takes precedence
    if "use_stategraph" in execution_config:
        return execution_config["use_stategraph"]

    # Check if checkpointing is enabled
    storage_config = config.storage
    return not storage_config.get("postgres", {}).get("checkpoint", False)


async def _run_stategraph(
    state: EngineState,
    config: WorkflowConfig,
    session_id: str,
    resume_from: str | None = None,
    stream: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Run workflow using StateGraph with checkpointing and store."""
    logger = get_loguru_logger("engine.stategraph")

    async with get_checkpointer(config) as checkpointer, get_store(config) as store:
        # Compile graph with persistence
        graph = compile_workflow_graph(config, checkpointer=checkpointer, store=store)

        logger.info("Compiled workflow graph with checkpointing and store")

        # Configure thread for checkpointing
        thread_config = {"configurable": {"thread_id": session_id}}

        # Handle resume from checkpoint
        if resume_from:
            logger.info(f"Resuming workflow from checkpoint: {resume_from}")
            # State will be loaded from checkpoint by the graph
            # No need to pass initial state

        # Run graph
        if stream:
            async for event in graph.astream(
                state,
                thread_config,
                stream_mode=["updates", "custom"],
            ):
                yield _transform_stategraph_event(event)
        else:
            result = await graph.ainvoke(state, thread_config)
            yield {
                "type": "complete",
                "success": True,
                "output": result.get("synthesis_output"),
                "state": result,
            }


async def _run_legacy_loop(
    state: EngineState, config: WorkflowConfig
) -> AsyncIterator[dict[str, Any]]:
    """Run workflow using legacy manual loop."""
    logger = get_loguru_logger("engine.legacy")

    # Execute phases sequentially
    phases = ["ingestion", "drafter", "critic", "neo4j_builder", "synthesizer"]

    # Initial state phase
    state["current_phase"] = "ingestion"

    while state["current_phase"] in phases:
        phase = state["current_phase"]
        log_phase_start(phase)
        yield {"type": "phase_start", "phase": phase}

        try:
            if phase == "ingestion":
                state = await ingestion_node(state)
                next_phase = route_after_ingestion(state)
            elif phase == "drafter":
                state = await drafter_node(state)
                next_phase = route_after_drafter(state)
            elif phase == "critic":
                state = await critic_node(state)
                next_phase = route_after_critic(state)

                # Check if it was a retry loop
                if next_phase == "drafter":
                    log_phase_start("drafter", {"reason": "critic_revision"})
                    yield {"type": "phase_retry", "phase": "drafter", "reason": "critic_revision"}
            elif phase == "neo4j_builder":
                state = await neo4j_builder_node(state)
                next_phase = route_after_neo4j_builder(state)
            elif phase == "synthesizer":
                state = await synthesizer_node(state)
                next_phase = route_after_synthesizer(state)
            else:
                next_phase = "error"

            log_phase_complete(phase)
            yield {
                "type": "phase_complete",
                "phase": phase,
                "duration": 0.0,
            }

            if next_phase == "error":
                logger.error(f"Workflow failed at phase: {phase}")
                yield {
                    "type": "error",
                    "error": state.get("fatal_error", f"Failed at phase: {phase}"),
                    "phase": phase,
                }
                return
            elif next_phase == "complete":
                break

            state["current_phase"] = next_phase

        except Exception as e:
            logger.error(f"Phase {phase} failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "phase": phase,
            }
            return

    # Workflow complete
    yield {
        "type": "complete",
        "success": True,
        "output": state.get("synthesis_output"),
        "state": state,
    }


def _transform_stategraph_event(event: dict) -> dict[str, Any]:
    """Transform LangGraph events to our standard event format."""
    # Handle different event types from LangGraph
    if isinstance(event, dict):
        # Stream mode: updates
        if "node" in event:
            node_name = event["node"]
            if "updates" in event:
                updates = event["updates"]
                phase = updates.get("current_phase", node_name)

                # Emit phase start/complete based on phase transitions
                return {
                    "type": "phase_update",
                    "phase": phase,
                    "node": node_name,
                    "updates": updates,
                }
        # Custom events (e.g., from nodes)
        elif "type" in event:
            return event

    # Fallback: pass through unknown event structure
    return event
