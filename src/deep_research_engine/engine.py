"""Main engine entry point."""

import uuid
from collections.abc import AsyncIterator
from typing import Any

from .config import WorkflowConfig
from .nodes import boss_node, critic_node, synthesis_node, workers_node
from .nodes.boss import route_after_boss
from .nodes.critic import route_after_critic
from .nodes.synthesis import route_after_synthesis
from .nodes.workers import route_after_workers
from .state import EngineState, create_initial_state
from .storage import StorageManager
from .utils.logging import get_logger, log_phase_complete, log_phase_start, setup_logging


async def run_workflow(
    input_data: str | dict[str, Any],
    config: WorkflowConfig,
    session_id: str | None = None,
    stream: bool = True,
    storage_backends: list | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run a workflow with the given configuration.

    Args:
        input_data: Query string or dict with workflow input
        config: Workflow configuration
        session_id: Optional session ID for tracing
        stream: If True, yield events as they happen
        storage_backends: List of backends to enable

    Yields:
        Events (if streaming) or final result
    """
    # Setup logging
    setup_logging(level=config.execution.get("logging", "info"))
    logger = get_logger("engine")

    # Generate IDs
    run_id = str(uuid.uuid4())
    session_id = session_id or run_id

    logger.info(
        f"Starting workflow: {config.workflow['name']}",
        run_id=run_id,
        session_id=session_id,
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
    # Handle phases as list (from YAML) or dict
    phases_config = config.phases
    boss_config = {}
    if isinstance(phases_config, list):
        # Find boss phase in list
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

    # Build and run graph
    try:
        if stream:
            async for event in _run_graph_streaming(state, config):
                yield event
        else:
            final_state = await _run_graph_sync(state, config)
            yield {
                "type": "complete",
                "success": True,
                "output": final_state.get("synthesis_output"),
                "state": final_state,
            }
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        yield {
            "type": "error",
            "error": str(e),
            "phase": "execution",
        }
    finally:
        await storage.cleanup()


async def _run_graph_streaming(state: EngineState, config: WorkflowConfig):
    """Run graph with streaming."""
    logger = get_logger("engine.graph")

    # Execute phases sequentially (can be parallelized later)
    phases = ["boss", "workers", "critic", "synthesis"]

    for phase in phases:
        log_phase_start(phase)
        yield {"type": "phase_start", "phase": phase}

        try:
            # Execute phase node
            if phase == "boss":
                state = await boss_node(state)
                next_phase = route_after_boss(state)
            elif phase == "workers":
                state = await workers_node(state)
                next_phase = route_after_workers(state)
            elif phase == "critic":
                state = await critic_node(state)
                next_phase = route_after_critic(state)

                # Handle critic decision
                if next_phase == "workers":
                    # Retry workers
                    log_phase_start("workers", {"reason": "critic_revision"})
                    yield {"type": "phase_retry", "phase": "workers", "reason": "critic_revision"}
                    state = await workers_node(state)
                    next_phase = "critic"  # Re-evaluate
                elif next_phase == "error":
                    # Max retries exceeded, proceed to synthesis anyway
                    logger.warning(
                        "Critic rejected but max retries reached, proceeding to synthesis"
                    )
                    next_phase = "synthesis"

            elif phase == "synthesis":
                state = await synthesis_node(state)
                next_phase = route_after_synthesis(state)

            log_phase_complete(phase)
            yield {
                "type": "phase_complete",
                "phase": phase,
                "duration": 0.0,  # TODO: Track actual duration
            }

            # Check for errors
            if next_phase == "error":
                logger.error(f"Workflow failed at phase: {phase}")
                yield {
                    "type": "error",
                    "error": state.get("fatal_error", f"Failed at phase: {phase}"),
                    "phase": phase,
                }
                return

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


async def _run_graph_sync(state: EngineState, config: WorkflowConfig) -> EngineState:
    """Run graph synchronously."""
    logger = get_logger("engine.graph")

    # Execute phases in order
    state = await boss_node(state)
    if route_after_boss(state) == "error":
        raise RuntimeError(state.get("fatal_error", "Boss node failed"))

    state = await workers_node(state)
    if route_after_workers(state) == "error":
        raise RuntimeError(state.get("fatal_error", "Workers node failed"))

    # Critic loop
    max_iterations = config.phases.get("critic", {}).get("config", {}).get("max_retries", 3)
    for iteration in range(max_iterations):
        state = await critic_node(state)
        next_phase = route_after_critic(state)

        if next_phase == "synthesis":
            break
        elif next_phase == "workers":
            # Retry workers
            logger.info(f"Retrying workers (iteration {iteration + 1}/{max_iterations})")
            state = await workers_node(state)
        else:
            raise RuntimeError(state.get("fatal_error", "Critic node failed"))

    # Synthesis
    state = await synthesis_node(state)
    if route_after_synthesis(state) == "error":
        raise RuntimeError(state.get("fatal_error", "Synthesis node failed"))

    return state
