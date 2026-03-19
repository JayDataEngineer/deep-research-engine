"""Synthesis Node - LangGraph node for research synthesis.

This node wraps the SynthesisWorker for use in LangGraph workflows.
It extracts worker results from state and produces formatted output.
"""

from ..state import EngineState
from ..utils.logging import get_logger, get_loguru_logger
from ..workers.base import WorkerConfig, WorkerRegistry

logger = get_logger("nodes.synthesis")
loguru_logger = get_loguru_logger("synthesis")


async def synthesis_node(state: EngineState) -> EngineState:
    """Synthesis node for LangGraph workflow.

    This node:
    1. Extracts worker results from state
    2. Creates synthesis worker with config
    3. Executes synthesis
    4. Updates state with formatted output

    Args:
        state: Current engine state

    Returns:
        Updated state with synthesis_output
    """
    logger.info("Starting synthesis node", phase="synthesis")

    try:
        # Get synthesis config from state context
        synthesis_config = state.get("context", {}).get("synthesis", {})

        # Create worker config
        worker_config = WorkerConfig(
            worker_id="synthesis_0",
            worker_type="synthesis",
            task="Synthesize research findings into formatted output",
            tools=[],
            max_retries=synthesis_config.get("max_retries", 10),
            timeout=600,
            custom={
                "format": synthesis_config.get("format", "research_report"),
                "config": synthesis_config,
            },
        )

        # Get worker class from registry
        synthesis_worker_class = WorkerRegistry.get("synthesis")

        # Create and execute worker
        worker = synthesis_worker_class(worker_config)
        result = await worker.execute(state, worker_config)

        # Update state based on result
        if result.get("success", False):
            # Store output in state
            output = result.get("output")

            # Handle both BaseModel and dict outputs
            if hasattr(output, "model_dump_json"):
                state["synthesis_output"] = output.model_dump_json()
                state["synthesis_metadata"] = {
                    "format": synthesis_config.get("format", "research_report"),
                    "worker_count": len(state.get("worker_results", [])),
                    "citations_enabled": synthesis_config.get("citations", {}).get("enabled", True),
                    **(output.model_dump() if hasattr(output, "model_dump") else {}),
                }
            else:
                state["synthesis_output"] = str(output) if output else ""
                state["synthesis_metadata"] = {
                    "format": synthesis_config.get("format", "research_report"),
                    "worker_count": len(state.get("worker_results", [])),
                }

            state["current_phase"] = "complete"
            state["phase_history"].append("synthesis")

            logger.info(
                "Synthesis complete",
                output_type=result.get("output_type"),
                confidence=result.get("confidence"),
            )

        else:
            # Handle failure
            error_msg = result.get("error", "Unknown synthesis error")
            state["errors"].append(
                {
                    "phase": "synthesis",
                    "error": error_msg,
                    "worker_id": "synthesis_0",
                }
            )
            state["fatal_error"] = f"Synthesis failed: {error_msg}"

            logger.error(f"Synthesis failed: {error_msg}")

        return state

    except Exception as e:
        logger.error(f"Synthesis node error: {e}", exc_info=True)
        state["errors"].append(
            {
                "phase": "synthesis",
                "error": str(e),
                "type": type(e).__name__,
            }
        )
        state["fatal_error"] = f"Synthesis node crashed: {e!s}"
        return state


async def route_after_synthesis(state: EngineState) -> str:
    """Route decision after synthesis node.

    Returns:
        "end" - Workflow complete
    """
    if state.get("fatal_error"):
        return "error"
    return "end"
