from typing import Any

from ..config import get_phase_config
from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..workers import WorkerConfig, WorkerRegistry

logger = get_loguru_logger("nodes.artifact_maker")


@node_context("artifact_maker")
async def artifact_maker_node(state: EngineState) -> dict[str, Any]:
    """Artifact Maker node - Formats the final research output into an artifact."""
    logger.info("Starting artifact maker node")

    # Get config and check if enabled
    config_data = state.get("context", {}).get("config", {})
    # Fallback to 'synthesizer' for backwards compatibility
    phase_config = get_phase_config(config_data, "artifact_maker")
    if not phase_config:
        phase_config = get_phase_config(config_data, "synthesizer")

    artifact_maker_enabled = phase_config.get("enabled", True)

    if not artifact_maker_enabled:
        logger.info("Artifact maker node disabled. Skipping.")
        return {"__goto__": "complete"}

    # 1. Try to use ArtifactMakerWorker if available
    worker_class = WorkerRegistry.get("artifact_maker")
    # Fallback for backwards compatibility
    if not worker_class:
        worker_class = WorkerRegistry.get("synthesis")

    if worker_class:
        try:
            worker_config = WorkerConfig(
                worker_id="artifact_maker_worker",
                worker_type="artifact_maker",
                task="Produce research artifact",
                custom=phase_config
            )
            worker = worker_class(worker_config)
            worker_result = await worker.execute(state, worker_config)

            if worker_result.get("success"):
                logger.info("Artifact maker worker completed successfully.")

                # Check for either key to be safe
                output = state.get("artifact_output") or state.get("synthesis_output")
                metadata = state.get("artifact_metadata") or state.get("synthesis_metadata", {})

                return {
                    "artifact_output": output,
                    "artifact_metadata": {
                        **metadata,
                        "worker_count": len(state.get("worker_results", []))
                    },
                    "__goto__": "complete"
                }
            else:
                logger.warning(f"Artifact maker worker failed: {worker_result.error}. Falling back to basic aggregation.")
        except Exception as e:
            logger.error(f"Error executing artifact maker worker: {e}. Falling back.")

    # 2. Basic fallback aggregation
    worker_results = state.get("worker_results", [])
    successful_results = [r for r in worker_results if r.get("success")]

    if not successful_results:
        logger.warning("No successful worker results to turn into an artifact.")
        return {
            "artifact_output": "No research results were generated.",
            "__goto__": "complete"
        }

    # Aggregate results
    content_parts = []
    for res in successful_results:
        output = res.get("output")
        if isinstance(output, str):
            content_parts.append(output)
        else:
            content_parts.append(str(output))

    final_output = "# Final Research Artifact\n\n" + "\n\n".join(content_parts)

    logger.info(f"Basic aggregation complete. Output length: {len(final_output)} chars.")
    return {
        "artifact_output": final_output,
        "artifact_metadata": {
            "format": "basic_aggregation",
            "worker_count": len(successful_results)
        },
        "__goto__": "complete"
    }
