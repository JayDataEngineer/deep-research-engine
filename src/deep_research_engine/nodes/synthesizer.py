import asyncio
from typing import Any, List, Dict
from ..state import EngineState, WorkerResult
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config
from ..workers import WorkerConfig, WorkerRegistry

logger = get_loguru_logger("nodes.synthesizer")


@node_context("synthesizer")
async def synthesizer_node(state: EngineState) -> dict[str, Any]:
    """Synthesizer node - Formats the final research output."""
    logger.info("Starting synthesizer node")

    # Get config and check if enabled
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "synthesizer")
    synth_enabled = phase_config.get("enabled", True)

    if not synth_enabled:
        logger.info("Synthesizer node disabled. Skipping.")
        return {"__goto__": "complete"}

    # 1. Try to use SynthesisWorker if available
    worker_class = WorkerRegistry.get("synthesis")
    if worker_class:
        try:
            worker_config = WorkerConfig(
                worker_id="synthesizer_worker",
                worker_type="synthesis",
                task="Synthesize research results",
                custom=phase_config
            )
            worker = worker_class(worker_config)
            worker_result = await worker.execute(state, worker_config)
            
            if worker_result.success:
                logger.info("Synthesis worker completed successfully.")
                return {
                    "synthesis_output": state.get("synthesis_output"),
                    "synthesis_metadata": {
                        **state.get("synthesis_metadata", {}),
                        "worker_count": len(state.get("worker_results", []))
                    },
                    "__goto__": "complete"
                }
            else:
                logger.warning(f"Synthesis worker failed: {worker_result.error}. Falling back to basic synthesis.")
        except Exception as e:
            logger.error(f"Error executing synthesis worker: {e}. Falling back.")

    # 2. Basic fallback synthesis
    worker_results = state.get("worker_results", [])
    successful_results = [r for r in worker_results if r.get("success")]

    if not successful_results:
        logger.warning("No successful worker results to synthesize.")
        return {
            "synthesis_output": "No research results were generated.",
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

    final_output = "# Final Research Synthesis\n\n" + "\n\n".join(content_parts)
    
    logger.info(f"Basic synthesis complete. Output length: {len(final_output)} chars.")
    return {
        "synthesis_output": final_output,
        "synthesis_metadata": {
            "format": "basic_aggregation",
            "worker_count": len(successful_results)
        },
        "__goto__": "complete"
    }

