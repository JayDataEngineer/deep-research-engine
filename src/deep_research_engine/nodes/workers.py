"""Workers Node - LangGraph node for parallel worker execution.

This node executes worker agents in parallel based on the plan
created by the boss node.
"""

import asyncio
from typing import Any

from ..state import EngineState
from ..utils.logging import get_logger, get_loguru_logger, log_worker_spawn
from ..workers.base import WorkerConfig, WorkerRegistry

logger = get_logger("nodes.workers")
loguru_logger = get_loguru_logger("workers")


async def workers_node(state: EngineState) -> EngineState:
    """Workers node for LangGraph workflow.

    This node:
    1. Gets plan from state
    2. Creates worker configs for each task
    3. Executes workers in parallel
    4. Accumulates results in state

    Args:
        state: Current engine state

    Returns:
        Updated state with worker_results
    """
    logger.info("Starting workers node", phase="workers")

    try:
        # Get plan
        plan = state.get("plan", [])
        if not plan:
            logger.warning("No plan found, skipping workers")
            return state

        # Get worker config
        workers_config = state.get("context", {}).get("workers", {})
        worker_type = workers_config.get("worker_type", "web_researcher")
        parallel = workers_config.get("parallel", True)
        max_concurrent = workers_config.get("max_concurrent", 10)

        # Create worker tasks
        worker_tasks = []
        for idx, task in enumerate(plan):
            worker_config = WorkerConfig(
                worker_id=f"{worker_type}_{idx}",
                worker_type=worker_type,
                task=task,
                tools=[],
                max_retries=3,
                timeout=300,
                custom={},
            )
            log_worker_spawn(
                worker_id=f"{worker_type}_{idx}",
                task=task if isinstance(task, str) else str(task),
                worker_num=idx + 1,
            )
            worker_tasks.append(execute_worker(worker_config, state))

        # Execute workers
        if parallel:
            # Limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def bounded_execute(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(*[bounded_execute(t) for t in worker_tasks])
        else:
            results = []
            for task in worker_tasks:
                result = await task
                results.append(result)

        # Accumulate results in state
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]

        # Use reducer pattern for accumulation
        state["worker_results"] = state.get("worker_results", []) + successful_results
        if failed_results:
            state["worker_errors"] = state.get("worker_errors", []) + failed_results

        state["current_phase"] = "critic"
        state["phase_history"].append("critic")

        logger.info(
            "Workers complete",
            total=len(results),
            successful=len(successful_results),
            failed=len(failed_results),
        )

        return state

    except Exception as e:
        logger.error(f"Workers node error: {e}", exc_info=True)
        state["errors"].append(
            {
                "phase": "workers",
                "error": str(e),
                "type": type(e).__name__,
            }
        )
        return state


async def execute_worker(config: WorkerConfig, state: EngineState) -> dict[str, Any]:
    """Execute a single worker.

    Args:
        config: Worker configuration
        state: Current engine state

    Returns:
        Worker result dict
    """
    try:
        # Get worker class from registry
        worker_class = WorkerRegistry.get(config.worker_type)
        worker = worker_class(config)

        # Execute
        result = await worker.execute(state, config)

        # Convert to dict if needed
        if hasattr(result, "dict"):
            return result.dict()
        elif hasattr(result, "model_dump"):
            return result.model_dump()
        return result

    except Exception as e:
        logger.error(f"Worker {config.worker_id} failed: {e}")
        return {
            "worker_id": config.worker_id,
            "worker_type": config.worker_type,
            "task": config.task,
            "success": False,
            "error": str(e),
        }


async def route_after_workers(state: EngineState) -> str:
    """Route decision after workers node.

    Returns:
        "critic" - Proceed to critic evaluation
        "error" - If all workers failed
    """
    worker_results = state.get("worker_results", [])
    worker_errors = state.get("worker_errors", [])

    if not worker_results and worker_errors:
        return "error"

    return "critic"
