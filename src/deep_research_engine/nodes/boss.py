"""Boss Node - LangGraph node for planning and task decomposition.

This node analyzes the input query and creates a research plan,
decomposing it into tasks for worker agents.
"""

import json
from typing import Any

from ..prompts import PromptRegistry, get_prompt
from ..state import EngineState
from ..utils.logging import get_logger, get_loguru_logger

logger = get_logger("nodes.boss")
loguru_logger = get_loguru_logger("boss")


async def boss_node(state: EngineState) -> EngineState:
    """Boss node for LangGraph workflow.

    This node:
    1. Analyzes the input query
    2. Creates a research plan
    3. Decomposes into worker tasks
    4. Updates state with plan

    Args:
        state: Current engine state

    Returns:
        Updated state with plan and worker tasks
    """
    logger.info("Starting boss node", phase="boss")

    try:
        # Get query from input
        input_data = state.get("input_data", {})
        query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)

        # Get boss config
        boss_config = state.get("context", {}).get("boss", {})
        strategy = boss_config.get("strategy", "complexity_based")
        min_workers = boss_config.get("workers", {}).get("min", 1)
        max_workers = boss_config.get("workers", {}).get("max", 3)

        # Create plan using LLM
        plan = await _create_research_plan(
            query=query,
            strategy=strategy,
            min_workers=min_workers,
            max_workers=max_workers,
        )

        # Update state
        state["plan"] = plan.get("tasks", [])
        state["plan_metadata"] = {
            "strategy": strategy,
            "complexity": plan.get("complexity", "medium"),
            "worker_count": len(plan.get("tasks", [])),
        }
        state["current_phase"] = "workers"
        state["phase_history"].append("workers")

        logger.info(
            "Planning complete",
            task_count=len(plan.get("tasks", [])),
            complexity=plan.get("complexity"),
        )

        return state

    except Exception as e:
        logger.error(f"Boss node error: {e}", exc_info=True)
        state["errors"].append(
            {
                "phase": "boss",
                "error": str(e),
                "type": type(e).__name__,
            }
        )
        return state


async def _create_research_plan(
    query: str,
    strategy: str,
    min_workers: int,
    max_workers: int,
) -> dict[str, Any]:
    """Create research plan using LLM.

    Args:
        query: Research query
        strategy: Planning strategy
        min_workers: Minimum number of worker tasks
        max_workers: Maximum number of worker tasks

    Returns:
        Plan dict with tasks and metadata
    """
    from deepagents import create_deep_agent

    # Get prompt from LangFuse
    prompt = await get_prompt(PromptRegistry.RESEARCH_PLANNER)

    # Compile prompt with variables
    system_prompt = prompt.compile(
        strategy=strategy,
        min_workers=min_workers,
        max_workers=max_workers,
        query=query,
    )

    # Get model config from prompt
    model_config = prompt.get_model_config()

    agent = create_deep_agent(model=model_config["model"], tools=[], system_prompt=system_prompt)

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Create a research plan."}]}
        )

        output = str(result.get("messages", [])[-1].content) if result else "{}"

        # Parse JSON output
        try:
            # Try to extract JSON from the output
            import re

            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            plan = json.loads(json_match.group()) if json_match else json.loads(output)
        except json.JSONDecodeError:
            # Fallback: create simple plan
            plan = {
                "tasks": [f"Research aspect of: {query}"],
                "complexity": "simple",
                "reasoning": "Fallback plan due to parsing error",
            }

        return plan

    except Exception as e:
        logger.error(f"Plan creation failed: {e}")
        # Return fallback plan
        return {
            "tasks": [f"Research: {query}"],
            "complexity": "simple",
            "reasoning": "Fallback plan due to error",
        }


async def route_after_boss(state: EngineState) -> str:
    """Route decision after boss node.

    Returns:
        "workers" - Proceed to worker execution
        "error" - If planning failed
    """
    if state.get("fatal_error") or not state.get("plan"):
        return "error"
    return "workers"
