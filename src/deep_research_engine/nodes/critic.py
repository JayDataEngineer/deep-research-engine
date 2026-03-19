"""Critic Node - LangGraph node for quality evaluation.

This node evaluates the quality of worker outputs and decides
whether to pass to synthesis or retry workers.
"""

import json
from typing import Any

from ..prompts import PromptRegistry, get_prompt
from ..state import EngineState
from ..utils.logging import get_logger, get_loguru_logger

logger = get_logger("nodes.critic")
loguru_logger = get_loguru_logger("critic")


async def critic_node(state: EngineState) -> EngineState:
    """Critic node for LangGraph workflow.

    This node:
    1. Evaluates worker outputs against criteria
    2. Scores quality
    3. Decides pass/fail
    4. Updates state with critique

    Args:
        state: Current engine state

    Returns:
        Updated state with critique and decision
    """
    logger.info("Starting critic node", phase="critic")

    try:
        # Get worker results
        worker_results = state.get("worker_results", [])
        if not worker_results:
            logger.warning("No worker results to evaluate")
            state["critique"] = {
                "passed": False,
                "score": 0.0,
                "issues": ["No worker results to evaluate"],
                "recommendation": "terminate",
            }
            return state

        # Get critic config
        critic_config = state.get("context", {}).get("critic", {})
        criteria = critic_config.get(
            "criteria",
            [
                {"name": "completeness", "weight": 0.4, "threshold": 0.7},
                {"name": "accuracy", "weight": 0.4, "threshold": 0.8},
            ],
        )
        pass_threshold = critic_config.get("pass_threshold", 0.7)

        # Evaluate
        critique_result = await _evaluate_outputs(
            worker_results=worker_results,
            criteria=criteria,
        )

        # Decision
        passed = critique_result.get("score", 0.0) >= pass_threshold
        max_iterations = state.get("max_iterations", 3)
        current_iteration = state.get("iteration_count", 0)

        if passed:
            decision = "pass"
        elif current_iteration >= max_iterations:
            decision = "terminate"  # Max retries reached
        else:
            decision = "revise"

        # Update state
        state["critique"] = {
            "passed": passed,
            "score": critique_result.get("score", 0.0),
            "criteria_scores": critique_result.get("criteria_scores", {}),
            "issues": critique_result.get("issues", []),
            "suggestions": critique_result.get("suggestions", []),
            "recommendation": decision,
        }
        state["critique_history"] = [*state.get("critique_history", []), state["critique"]]
        state["critic_passed"] = passed

        if passed:
            state["current_phase"] = "synthesis"
            state["phase_history"].append("synthesis")
        else:
            state["iteration_count"] = current_iteration + 1
            state["current_phase"] = "workers"
            state["phase_history"].append("workers")

        logger.info(
            "Critic evaluation complete",
            score=critique_result.get("score"),
            decision=decision,
        )

        return state

    except Exception as e:
        logger.error(f"Critic node error: {e}", exc_info=True)
        state["errors"].append(
            {
                "phase": "critic",
                "error": str(e),
                "type": type(e).__name__,
            }
        )
        return state


async def _evaluate_outputs(
    worker_results: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate worker outputs against criteria.

    Args:
        worker_results: List of worker result dicts
        criteria: List of criteria with weights and thresholds

    Returns:
        Critique result dict
    """
    from deepagents import create_deep_agent

    # Build evaluation prompt components
    criteria_text = "\n".join(
        f"- {c['name']} (weight: {c['weight']}, threshold: {c['threshold']})" for c in criteria
    )

    results_text = "\n\n".join(
        f"Task: {r.get('task', 'Unknown')}\n"
        f"Success: {r.get('success', False)}\n"
        f"Output: {r.get('output', 'No output')[:500]}\n"
        f"Confidence: {r.get('confidence', 0.0)}"
        for r in worker_results
    )

    # Get prompt from LangFuse
    prompt = await get_prompt(PromptRegistry.RESEARCH_EVALUATOR)

    # Compile prompt with variables
    system_prompt = prompt.compile(
        criteria_text=criteria_text,
        results_text=results_text,
    )

    # Get model config from prompt
    model_config = prompt.get_model_config()

    agent = create_deep_agent(model=model_config["model"], tools=[], system_prompt=system_prompt)

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Evaluate the research outputs."}]}
        )

        output = str(result.get("messages", [])[-1].content) if result else "{}"

        # Parse JSON
        try:
            import re

            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            critique = json.loads(json_match.group()) if json_match else json.loads(output)
        except json.JSONDecodeError:
            # Fallback
            critique = {
                "criteria_scores": {c["name"]: 0.5 for c in criteria},
                "overall_score": 0.5,
                "issues": ["Could not parse evaluation"],
                "suggestions": ["Retry with clearer instructions"],
            }

        return critique

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "criteria_scores": {c["name"]: 0.5 for c in criteria},
            "overall_score": 0.5,
            "issues": [f"Evaluation error: {e!s}"],
            "suggestions": ["Retry workers"],
        }


async def route_after_critic(state: EngineState) -> str:
    """Route decision after critic node.

    Returns:
        "synthesis" - Pass to synthesis
        "workers" - Retry workers
        "error" - Terminate
    """
    critique = state.get("critique", {})
    recommendation = critique.get("recommendation", "pass")

    if recommendation == "pass":
        return "synthesis"
    elif recommendation == "revise":
        return "workers"
    else:  # terminate
        return "error"
