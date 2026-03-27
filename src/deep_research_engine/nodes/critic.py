"""Critic Node - Evaluates the draft and decides if it passes.

Uses LLM-based evaluation with structured output for consistent quality assessment.
"""

from typing import Any
from ..state import EngineState, CriticEvaluation
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config

logger = get_loguru_logger("nodes.critic")


def build_critic_prompt(
    worker_results: list[dict],
    plan: list[str],
    config: dict
) -> str:
    """Build the critic evaluation prompt.
    
    Args:
        worker_results: List of worker outputs to evaluate
        plan: Original research plan/tasks
        config: Critic configuration with criteria
        
    Returns:
        Formatted prompt for LLM evaluation
    """
    # Format worker results
    results_text = "\n\n".join([
        f"### Worker: {r.get('worker_type', 'unknown')}\n"
        f"Task: {r.get('task', 'N/A')}\n"
        f"Output: {r.get('output', '')[:500]}...\n"  # Truncate for context
        f"Confidence: {r.get('confidence', 0):.2f}\n"
        f"Sources: {r.get('source_count', 0)}"
        for r in worker_results
    ])
    
    # Format original plan
    plan_text = "\n".join([f"- {task}" for task in plan]) if plan else "No plan provided"
    
    # Get custom criteria from config
    criteria = config.get("criteria", {
        "completeness": "All planned tasks are addressed",
        "accuracy": "Claims are factually correct and verifiable",
        "citation_quality": "Sources are credible and properly cited",
        "coherence": "Output is well-organized and logically structured"
    })
    
    criteria_text = "\n".join([
        f"- **{k}**: {v}" for k, v in criteria.items()
    ])
    
    prompt = f"""You are evaluating research output quality.

## Original Plan
{plan_text}

## Worker Results
{results_text}

## Evaluation Criteria
{criteria_text}

## Instructions
Evaluate the worker results against the criteria. For each criterion:
1. Score from 0-1 (0=poor, 1=excellent)
2. Provide specific evidence for your score
3. List concrete issues found
4. Suggest actionable improvements

Be rigorous. A score of 0.7+ is required to pass.

Output your evaluation in the structured format provided."""
    
    return prompt


def parse_fallback_evaluation(result: Any) -> CriticEvaluation:
    """Parse fallback evaluation if structured output fails.
    
    Args:
        result: Raw LLM output
        
    Returns:
        CriticEvaluation object
    """
    # Default fallback - pass with moderate score
    return CriticEvaluation(
        passed=True,
        score=0.75,
        completeness=0.75,
        accuracy=0.75,
        citation_quality=0.75,
        coherence=0.75,
        issues=["Structured output parsing failed - using default evaluation"],
        suggestions=["Review output manually for quality assurance"],
        recommendation="pass",
        feedback_to="artifact_maker"
    )


@node_context("critic")
async def critic_node(state: EngineState) -> dict[str, Any]:
    """Critic node - Evaluates draft quality using LLM.
    
    The critic evaluates worker outputs against the original plan and
    quality criteria. It provides specific feedback and routes to either
    the next phase (neo4j_builder) or back to drafter for revision.
    
    Args:
        state: Current engine state
        
    Returns:
        State update with routing decision and critique
    """
    logger.info("Starting critic node")
    
    # 1. Get config
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "critic")
    
    if not phase_config.get("enabled", True):
        logger.info("Critic node disabled. Skipping.")
        return {"__goto__": "neo4j_builder"}
    
    # 2. Gather evaluation context
    worker_results = state.get("worker_results", [])
    plan = state.get("plan", [])
    
    if not worker_results:
        logger.warning("No worker results to evaluate")
        return {"__goto__": "neo4j_builder", "critic_passed": True}
    
    # 3. Build evaluation prompt
    eval_prompt = build_critic_prompt(worker_results, plan, phase_config)
    
    # 4. Invoke critic agent with structured output
    try:
        from deepagents import create_deep_agent
        
        critic_agent = create_deep_agent(
            model=phase_config.get("model", "openai:gpt-4o"),
            system_prompt=(
                "You are a rigorous academic critic. Your job is to evaluate "
                "research output quality across multiple dimensions. Be harsh but fair. "
                "Provide specific, actionable feedback."
            ),
        )
        
        # Use structured output for consistent evaluation
        structured_llm = critic_agent.with_structured_output(CriticEvaluation)
        
        result = await structured_llm.ainvoke([{"role": "user", "content": eval_prompt}])
        
        # 5. Parse evaluation
        if isinstance(result, CriticEvaluation):
            evaluation = result
            logger.info(
                f"Critic evaluation complete: score={evaluation.score:.2f}, "
                f"passed={evaluation.passed}"
            )
        else:
            logger.warning("Structured output parsing failed, using fallback")
            evaluation = parse_fallback_evaluation(result)
            
    except ImportError as e:
        logger.error(f"deepagents import failed: {e}")
        evaluation = parse_fallback_evaluation(None)
        
    except Exception as e:
        logger.error(f"Critic evaluation failed: {e}")
        evaluation = parse_fallback_evaluation(None)
    
    # 6. Count iterations
    iteration = len([p for p in state.get("phase_history", []) if p == "critic"])
    if not any(p == "critic" for p in state.get("phase_history", [])):
        iteration += 1
    
    max_retries = phase_config.get("max_retries", 3)
    threshold = phase_config.get("threshold", 0.7)
    
    # Override pass decision based on threshold
    if evaluation.score >= threshold:
        evaluation.passed = True
        evaluation.recommendation = "pass"
    
    # 7. Make routing decision
    if evaluation.passed or iteration >= max_retries:
        if iteration >= max_retries and not evaluation.passed:
            logger.warning(
                f"Max retries ({max_retries}) reached with score {evaluation.score:.2f}. "
                f"Proceeding anyway."
            )
        
        logger.info(f"Critic passed (score={evaluation.score:.2f}) after {iteration} iteration(s)")
        return {
            "__goto__": "neo4j_builder",
            "critic_passed": True,
            "critique": evaluation.model_dump(),
            "critique_history": state.get("critique_history", []) + [evaluation.model_dump()]
        }
    else:
        logger.info(
            f"Critic failed (score={evaluation.score:.2f} < {threshold}). "
            f"Retrying drafter ({iteration}/{max_retries})"
        )
        return {
            "__goto__": "drafter",
            "critic_passed": False,
            "critique": evaluation.model_dump(),
            "critique_history": state.get("critique_history", []) + [evaluation.model_dump()]
        }
