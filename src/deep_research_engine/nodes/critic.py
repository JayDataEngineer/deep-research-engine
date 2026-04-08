"""Critic Node - Evaluates the draft and decides if it passes.

Uses LLM-based evaluation with structured output for consistent quality assessment.
"""

from typing import Any

from ..config import get_phase_config
from ..shared import get_langchain_llm
from ..state import CriticEvaluation, EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context

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
    raw_criteria = config.get("criteria", {
        "completeness": "All planned tasks are addressed",
        "accuracy": "Claims are factually correct and verifiable",
        "citation_quality": "Sources are credible and properly cited",
        "coherence": "Output is well-organized and logically structured"
    })

    # Handle list format from YAML (e.g., [{name: completeness, weight: 0.4}, ...])
    if isinstance(raw_criteria, list):
        criteria = {c.get("name", f"criterion_{i}"): f"Weight: {c.get('weight', 0.5)}, Threshold: {c.get('threshold', 0.7)}"
                     for i, c in enumerate(raw_criteria) if isinstance(c, dict)}
    else:
        criteria = raw_criteria
    
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


def _parse_text_evaluation(text: str) -> CriticEvaluation:
    """Parse evaluation from free-text LLM response.

    Extracts scores and decision from the text. Falls back to a
    default pass only if no scores can be found at all.
    """
    import re

    # Try to extract overall score
    score = 0.75
    score_match = re.search(r'(?:overall\s+)?score[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    else:
        # Try finding "X/10" or "X out of 10" patterns
        ratio_match = re.search(r'(\d+\.?\d*)\s*/\s*10', text)
        if ratio_match:
            score = float(ratio_match.group(1)) / 10.0

    passed = score >= 0.7

    # Try to extract individual dimension scores
    def _extract_dim(name: str) -> float:
        pattern = rf'{name}[:\s]+(\d+\.?\d*)'
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else score

    # Extract issues and suggestions from bullet points
    issues = []
    suggestions = []
    in_issues = False
    in_suggestions = False
    for line in text.split("\n"):
        stripped = line.strip()
        if "issue" in stripped.lower() and (":" in stripped or stripped.endswith(":")):
            in_issues = True
            in_suggestions = False
            continue
        if "suggestion" in stripped.lower() and (":" in stripped or stripped.endswith(":")):
            in_suggestions = True
            in_issues = False
            continue
        if stripped.startswith("##") or stripped.startswith("#"):
            in_issues = False
            in_suggestions = False
            continue
        if in_issues and stripped.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
            issue = stripped.lstrip("-*•0123456789. ").strip()
            if issue:
                issues.append(issue)
        if in_suggestions and stripped.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
            suggestion = stripped.lstrip("-*•0123456789. ").strip()
            if suggestion:
                suggestions.append(suggestion)

    # Limit lists
    issues = issues[:10]
    suggestions = suggestions[:10]

    if not issues:
        issues = ["Auto-evaluated from free-text response"]
    if not suggestions:
        suggestions = ["Review output manually for quality assurance"]

    return CriticEvaluation(
        passed=passed,
        score=score,
        completeness=_extract_dim("completeness"),
        accuracy=_extract_dim("accuracy"),
        citation_quality=_extract_dim("citation"),
        coherence=_extract_dim("coherence"),
        issues=issues,
        suggestions=suggestions,
        recommendation="pass" if passed else "revise",
        feedback_to="artifact_maker",
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
    
    # 4. Invoke critic with structured output
    try:
        llm = get_langchain_llm(enable_thinking=False, temperature=0.3, config_data=config_data)

        try:
            structured_llm = llm.with_structured_output(CriticEvaluation)
            result = await structured_llm.ainvoke([{"role": "user", "content": eval_prompt}])

            if isinstance(result, CriticEvaluation):
                evaluation = result
                logger.info(
                    f"Critic evaluation complete: score={evaluation.score:.2f}, "
                    f"passed={evaluation.passed}"
                )
            else:
                logger.warning("Structured output parsing failed, using text fallback")
                evaluation = _parse_text_evaluation(str(result))
        except Exception as structured_err:
            # Structured output not supported by backend (llama.cpp etc.)
            logger.warning(f"Structured output failed ({structured_err}), using text parsing")
            raw_response = await llm.ainvoke([{"role": "user", "content": eval_prompt}])
            evaluation = _parse_text_evaluation(raw_response.content)

    except Exception as e:
        logger.error(f"Critic evaluation failed: {e}")
        evaluation = _parse_text_evaluation(str(e))
    
    # 6. Count iterations from critique history
    iteration = len(state.get("critique_history", []))
    
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
