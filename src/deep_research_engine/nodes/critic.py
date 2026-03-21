"""Critic Node - Evaluates the draft and decides if it passes.
"""

import time
from typing import Any
from pydantic import BaseModel, Field

from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config
from ..prompts import get_prompt, PromptRegistry
from shared.chat_llm import get_chat_llm

logger = get_loguru_logger("nodes.critic")


class CriticEvaluation(BaseModel):
    """Structured output for critic evaluation."""
    criteria_scores: dict[str, float] = Field(..., description="Scores for each criterion (0.0 to 1.0)")
    overall_score: float = Field(..., description="Overall quality score (0.0 to 1.0)")
    issues: list[str] = Field(default_factory=list, description="Specific problems found")
    suggestions: list[str] = Field(default_factory=list, description="Actionable improvements for the next iteration")


@node_context("critic")
async def critic_node(state: EngineState) -> dict[str, Any]:
    """Critic node for LangGraph workflow."""
    logger.info("Starting critic node")

    # 1. Get configuration
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "critic")
    critic_enabled = phase_config.get("enabled", True)

    if not critic_enabled:
        logger.info("Critic node disabled. Skipping.")
        return {"__goto__": "neo4j_builder"}

    # 2. Get the draft to evaluate
    # The draft is usually the output of the king_drafter in worker_results
    worker_results = state.get("worker_results", [])
    if not worker_results:
        logger.warning("No worker results found to evaluate.")
        return {"__goto__": "neo4j_builder", "critic_passed": True}

    # Latest result (assuming it's the draft)
    # Filter for results that actually have output
    valid_results = [r for r in worker_results if r.get("output")]
    if not valid_results:
        logger.warning("No worker results with output found.")
        return {"__goto__": "neo4j_builder", "critic_passed": True}

    draft_content = valid_results[-1].get("output", "")

    # 3. Prepare evaluation criteria
    criteria = phase_config.get("criteria", [])
    if not criteria:
        # Default criteria
        criteria_text = (
            "- Completeness: Does the draft cover all research objectives?\n"
            "- Accuracy: Is the information factual and well-supported?\n"
            "- Clarity: Is the draft well-structured and easy to understand?"
        )
    else:
        criteria_lines = []
        for c in criteria:
            if isinstance(c, dict):
                name = c.get("name", "Criterion")
                desc = c.get("description") or c.get("name") or "Quality check"
                criteria_lines.append(f"- {name}: {desc}")
            elif isinstance(c, str):
                criteria_lines.append(f"- {c}")
        criteria_text = "\n".join(criteria_lines)

    try:
        # 4. Fetch prompt and initialize LLM
        prompt_wrapper = await get_prompt(PromptRegistry.RESEARCH_EVALUATOR)

        # Determine model - use config if available, otherwise fallback
        model_name = phase_config.get("model")
        if not model_name:
            # Try to get from global llm config
            llm_config = config_data.get("llm", {})
            model_name = llm_config.get("models", {}).get("main", "openai:gpt-4o")

        llm = await get_chat_llm(model=model_name)
        # Use structured output with GBNF grammar via LiteLLM/llama.cpp if local
        structured_llm = llm.with_structured_output(CriticEvaluation)

        # 5. Compile prompt and invoke LLM
        logger.info(f"Evaluating research draft using {model_name}...")
        start_time = time.time()

        # The prompt template uses {{ criteria_text }} and {{ results_text }}
        compiled_prompt = prompt_wrapper.compile(
            criteria_text=criteria_text,
            results_text=draft_content
        )

        evaluation = await structured_llm.ainvoke(compiled_prompt)
        eval_time = time.time() - start_time

        logger.info(f"Evaluation complete in {eval_time:.2f}s. Score: {evaluation.overall_score}")

        # 6. Determine pass/fail
        threshold = phase_config.get("pass_threshold", 0.7)
        passed = evaluation.overall_score >= threshold

        # Count iterations
        iteration = state.get("iteration_count", 0) + 1
        max_retries = phase_config.get("max_retries", 3)

        # Prepare critique result matching CriticResult TypedDict
        critique = {
            "passed": passed,
            "score": evaluation.overall_score,
            "criteria_scores": evaluation.criteria_scores,
            "issues": evaluation.issues,
            "suggestions": evaluation.suggestions,
            "recommendation": "pass" if passed else ("revise" if iteration <= max_retries else "terminate"),
            "feedback_to": "drafter" if not passed else "neo4j_builder",
            "evaluation_time": eval_time,
            "model_info": {"model": model_name}
        }

        # Update history
        critique_history = state.get("critique_history", []) + [critique]

        # Determine next step
        if passed or iteration > max_retries:
            if not passed:
                logger.info(f"Critic failed but max retries ({iteration}/{max_retries}) reached. Continuing to neo4j_builder.")
            else:
                logger.info(f"Critic passed with score {evaluation.overall_score}. Proceeding to neo4j_builder.")

            return {
                "critique": critique,
                "critique_history": critique_history,
                "critic_passed": True,
                "iteration_count": iteration,
                "__goto__": "neo4j_builder"
            }
        else:
            logger.info(f"Critic failed ({iteration}/{max_retries}). Retrying drafter with feedback.")
            return {
                "critique": critique,
                "critique_history": critique_history,
                "critic_passed": False,
                "iteration_count": iteration,
                "__goto__": "drafter"
            }

    except Exception as e:
        logger.error(f"Error during critic evaluation: {e}", exc_info=True)
        # Fallback: log error and continue to avoid pipeline stall
        return {
            "critic_passed": True,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "__goto__": "neo4j_builder",
            "errors": state.get("errors", []) + [{"node": "critic", "error": str(e)}]
        }
