"""Critic Node - uses universal draft_extract pattern."""

from typing import Literal, Optional, Any, List, Dict
from pydantic import BaseModel, Field

from ..common import (
    GlobalState,
    ResearchRuntimeConfig,
    get_run_logger_from_config,
    get_yaml_config_from_context,
    create_llm_from_config,
    get_llms_for_node,
    require_run_logger,
    get_prompt_from_config,
)
from ...rating import (
    normalize_confidence,
    Rating,
    RatingDecision,
)
from ....logging_utils import log_component

from ...decorators import research_node, Phase
from .base import draft_extract

# Import streaming helpers for SSE events
from ...helpers.streaming import (
    get_writer,
    emit_phase_change,
    emit_quality_metrics,
)

from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


class CriticEvaluation(BaseModel):
    """Critic evaluation output schema."""
    sufficient: bool = Field(
        description="Is the research sufficient for synthesis?"
    )
    confidence: int = Field(
        ge=0, le=100,
        description="Confidence score: 0=none, 50=moderate, 100=complete. Use 0-100 scale."
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Research gaps that need to be filled (if not sufficient)"
    )
    feedback: str = Field(
        default="",
        description="Feedback for next iteration (if not sufficient)"
    )


@research_node(Phase.CRITIC)
async def unified_critic_node(
    state: GlobalState,
    writer: Optional[Any] = None,
    *,
    runtime: Optional[Runtime[ResearchRuntimeConfig]] = None,
    store: Optional[Any] = None,
    config: RunnableConfig | None = None,
) -> Command[Literal["boss", "formatter"]]:
    """Critic using universal draft_extract pattern.

    Evaluates:
    - Is research sufficient for synthesis?
    - What gaps remain?
    - Should we iterate or proceed?
    """
    run_logger = get_run_logger_from_config(config)
    require_run_logger(run_logger, "unified critic node")

    yaml_config = get_yaml_config_from_context()
    query = state.get("query")
    session_id = state.get("session_id")
    iteration_count = state.get("boss_iteration_count", 0)
    final_response = state.get("final_response", "")

    # Get stream writer for SSE events (from positional arg or config)
    stream_writer = get_writer(writer, config)

    # Emit phase change event at start
    emit_phase_change(stream_writer, "critic", "Quality Review", "Evaluating research quality...", "active")

    log_component(run_logger, "CRITIC", f"Evaluating iteration {iteration_count}")
    log_component(run_logger, "CRITIC", f"Final response length: {len(final_response)} chars")

    # Get max iterations from config FIRST - this is the circuit breaker
    spawner_config = yaml_config.get("spawner_config", {})
    max_iterations = spawner_config.get("max_iterations", 3)

    # Check if we've hit max iterations - ALWAYS check this first to prevent infinite loops
    # CRITICAL: Must return immediately to prevent any further iteration
    if iteration_count >= max_iterations:
        log_component(run_logger, "CRITIC", f"Max iterations ({max_iterations}) reached, proceeding to formatter")

        # Emit quality metrics for max iterations
        emit_quality_metrics(
            writer=stream_writer,
            quality=0.8,  # Default quality when forced to proceed
            completeness=0.8,
            passed=True,
            feedback="Max iterations reached, proceeding with current research"
        )
        emit_phase_change(stream_writer, "critic", "Quality Review", "Quality review complete", "complete")

        return Command(goto="formatter")

    # VALIDATION: Check for placeholder/garbage content (only if not at max iterations)
    placeholder_phrases = [
        "ERROR:",
        "I'll help you",
        "Synthesizing research",
        "I will search",
        "Let me find",
        "searching for",
        "query the",
        "No research findings available",
    ]
    is_placeholder = any(phrase.lower() in final_response.lower() for phrase in placeholder_phrases)
    is_too_short = len(final_response) < 200  # Real synthesis should be longer

    # Only reject and loop back if we haven't hit max iterations
    if is_placeholder and iteration_count < max_iterations:
        log_component(run_logger, "CRITIC", "REJECTED: Response contains placeholder text")
        return Command(
            update={
                "current_phase": "boss",
                "critic_feedback": "Synthesis contained placeholder text instead of actual research. Regenerate with actual findings.",
            },
            goto="boss"
        )

    if is_too_short and iteration_count < max_iterations:
        log_component(run_logger, "CRITIC", f"REJECTED: Response too short ({len(final_response)} chars)")
        return Command(
            update={
                "current_phase": "boss",
                "critic_feedback": f"Synthesis too short ({len(final_response)} chars). Need more comprehensive output.",
            },
            goto="boss"
        )

    # Get LLMs (main for reasoning, tool for binding)
    main_llm, tool_llm = await get_llms_for_node(yaml_config)

    # Check if we have existing data in context engine
    has_memories = False
    memory_count = 0
    if store and session_id:
        try:
            from src.extensions.content.deep_research.storage import namespace_prefix
            ns_prefix = namespace_prefix(session_id, "memories")
            existing_memories = await store.asearch(ns_prefix)
            memory_count = len(list(existing_memories)) if existing_memories else 0
            has_memories = memory_count > 0
        except Exception:
            pass

    # Build context - Critic uses TOOLS to check actual data
    synthesis_preview = final_response[:1000] if final_response else "No synthesis"

    # Use Jinja template from config.yaml
    try:
        system_prompt = get_prompt_from_config(
            yaml_config,
            "critic.system",
            tool_info="Use query_postgres and get_artifact tools to check Context Engine for research findings.",
        )
    except RuntimeError:
        # Fallback if template not found - use minimal hardcoded prompt
        log_component(run_logger, "CRITIC", "WARNING: critic.system template not found, using fallback")
        system_prompt = f"""You are the Critic. Evaluate research completeness.

Session: {session_id}
Iteration: {iteration_count}/{max_iterations}

YOUR JOB: Use your tools to CHECK what research findings actually exist in Context Engine.

SYNTHESIS OUTPUT PREVIEW:
{synthesis_preview}

Evaluate:
1. Use tools to CHECK actual findings in Context Engine
2. Is the synthesis based on REAL data or just placeholders?
3. What gaps remain?

CRITICAL: When setting confidence, use a NUMBER from 0 to 100:
- 0 = no confidence
- 50 = moderate confidence
- 100 = complete confidence
NEVER use values outside 0-100 range."""

    # Universal draft_extract pattern
    # Critic gets tools when there might be data to check (past first iteration)
    critic_should_have_tools = iteration_count > 0 or has_memories
    result = await draft_extract(
        query=f"Evaluate research completeness for: {query}",
        system_prompt=system_prompt,
        llm=main_llm,
        output_schema=CriticEvaluation,
        extractor_llm=tool_llm,
        has_data=critic_should_have_tools,  # Critic gets tools to check findings
    )

    evaluation = result["extracted"]

    # Normalize confidence to 0-100 scale using centralized helper
    confidence = normalize_confidence(evaluation.confidence)
    has_no_gaps = len(evaluation.gaps) == 0

    log_component(run_logger, "CRITIC",
        f"Sufficient: {evaluation.sufficient}, "
        f"Confidence: {confidence} ({Rating.get_confidence_label(confidence)}), "
        f"Gaps: {len(evaluation.gaps)}"
    )

    # REJECT if LLM explicitly says not sufficient OR confidence is low
    if not evaluation.sufficient or confidence < 70:
        log_component(run_logger, "CRITIC",
            f"REJECTED by LLM evaluation "
            f"(sufficient={evaluation.sufficient}, confidence={confidence})"
        )

        # Emit quality metrics for rejection
        emit_quality_metrics(
            writer=stream_writer,
            quality=confidence / 100.0,  # Convert to 0-1 scale
            completeness=0.5 if has_no_gaps else 0.3,
            passed=False,
            feedback=evaluation.feedback or f"Low confidence ({confidence}) or insufficient research. Gaps: {evaluation.gaps[:3]}",
            issues=evaluation.gaps,
            missing_info=evaluation.gaps
        )
        emit_phase_change(stream_writer, "critic", "Quality Review", "Quality review complete - iteration needed", "complete")

        return Command(
            update={
                "current_phase": "boss",
                "critic_feedback": evaluation.feedback or f"Low confidence ({confidence}) or insufficient research. Gaps: {evaluation.gaps[:3]}",
            },
            goto="boss"
        )

    # ACCEPT: High confidence AND sufficient
    log_component(run_logger, "CRITIC",
        f"ACCEPTED - Proceeding to formatter "
        f"(sufficient={evaluation.sufficient}, confidence={confidence})"
    )

    # Emit quality metrics for acceptance
    emit_quality_metrics(
        writer=stream_writer,
        quality=confidence / 100.0,  # Convert to 0-1 scale
        completeness=1.0 if has_no_gaps else 0.8,
        passed=True,
        feedback=evaluation.feedback or "Research quality approved"
    )
    emit_phase_change(stream_writer, "critic", "Quality Review", "Quality review complete", "complete")

    return Command(
        update={"current_phase": "end"},
        goto="formatter"
    )
