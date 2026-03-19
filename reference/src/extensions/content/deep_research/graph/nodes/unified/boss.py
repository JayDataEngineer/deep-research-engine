"""Boss Node - uses universal draft_extract pattern."""

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
from ....logging_utils import log_component

from ...decorators import research_node, Phase
from .base import draft_extract

# Import streaming helpers for SSE events
from ...helpers.streaming import (
    get_writer,
    emit_phase_change,
    emit_boss_summary,
    emit_worker_task_start,
)

from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


class BossDecision(BaseModel):
    """Boss decision output schema."""
    workers_to_spawn: List[str] = Field(
        description="List of worker types to spawn (e.g., ['worker'], ['source_explorer'], ['worker', 'analyst_filter']). "
                    "Boss ALWAYS spawns workers - Critic decides when to synthesize. "
                    "Must use predefined_specialists from config.yaml or 'worker' (default)."
    )
    reasoning: str = Field(
        default="",
        description="Why these workers were chosen"
    )


@research_node(Phase.BOSS)
async def unified_boss_node(
    state: GlobalState,
    writer: Optional[Any] = None,
    *,
    runtime: Optional[Runtime[ResearchRuntimeConfig]] = None,
    store: Optional[Any] = None,
    config: RunnableConfig | None = None,
) -> Command[Literal["lab_architect"]]:
    """Boss using universal draft_extract pattern.

    Boss ALWAYS spawns workers (minimum worker if LLM returns empty list).
    Critic decides when research is sufficient and routes to synthesis.

    Uses predefined_specialists from config.yaml to validate worker types.
    """
    run_logger = get_run_logger_from_config(config)
    require_run_logger(run_logger, "unified boss node")

    yaml_config = get_yaml_config_from_context()
    query = state.get("query") or state.get("raw_query")  # Support both keys
    session_id = state.get("session_id")

    # Get stream writer for SSE events (from positional arg or config)
    stream_writer = get_writer(writer, config)

    # Emit phase change event at start
    emit_phase_change(stream_writer, "boss", "Planning", "Planning research approach...", "active")

    log_component(run_logger, "BOSS", f"Query: {query[:100]}...")

    # Get LLMs (main for reasoning, tool for binding)
    main_llm, tool_llm = await get_llms_for_node(yaml_config)

    # Get any existing research findings
    architect_findings = state.get("architect_findings")
    critic_feedback = state.get("critic_feedback")
    iteration_count = state.get("boss_iteration_count", 0)

    # Build context - Boss will use TOOLS to check actual data
    # Get available specialist types from config
    predefined_specialists = yaml_config.get("predefined_specialists", {})
    available_workers = list(predefined_specialists.keys()) + ["worker"]  # worker is default

    context = f"""Session: {session_id}
Iteration: {iteration_count}
Available workers: {available_workers}

YOUR JOB: Use your tools to check what data exists, then DECIDE WHICH WORKERS TO SPAWN.

WORKER TYPES:
- worker: Default worker for initial research and gathering information. Can additionally be used as a tool to gather information for the purpose of refining research directions and meta understanding of the task the user has provided.
- {', '.join([w for w in available_workers if w != 'worker'])}

OUTPUT: A list of worker types to spawn. NOTE: Boss ALWAYS spawns workers - Critic decides when to synthesize."""

    if critic_feedback:
        context += f"\n\nPrevious critic feedback: {critic_feedback[:500]}"

    # Use Jinja template from config.yaml
    try:
        system_prompt = get_prompt_from_config(
            yaml_config,
            "boss.spawn",
            original_query=query,
            iteration=iteration_count,
        )
    except RuntimeError:
        # Fallback if template not found - use minimal hardcoded prompt
        log_component(run_logger, "BOSS", "WARNING: boss.spawn template not found, using fallback")
        system_prompt = f"""You are the Boss orchestrator. Your job is to decide which workers to spawn for research.

ORIGINAL QUERY: {query}

ITERATION: {iteration_count}

YOUR TASK: Decide which worker types to spawn.

WORKER TYPES:
- "worker": Default research worker for gathering information

SPAWN RULES:
1. ALWAYS spawn at least one worker (default: ["worker"])
2. For simple queries: spawn 1 worker
3. For complex queries: spawn 2-3 workers for parallel investigation

CRITICAL: Boss ALWAYS spawns workers. The Critic decides when research is sufficient.
You do NOT decide "deep" vs "shallow" - you just decide HOW MANY workers and WHAT TYPE.

OUTPUT: A list of worker types to spawn (e.g., ["worker"] or ["worker", "worker"])."""

    # Universal draft_extract pattern
    # Boss gets tools when there might be data to check (past first iteration or findings exist)
    boss_should_have_tools = iteration_count > 0 or architect_findings is not None
    result = await draft_extract(
        query=f"Decide next action for: {query}",
        system_prompt=system_prompt,
        llm=main_llm,
        output_schema=BossDecision,
        extractor_llm=tool_llm,
        has_data=boss_should_have_tools,  # Boss gets tools to check data on subsequent runs
    )

    decision = result["extracted"]
    workers_to_spawn_types = decision.workers_to_spawn
    log_component(run_logger, "BOSS", f"Workers to spawn: {workers_to_spawn_types}, Reasoning: {decision.reasoning[:100] if decision.reasoning else 'N/A'}")

    # Boss ALWAYS spawns workers - Critic decides when to synthesize
    # If Boss returned empty list, default to worker
    if not workers_to_spawn_types:
        log_component(run_logger, "BOSS", "No workers specified, defaulting to worker")
        workers_to_spawn_types = ["worker"]

    # VALIDATE: Only spawn workers that are actually defined in config
    valid_workers = set(available_workers)
    validated_workers = []
    for worker_type in workers_to_spawn_types:
        if worker_type in valid_workers:
            validated_workers.append(worker_type)
        else:
            log_component(run_logger, "BOSS", f"WARNING: Unknown worker type '{worker_type}', skipping")

    # If no valid workers after validation, default to worker
    if not validated_workers:
        log_component(run_logger, "BOSS", "No valid workers after validation, defaulting to worker")
        validated_workers = ["worker"]

    workers_to_spawn_types = validated_workers
    log_component(run_logger, "BOSS", f"Validated workers: {workers_to_spawn_types}")

    # Spawn all workers Boss decided on (worker is just another worker type)
    log_component(run_logger, "BOSS", f"Spawning workers: {workers_to_spawn_types}")

    worker_tasks = {}
    worker_types = {}

    for i, worker_type in enumerate(workers_to_spawn_types):
        worker_id = f"{worker_type}_{i}"
        worker_tasks[worker_id] = f"Process query for: {query}"
        worker_types[worker_id] = worker_type

    # Emit boss summary event
    emit_boss_summary(
        writer=stream_writer,
        decision="spawn_workers",
        num_workers=len(workers_to_spawn_types),
        worker_types=workers_to_spawn_types,
        reasoning=decision.reasoning
    )

    # Emit worker task start events for each worker
    for worker_id, task in worker_tasks.items():
        emit_worker_task_start(
            writer=stream_writer,
            worker_id=worker_id,
            task=task,
            description=f"Worker: {worker_types.get(worker_id, 'unknown')}"
        )

    # Emit phase change complete event
    emit_phase_change(stream_writer, "boss", "Planning", "Planning complete", "complete")

    return Command(
        update={
            "boss_decision": "spawn_workers",
            "workers_to_spawn": list(worker_tasks.keys()),
            "worker_tasks": worker_tasks,
            "worker_types": worker_types,
            "worker_dependencies": {},
            "research_brief": query,
            "boss_iteration_count": iteration_count + 1,
        },
        goto="lab_architect"
    )
