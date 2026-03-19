"""Common initialization and utilities for graph nodes.

This module provides shared imports and initialization functions that are
used across multiple graph nodes (worker, boss, lab, critic, synthesis).

By centralizing these common patterns, we reduce code duplication and make
it easier to maintain consistent behavior across nodes.

USAGE:
    Instead of importing all dependencies separately, nodes can import from here:

    from .common import (
        # State types
        GlobalState, ResearchRuntimeConfig,
        # Helper functions
        get_run_logger_from_config, get_yaml_config_from_context,
        get_runtime_config, create_llm_from_config, get_prompt_from_config,
        # Node helpers
        initialize_node, require_run_logger, create_node_llm,
        # Unified structured output pattern
        unified_structured_output_node,
        # Validation
        validate_node_state, validate_config,
        # Profile caching
        get_profile_with_cache,
        # Confidence normalization (DRY - single source of truth)
        normalize_confidence,
    )
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add backend to path for absolute imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=6)

from langgraph.runtime import Runtime

# Import commonly used state types
from src.extensions.content.deep_research.graph.state import (
    GlobalState,
    ResearchRuntimeConfig,
    ResearchBrief,
    CritiqueResult,
    WorkerResult,
    Citation,
    PromptAdjustment,
)

# CitationManifest and CitationClaim are in state_models (not exported from state)
from src.extensions.content.deep_research.graph.state_models import (
    CitationManifest,
    CitationClaim,
)

# Import commonly used helpers
from src.extensions.content.deep_research.graph.helpers.config import (
    get_run_logger_from_config,
    get_yaml_config_from_context,
    get_runtime_config,
    get_config_from_runnable_config,
    get_context_engine_from_context,
)

from src.extensions.content.deep_research.graph.helpers.llm import create_llm_from_config
from src.extensions.content.deep_research.graph.helpers.prompts import get_prompt_from_config

# Import message structure helpers (proper role attribution)
from src.extensions.content.deep_research.graph.helpers.message_structure import (
    build_research_messages,
    build_context_messages,
    build_system_only,
    validate_message_roles,
    get_message_summary,
)

# Import central tools registry
from src.shared.tools.registry import load_tools, get_tool

# Import tool constants (single source of truth)
from src.extensions.content.deep_research.graph.constants import TOOL_TO_SKILL_MAPPING


def build_available_skills(enabled_tools: list) -> list:
    """Build available_skills list from enabled_tools using centralized mapping.

    This replaces the hardcoded tool_to_skill dictionary that was duplicated
    across multiple nodes. The mapping is now centralized in common.py.

    Args:
        enabled_tools: List of enabled tool names from YAML config

    Returns:
        List of unique skill names for Boss to advertise
    """
    available_skills = []
    seen_skills = set()

    for tool in enabled_tools:
        if tool in TOOL_TO_SKILL_MAPPING:
            skill = TOOL_TO_SKILL_MAPPING[tool]
            if skill not in seen_skills:
                available_skills.append(skill)
                seen_skills.add(skill)

    return available_skills


# =============================================================================
# NODE INITIALIZATION
# =============================================================================

def initialize_node(
    runtime: Optional[Runtime[ResearchRuntimeConfig]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Optional[Any]]:
    """Initialize common node resources.

    This function provides a single point for initializing the resources
    that every graph node needs: yaml_config and run_logger.

    Args:
        runtime: Optional Runtime context (not used by most nodes, but kept for consistency)
        config: Optional RunnableConfig (for run_logger propagation through LangGraph)

    Returns:
        Tuple of (yaml_config, run_logger)

    Raises:
        RuntimeError: If run_logger is None (indicates configuration error)
    """
    yaml_config = get_yaml_config_from_context()
    run_logger = get_run_logger_from_config(config)

    return yaml_config, run_logger


def require_run_logger(run_logger: Optional[Any], node_name: str = "node") -> Any:
    """Ensure run_logger is available, raise error if not.

    Args:
        run_logger: The run_logger instance (may be None)
        node_name: Name of the node for error message

    Returns:
        The run_logger (if not None)

    Raises:
        RuntimeError: If run_logger is None
    """
    if run_logger is None:
        raise RuntimeError(
            f"CRITICAL: run_logger is None in {node_name}! "
            "This indicates a configuration error in the deep research graph. "
            "Failing immediately to prevent wasted computation."
        )
    return run_logger


# =============================================================================
# LLM CREATION HELPERS
# =============================================================================

async def create_node_llm(
    yaml_config: Dict[str, Any],
    role: str = "default",
) -> Any:
    """Create an LLM instance for a graph node.

    This is a convenience wrapper around create_llm_from_config that
    provides common defaults for graph nodes.

    Temperature is read from model_configs in yaml_config.

    Args:
        yaml_config: Full YAML configuration
        role: LLM role (default, worker, boss, researcher, critic, synthesis)

    Returns:
        LLM instance
    """
    return await create_llm_from_config(yaml_config, role=role)


async def get_llms_for_node(
    yaml_config: Dict[str, Any],
    main_role: str = "spawner",
    tool_role: str = "worker",
) -> tuple[Any, Any]:
    """Get both main and tool LLMs for a graph node.

    This is a DRY helper that eliminates the repeated pattern of:
        main_llm = await create_llm_from_config(yaml_config, role="spawner")
        tool_llm = await create_llm_from_config(yaml_config, role="worker")

    Args:
        yaml_config: Full YAML configuration
        main_role: Role for the main/reasoning LLM (default: "spawner")
        tool_role: Role for the tool/binding LLM (default: "worker")

    Returns:
        Tuple of (main_llm, tool_llm)
    """
    main_llm = await create_node_llm(yaml_config, role=main_role)
    tool_llm = await create_node_llm(yaml_config, role=tool_role)
    return main_llm, tool_llm


# =============================================================================
# STATE ACCESS HELPERS (DRY)
# =============================================================================

def get_common_state(state: GlobalState) -> Dict[str, Any]:
    """Extract commonly used state values in one call.

    This DRY helper eliminates repeated patterns like:
        query = state.get("query")
        session_id = state.get("session_id")
        iteration_count = state.get("boss_iteration_count", 0)

    Args:
        state: GlobalState dictionary

    Returns:
        Dict with: query, session_id, iteration_count, architect_findings,
        critic_feedback, final_response
    """
    return {
        "query": state.get("query", ""),
        "session_id": state.get("session_id", ""),
        "iteration_count": state.get("boss_iteration_count", 0),
        "architect_findings": state.get("architect_findings"),
        "critic_feedback": state.get("critic_feedback"),
        "final_response": state.get("final_response", ""),
    }


# =============================================================================
# PROFILE CACHING (TTL-based)
# =============================================================================

_profile_cache: Dict[str, tuple[Any, float]] = {}
_PROFILE_TTL_SECONDS = 300  # 5 minutes default TTL


def get_profile_with_cache(
    profile_name: str,
    config: Optional[Dict[str, Any]] = None,
    ttl_seconds: int = _PROFILE_TTL_SECONDS,
) -> Optional[Any]:
    """Get a profile with caching to avoid repeated loading.

    Profiles are cached with a TTL (time-to-live) to avoid repeatedly
    loading the same profile from disk within a short time window.

    Args:
        profile_name: Name of the profile to load
        config: Optional YAML config to pass to profile loader
        ttl_seconds: Cache TTL in seconds (default: 300 = 5 minutes)

    Returns:
        Profile instance, or None if not found
    """
    import time

    current_time = time.time()

    # Check cache
    if profile_name in _profile_cache:
        profile, cached_time = _profile_cache[profile_name]
        age = current_time - cached_time
        if age < ttl_seconds:
            # Cache hit within TTL
            return profile
        else:
            # Cache expired
            del _profile_cache[profile_name]

    # Cache miss - load profile
    from src.extensions.content.deep_research.profiles import load_profile

    try:
        profile = load_profile(profile_name, config=config or {})
        if profile:
            _profile_cache[profile_name] = (profile, current_time)
        return profile
    except Exception as e:
        # CRITICAL: Profile load failure is catastrophic - re-raise
        # Note: run_logger may not be available at module level, so we re-raise directly
        raise RuntimeError(f"Profile load failed for '{profile_name}': {e}") from e


def clear_profile_cache() -> None:
    """Clear the profile cache.

    This can be useful for testing or when profiles are updated.
    """
    global _profile_cache
    _profile_cache = {}


def set_profile_ttl(ttl_seconds: int) -> None:
    """Set the global profile cache TTL.

    Args:
        ttl_seconds: New TTL in seconds
    """
    global _PROFILE_TTL_SECONDS
    _PROFILE_TTL_SECONDS = ttl_seconds


# =============================================================================
# COMMON VALIDATION HELPERS
# =============================================================================

def validate_node_state(
    state: GlobalState,
    required_keys: List[str],
    node_name: str = "node",
) -> None:
    """Validate that required keys exist in the state.

    Args:
        state: GlobalState to validate
        required_keys: List of required key names
        node_name: Name of the node for error message

    Raises:
        ValueError: If any required key is missing
    """
    missing_keys = [key for key in required_keys if key not in state or state[key] is None]
    if missing_keys:
        raise ValueError(
            f"{node_name} requires the following state keys: {', '.join(missing_keys)}. "
            f"Missing: {missing_keys}"
        )


# =============================================================================
# THRESHOLDS - Read from config with fallbacks
# =============================================================================

def get_thresholds(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get all thresholds from config with fallbacks.

    Args:
        yaml_config: The full YAML configuration

    Returns:
        Dict with all threshold values
    """
    thresholds = yaml_config.get("thresholds", {})

    return {
        # Confidence thresholds (0-100 scale)
        "confidence_high": thresholds.get("confidence", {}).get("high", 70),
        "confidence_medium": thresholds.get("confidence", {}).get("medium", 50),
        "confidence_low": thresholds.get("confidence", {}).get("low", 30),
        "confidence_default": thresholds.get("confidence", {}).get("default", 50),
        # Quality thresholds (0-100 scale)
        "quality_default": thresholds.get("quality", {}).get("default", 70),
        # Token limits
        "max_chars_per_chunk": thresholds.get("tokens", {}).get("max_chars_per_chunk", 24000),
        "max_output_length": thresholds.get("tokens", {}).get("max_output_length", 500),
        # Planning parameters
        "min_angles": thresholds.get("planning", {}).get("min_angles", 3),
        "max_angles": thresholds.get("planning", {}).get("max_angles", 5),
    }


def validate_config(
    yaml_config: Dict[str, Any],
    required_keys: List[str],
    node_name: str = "node",
) -> None:
    """Validate that required keys exist in the YAML config.

    Args:
        yaml_config: YAML config to validate
        required_keys: List of required key names
        node_name: Name of the node for error message

    Raises:
        ValueError: If any required key is missing
    """
    missing_keys = [key for key in required_keys if key not in yaml_config or yaml_config[key] is None]
    if missing_keys:
        raise ValueError(
            f"{node_name} requires the following config keys: {', '.join(required_keys)}. "
            f"Missing: {missing_keys}"
        )


# =============================================================================
# UNIFIED FLASH EXPLORER SPAWNER PATTERN
# =============================================================================

# Lab Architect uses this unified pattern for spawning Flash Explorers.
# The pattern is reusable for different planning contexts.


async def _plan_avenues(
    input_data: str,
    yaml_config: Dict[str, Any],
    run_logger: Any = None,
    planning_context: Optional[str] = None,
) -> List[str]:
    """Plan research avenues using LLM with STRUCTURED OUTPUT.

    Used by Lab Architect for iterative research planning.

    Args:
        input_data: Query or worker_tasks JSON for planning
        yaml_config: Full YAML configuration
        run_logger: Optional logger
        planning_context: Optional additional context (e.g., use_case, brand info)

    Returns:
        List of research avenue strings
    """
    from ..helpers.llm import create_llm_from_config
    from pydantic import BaseModel, Field, field_validator

    class ResearchPlan(BaseModel):
        """Research plan with avenues."""
        avenues: List[str] = Field(..., min_length=1, description="Research avenues to investigate")

        @field_validator('avenues')
        @classmethod
        def validate_avenues(cls, v: List[str]) -> List[str]:
            """Validate that avenues are meaningful research topics, not garbage."""
            import re
            validated = []
            for avenue in v:
                # Skip empty strings
                if not avenue or not avenue.strip():
                    continue
                # Skip strings that are just JSON fragments or garbage
                # Patterns like "]}", "```", "{", "}" are not valid research avenues
                if re.match(r'^[\[\]{}(),"`\s]+$', avenue.strip()):
                    raise ValueError(
                        f"Invalid research avenue: '{avenue}' appears to be a JSON fragment, "
                        f"not a meaningful research topic. Each avenue must be a descriptive "
                        f"research angle like 'Investigate gym motivation techniques'"
                    )
                # Skip very short strings (less than 10 chars = not meaningful)
                if len(avenue.strip()) < 10:
                    raise ValueError(
                        f"Research avenue too short: '{avenue}'. "
                        f"Each avenue must be at least 10 characters and describe a specific research angle."
                    )
                validated.append(avenue.strip())

            if not validated:
                raise ValueError(
                    "No valid research avenues extracted. Each avenue must be a meaningful "
                    "research topic description, not a JSON fragment or single word."
                )
            return validated

    # Use config to control thinking mode (spawner should have thinking ON for complex planning)
    llm = await create_llm_from_config(yaml_config, role="spawner")

    # Build context section if provided
    context_section = ""
    if planning_context:
        context_section = f"""
ADDITIONAL CONTEXT:
{planning_context}
"""

    # ARCHITECTURE: There is only ONE user message - the original query.
    # Config prompts are SYSTEM messages (define role/behavior).
    # Internal context (research_avenue, etc.) is added as additional system messages.

    # Step 1: Get SYSTEM instructions from config
    system_prompt = get_prompt_from_config(
        yaml_config,
        "flash_explorer.planning.system",
    )

    # Build context sections (SYSTEM-GENERATED, not user!)
    context_sections = []
    context_sections.append(f"Current task: Generate research avenues for: {input_data}")
    if context_section.strip():
        context_sections.append(context_section)

    # Build properly structured messages using new builders
    # input_data IS the original human query (THE ONLY USER MESSAGE)
    messages = build_context_messages(
        system_prompt=system_prompt,
        human_query=input_data,  # THE ONLY USER MESSAGE
        context_sections=context_sections,  # SYSTEM-GENERATED
    )

    # Validate message structure
    issues = validate_message_roles(messages)
    if issues:
        if run_logger:
            run_logger._write_to_file(f"[AVENUE_PLANNER] Message structure issues: {issues}")

    # DEBUG: Log the actual messages being sent
    if run_logger:
        run_logger._write_to_file(f"[AVENUE_PLANNER] Message structure: {get_message_summary(messages)}")
        run_logger._write_to_file(f"[AVENUE_PLANNER] System prompt:\n{system_prompt[:500]}...")
        run_logger._write_to_file(f"[AVENUE_PLANNER] User query: {input_data}")

    # Step 2: Get free-form response with proper message structure
    response = await llm.ainvoke(messages)

    # Use helper to extract content (handles thinking mode where content is in reasoning field)
    from ..helpers.deep_agent_helpers import get_llm_response_content
    agent_output = get_llm_response_content(response, run_logger=run_logger)

    # DEBUG: Log the extracted content
    if run_logger:
        run_logger._write_to_file(f"[AVENUE_PLANNER] Raw response content length: {len(response.content) if hasattr(response, 'content') else 0}")
        run_logger._write_to_file(f"[AVENUE_PLANNER] Response has reasoning: {hasattr(response, 'reasoning') and bool(response.reasoning)}")
        run_logger._write_to_file(f"[AVENUE_PLANNER] Extracted agent_output ({len(agent_output)} chars): {agent_output[:500]}...")

    # DIAGNOSTIC: Validate LLM response is relevant to the input
    from ...observability.diagnostic_logging import validate_llm_response_relevance
    is_relevant, reason, diagnostic = validate_llm_response_relevance(
        original_query=input_data,
        llm_response=agent_output,
        context="avenue_planning",
        run_logger=run_logger,
        min_overlap=0.05,  # Lower threshold for planning
    )
    if not is_relevant:
        # Log critical issue but continue - the extraction may still work
        if run_logger:
            run_logger._write_to_file(
                f"[DIAGNOSTIC] AVENUE_PLANNER_OFF_TOPIC: {reason}"
            )

    # Step 3: Extract structured output using extract_structured_output
    # Load extraction description from YAML config (not hardcoded)
    from ..helpers.deep_agent_helpers import extract_structured_output
    research_plan = await extract_structured_output(
        llm=llm,
        agent_output=agent_output,
        pydantic_model=ResearchPlan,
        extraction_prompt_key="extraction.research_plan",
        role="avenue_planner",
        yaml_config=yaml_config,
        run_logger=run_logger
    )

    if run_logger:
        run_logger._write_to_file(f"[UNIFIED] Planned {len(research_plan.avenues)} avenues")

    return research_plan.avenues


async def _run_flash_explorer(
    research_avenue: str,
    session_id: str,
    context_engine: Any,
    yaml_config: Dict[str, Any],
    run_logger: Any = None,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a Flash Explorer for a single research avenue.

    Used by Lab Architect for iterative research.
    """
    from ...agents.flash_explorer import run_flash_explorer

    spawner_config = yaml_config.get("spawner_config", {})
    flash_recursion_limit = spawner_config.get("flash_recursion_limit", 50)

    result = await run_flash_explorer(
        research_avenue=research_avenue,
        session_id=session_id,
        context_engine=context_engine,
        yaml_config=yaml_config,
        recursion_limit=flash_recursion_limit,
        run_logger=run_logger,
        store=store,  # Pass LangGraph Store for memory persistence
    )

    findings = result.get("structured_findings", {})
    artifacts_created = result.get("artifacts_created", [])

    return {
        **findings,
        "artifacts_created": artifacts_created,
    }


async def _synthesize_results(
    explorer_results: List[Dict[str, Any]],
    yaml_config: Dict[str, Any],
    run_logger: Any = None,
) -> Dict[str, Any]:
    """Synthesize Flash Explorer results with Pydantic validation.

    Used by Lab Architect for iterative research synthesis.

    Args:
        explorer_results: Results from Flash Explorers
        yaml_config: Full configuration
        run_logger: Optional logger

    Returns:
        Dict (Pydantic model serialized to dict for LangGraph state)

    PATTERN: Validate with Pydantic at output boundary, return dict for state.
    Consumers use dict access: brief.get("original_query")
    """
    import pydantic
    from ..helpers.llm import create_llm_from_config

    # Use researcher role for synthesis (thinking OFF in config - better for structured output)
    llm = await create_llm_from_config(yaml_config, role="researcher")

    # NOTE: Pydantic model passed in via yaml_config context
    # This allows different synthesis models to be used
    pydantic_model_class = yaml_config.get("_synthesis_model", None)
    if pydantic_model_class is None:
        raise RuntimeError("_synthesis_model must be set in yaml_config")

    # Use native structured output
    llm_structured = llm.with_structured_output(pydantic_model_class)

    # Build findings summary (SYSTEM-GENERATED, not user!)
    findings_summary = "\n\n".join([
        f"=== RESULT {i+1} ===\nSummary: {r.get('summary', '')[:500]}\nKey Findings: {r.get('key_findings', [])[:3]}"
        for i, r in enumerate(explorer_results)
    ])

    # Get synthesis prompt from config (CRASH-ONLY - no fallbacks)
    # Note: The prompt template includes findings_summary as a placeholder
    # Since findings are SYSTEM-GENERATED, we use a single system message
    prompt = get_prompt_from_config(
        yaml_config,
        "flash_explorer.synthesis",
        num_explorers=len(explorer_results),
        findings_summary=findings_summary,
    )

    # Use build_system_only since synthesis prompt contains everything
    # (findings are embedded in the prompt via template placeholders)
    messages = build_system_only(prompt)

    validated_result = await llm_structured.ainvoke(messages)

    if run_logger:
        run_logger._write_to_file(f"[UNIFIED] Synthesis Pydantic validation passed")

    # Return dict for LangGraph state (TypedDict pattern)
    return validated_result.model_dump()


async def unified_flash_explorer_spawner(
    state: Dict[str, Any],
    next_phase: str,
    output_key: str,
    synthesis_model: type,
    input_for_planning: str,
    run_logger: Any = None,
    planning_context: Optional[str] = None,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """UNIFIED Flash Explorer spawner - used by Lab Architect.

    Spawns parallel Flash Explorers for research avenues.

    Args:
        state: GlobalState
        next_phase: Next phase to route to
        output_key: State key for result
        synthesis_model: Pydantic model for synthesis validation
        input_for_planning: Input data for planning avenues
        run_logger: Optional logger
        planning_context: Optional context for planning (e.g., brand info for content creation)
        store: Optional LangGraph Store for memory persistence

    Returns:
        Dict with output_key and current_phase
    """
    import asyncio
    import time

    total_start = time.time()
    timings = {}

    yaml_config = get_yaml_config_from_context()
    context_engine = get_context_engine_from_context()  # Use context variable, not yaml_config
    session_id = state.get("session_id")

    # Store synthesis model in yaml_config for _synthesize_results to use
    yaml_config["_synthesis_model"] = synthesis_model

    # Plan research avenues
    plan_start = time.time()
    avenues = await _plan_avenues(input_for_planning, yaml_config, run_logger, planning_context)
    timings["planning"] = time.time() - plan_start

    if run_logger:
        run_logger._write_to_file(f"[UNIFIED] Planned {len(avenues)} avenues in {timings['planning']:.2f}s")

    # Spawn Flash Explorers in parallel
    explorer_start = time.time()
    explorer_tasks = [
        _run_flash_explorer(avenue, session_id, context_engine, yaml_config, run_logger, store)
        for avenue in avenues
    ]

    results = await asyncio.gather(*explorer_tasks, return_exceptions=True)
    successful_results = [r for r in results if not isinstance(r, Exception)]
    timings["explorers"] = time.time() - explorer_start

    # DIAGNOSTIC: Track Flash Explorer results
    from ...observability.diagnostic_logging import diagnose_flash_explorer_batch, track_flash_explorer_result
    for i, (avenue, result) in enumerate(zip(avenues, results)):
        if isinstance(result, Exception):
            track_flash_explorer_result(
                avenue=avenue,
                success=False,
                elapsed_seconds=0,  # Unknown per-explorer time
                run_logger=run_logger,
                error=result,
            )

    # Log batch summary
    diagnose_flash_explorer_batch(
        results=results,
        elapsed_total=timings["explorers"],
        run_logger=run_logger,
    )

    if run_logger:
        run_logger._write_to_file(
            f"[UNIFIED] Flash Explorers: {len(successful_results)}/{len(explorer_tasks)} successful in {timings['explorers']:.2f}s"
        )

    # Synthesize results with Pydantic validation
    synth_start = time.time()
    synthesis = await _synthesize_results(successful_results, yaml_config, run_logger)
    timings["synthesis"] = time.time() - synth_start

    total_elapsed = time.time() - total_start
    if run_logger:
        run_logger._write_to_file(
            f"[UNIFIED] Spawner complete: total={total_elapsed:.2f}s "
            f"(planning={timings['planning']:.2f}s, explorers={timings['explorers']:.2f}s, synthesis={timings['synthesis']:.2f}s)"
        )

    return {
        output_key: synthesis,
        "current_phase": next_phase,
    }
