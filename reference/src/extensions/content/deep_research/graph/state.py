"""Global State - Main state definition and factory functions.

This module provides the GlobalState TypedDict and factory functions for the
deep research system. The Pydantic models, reducers, and other types have been
split into separate modules for better organization.

Related modules:
- state_types.py: Input/output Pydantic models (ResearchInput, ResearchOutput, etc.)
- state_models.py: State data models (Citation, WorkerResult, BossDecision, etc.)
- state_reducers.py: Reducer functions (merge_messages, merge_worker_results, etc.)
"""

from typing import Annotated, List, Dict, Any, Optional, Tuple
from datetime import datetime
from typing_extensions import TypedDict
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.managed import RemainingSteps


def last_value_reducer(left: Any, right: Any) -> Any:
    """Reducer that keeps the last value when multiple writes occur.

    Used for fields like current_phase where multiple nodes might write
    in the same step, but we only care about the final value.
    """
    return right if right is not None else left


# Import Pydantic models from split modules
from .state_types import (
    ResearchInput,
    ResearchOutput,
    ResearchMetadata,
    ResearchRuntimeConfig,
)
from .state_models import (
    Citation,
    WorkerResult,
    ResearchBrief,
    PromptAdjustment,
    CritiqueResult,
)
from .state_reducers import (
    merge_messages,
    merge_worker_results,
    merge_citations,
    overwrite_reducer,
    merge_prompt_adjustments,
)


# =============================================================================
# GLOBAL STATE (TypedDict for LangGraph with annotated reducers)
# =============================================================================

class GlobalState(TypedDict):
    """Global state for Deep Research V4 Graph with Pydantic validation.

    This state flows through all phases:
    - Phase I: Boss (Initial Planning) - Query analysis and research strategy
    - Phase II: Lab (Workers) - Parallel research execution
    - Phase III: Lab Architect (Iterative Planning) - Additional research planning
    - Phase IV: Critic (Quality Gate) - Quality assessment
    - Phase V: Synthesis - Final output generation

    The state uses LangGraph's Annotated reducers for automatic merging
    of concurrent updates from parallel workers.
    """
    # ===== User Input =====
    query: str  # Changed from raw_query to match ResearchInput
    # Using plain List without reducer to avoid LangGraph type conflicts
    # The add_messages reducer can be applied manually if needed
    chat_history: List[Any]
    user_profile: Dict[str, Any]

    # ===== Phase I Artifacts (Boss) =====
    # TypedDict pattern: Use dict for state, validate with Pydantic at boundaries only
    # LangGraph checkpoints serialize to dict anyway, so keep schema as dict
    research_brief: Optional[Dict[str, Any]]

    # ===== Phase II Artifacts (Boss) =====
    routing_decision: str  # LastValue channel
    worker_completed: bool  # Whether worker has been run
    workers_to_spawn: List[str]
    worker_tasks: Dict[str, str]
    worker_dependencies: Optional[Dict[str, List[str]]]
    worker_types: Optional[Dict[str, str]]
    worker_specialized_prompts: Dict[str, str]  # Specialized prompts for different worker types
    worker_roles: Optional[Dict[str, str]]
    iteration_count: int
    # Boss iteration counter for circuit breaker (prevents infinite loops)
    # Used by unified_boss_node and unified_critic_node to track research iterations
    boss_iteration_count: Annotated[int, last_value_reducer]
    boss_decision: Annotated[str, last_value_reducer]  # Last boss decision (spawn_worker, spawn_architects, synthesize)
    boss_guidance: Optional[str]

    # ===== Prompt Adjustment =====
    # Using plain List without reducer to avoid LangGraph type conflicts
    # The reducer merge_prompt_adjustments can be applied manually if needed
    prompt_adjustments: List[Any]

    # ===== Phase III Artifacts (Lab) =====
    # Using plain List without reducer to avoid LangGraph type conflicts
    # The reducer merge_worker_results can be applied manually if needed
    worker_outputs: List[Any]

    # Lab Architect findings (Pro-Flash hybrid architecture)
    architect_findings: Optional[Dict[str, Any]]

    # ===== Session Management (Golden Thread) =====
    # CRITICAL: session_id from SessionManager - single source of truth across ALL storage
    # This UUID links: Postgres artifacts, Neo4j nodes, MinIO files, LangGraph checkpoints
    # ALL data access MUST go through Context Engine using this session_id
    # Annotated with overwrite_reducer to ensure it's persisted to checkpoints
    session_id: Annotated[str, overwrite_reducer]
    # run_id from RunLogger - for observability/logging only (not used for storage)
    run_id: Annotated[str, overwrite_reducer]

    # ===== Research Brief Summary: Condensed Tier A summary for Critic =====
    research_brief_summary: Optional[str]

    # ===== Phase IV Artifacts (Critic) =====
    critique: Optional[CritiqueResult]
    critique_history: List[str]
    # Critic feedback for boss (when rejecting synthesis and requesting another iteration)
    critic_feedback: Annotated[Optional[str], last_value_reducer]

    # ===== Synthesis Draft (for Critic evaluation) =====
    draft_content: Optional[str]

    # ===== Critic Instructions for Re-synthesis =====
    # When critic detects synthesis issues (not worker issues), it can request
    # re-synthesis with specific instructions without rerunning workers
    critic_instructions_for_synthesis: Optional[str]

    # ===== Final Output (from ResearchOutput) =====
    final_response: Optional[str]
    final_sources: List[str]
    # Using plain List without reducer to avoid LangGraph type conflicts
    # The merge_citations reducer can be applied manually if needed
    all_citations: List[Any]

    # ===== Metadata =====
    start_time: str
    end_time: Optional[str]
    # Use last_value_reducer to handle multiple writes in the same step
    # (e.g., when synthesis and critic both update current_phase)
    current_phase: Annotated[str, last_value_reducer]

    # ===== Observability Metadata =====
    # These fields are stored in checkpoints for observability UI
    _use_case: str  # Research use case (rag, longform_gen, soc_media_gen, etc.)
    _variant: str  # Research variant configuration

    # ===== Refinement (Experimental) =====
    is_refinement: bool
    refine_feedback: str
    refine_custom_flow: Optional[Dict[str, Any]]
    previous_findings: str

    # ===== Recursion Limit Handling (NEW) =====
    # Managed value that tracks remaining steps before hitting recursion limit
    remaining_steps: RemainingSteps

    # ===== Dynamic Worker Results (for Send API parallel execution) =====
    # List of (worker_id, result) tuples for parallel worker execution
    # Uses operator.add to append results from each worker
    _worker_results_partial: Annotated[List[Tuple[str, Any]], operator.add]

    # ===== Sub-Worker Support (for parallel research) =====
    # List of sub-topic specs to spawn as parallel sub-workers
    sub_workers_to_spawn: List[Dict[str, str]]
    # List of research briefs from parallel sub-workers
    # Uses operator.add to append briefs from each sub-worker
    _sub_worker_briefs_partial: Annotated[List[Any], operator.add]
    # LLM instance passed from worker to sub-workers
    _worker_llm: Optional[Any]

    # ===== Tool Model Configuration =====
    # Provider for tool/sub-agent LLMs (e.g., 'llamacpp' for fast local models)
    # Resolved from models.tool slot in config and bound to sub-agents/tools
    # This allows tools like web_research_agent to use fast local models
    # instead of the main cloud model, reducing load on main model
    tool_model_provider: Optional[str]

    # ===== Graph Builder Output =====
    # Results from graph_builder_node - knowledge graph construction
    # Includes: chunks_processed, entities_extracted, relationships_created, communities_detected
    graph_builder_output: Optional[Dict[str, Any]]
    # Whether graph was built successfully
    graph_built: Optional[bool]
    # Error message if graph building failed
    graph_error: Optional[str]
    # Community summaries for synthesis to use
    community_summaries: Optional[List[Dict[str, Any]]]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_global_state(
    query: str,
    chat_history: List[BaseMessage] = None,
    use_case: str = "rag",
    variant: str = "default",
    tool_model_provider: Optional[str] = None,
) -> dict:
    """Create initial global state with validation.

    Validates input using ResearchInput Pydantic model before creating state.

    Args:
        query: User query
        chat_history: Optional conversation history
        use_case: Research use case (rag, longform_gen, soc_media_gen, etc.)
        variant: Research variant configuration
        tool_model_provider: Provider for tool/sub-agent LLMs (e.g., 'llamacpp')

    Returns:
        Initial state as dict compatible with GlobalState TypedDict

    Raises:
        ValidationError: If input validation fails
    """
    # Validate input using Pydantic
    validated_input = ResearchInput(query=query)

    return {
        "query": validated_input.query,
        "chat_history": chat_history or [],
        "user_profile": {},
        "research_brief": None,
        "routing_decision": "standard",
        "worker_completed": False,  # Track if worker has been run
        "workers_to_spawn": [],
        "worker_tasks": {},
        "worker_dependencies": None,
        "worker_types": None,
        "worker_roles": None,
        "iteration_count": 1,
        "boss_iteration_count": 0,  # Circuit breaker counter
        "boss_decision": "",  # Last boss decision
        "boss_guidance": None,
        "prompt_adjustments": [],
        "worker_outputs": [],
        "architect_findings": None,  # Lab Architect findings (Pro-Flash hybrid)
        "research_brief_summary": None,
        "critique": None,
        "critique_history": [],
        "critic_feedback": None,  # Feedback from critic to boss
        "draft_content": None,
        "critic_instructions_for_synthesis": None,  # Initialize new field
        "final_response": None,
        "final_sources": [],
        "all_citations": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "current_phase": "worker",
        "is_refinement": False,
        "refine_feedback": "",
        "refine_custom_flow": None,
        "previous_findings": "",
        # Add use_case and variant to state for observability
        "_use_case": use_case,
        "_variant": variant,
        # Initialize partial worker results list for Send API
        "_worker_results_partial": [],
        # Initialize sub-worker support fields
        "sub_workers_to_spawn": [],
        "_sub_worker_briefs_partial": [],
        "_worker_llm": None,
        # Bind tool model provider for sub-agents/tools
        "tool_model_provider": tool_model_provider,
        # CRITICAL: session_id and run_id MUST be set by run_v4_spawner via SessionManager
        # These are REQUIRED for all Context Engine operations - no fallbacks
        "session_id": "",  # MUST be set by run_v4_spawner
        "run_id": "",  # MUST be set by run_v4_spawner
        # Graph Builder output - set by graph_builder_node
        "graph_builder_output": None,
        "graph_built": False,
        "graph_error": None,
        "community_summaries": None,
        # remaining_steps is auto-populated by LangGraph, no need to set
    }


def create_research_output(state: GlobalState) -> Dict[str, Any]:
    """Extract ResearchOutput from GlobalState with validation.

    This function validates the output using ResearchOutput Pydantic model
    before returning to the user.

    Args:
        state: Full GlobalState after graph execution

    Returns:
        Validated dict matching ResearchOutput schema

    Raises:
        ValidationError: If output validation fails
    """
    # Build metadata with Pydantic validation
    metadata = ResearchMetadata(
        total_iterations=state.get("iteration_count", 1),
        total_workers=len(state.get("worker_outputs", [])),
        total_sources=len(state.get("final_sources", [])),
        execution_time=0.0,  # Would need to track actual time
        quality_score=state.get("critique", CritiqueResult()).quality_score,
        passed_quality_gate=state.get("critique", CritiqueResult()).passed_quality_gate,
        timestamps={
            "start": state.get("start_time", ""),
            "end": state.get("end_time", ""),
        },
    )

    # Build output with Pydantic validation
    output = ResearchOutput(
        response=state.get("final_response", ""),
        sources=state.get("final_sources", []),
        citations=[],  # Would need to convert Citation models
        metadata=metadata,
    )

    return output.model_dump()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main state definition
    "GlobalState",
    # Factory functions
    "create_global_state",
    "create_research_output",
    # Re-export commonly used types for convenience
    "ResearchInput",
    "ResearchOutput",
    "ResearchRuntimeConfig",
    "ResearchMetadata",
    "Citation",
    "WorkerResult",
    "ResearchBrief",
    "PromptAdjustment",
    "CritiqueResult",
    # Reducers
    "merge_messages",
    "merge_worker_results",
    "merge_citations",
    "overwrite_reducer",
]
