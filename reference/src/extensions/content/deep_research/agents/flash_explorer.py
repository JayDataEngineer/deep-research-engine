"""Flash Explorer Agent - Autonomous research agent with internal critic.

The Flash Explorer receives a RESEARCH AVENUE (not a specific query) and autonomously:
- Plans how to investigate the avenue
- Executes multiple searches from different angles IN PARALLEL
- Evaluates source quality and recency
- Synthesizes findings into structured output
- Writes everything to Context Engine (async side effect)
- Returns only structured findings to Lab Architect

This is NOT a tool wrapper - it's an autonomous researcher with its own critic loop.

DEEP AGENTS ALL THE WAY DOWN:
The Flash Explorer ONLY spawns deep agents (NO direct tool access):
- web_deep_agent: For all web research (handles web_search, web_get, web_map, web_crawl)
- pdf_deep_agent: For document RAG (handles vector search in documents)
- community_deep_agent: For community sentiment (handles reddit, hackernews, lemmy, 4chan)

The Flash Explorer does NOT have direct access to:
- web_search, web_get, web_map, web_crawl
- vector_search, add_documents, list_collections
- community_scrape (use community_deep_agent instead)

All research must go through the proper deep agent graphs.

PARALLEL EXECUTION:
- ALL research angles are executed in parallel using asyncio.gather
- Within each angle, web/pdf/community deep agents ALSO run in parallel
- This maximizes speed by eliminating sequential waits

BEST PRACTICES IMPLEMENTED:
- Proactive Recursion Handling: Manual step tracking with configurable threshold
- Structured Output: Pydantic validation with with_structured_output()
- Input/Output Schemas: Clean API boundaries
- Streaming: Real-time progress updates via events
- Checkpointing: Full LangGraph checkpointing support
- Parallel Execution: All angles and deep agents run in parallel
"""

from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from datetime import datetime
import logging
import sys

import json
import re

# Use centralized logging with component-specific level
from ..logging_utils import get_logger, log_timing_breakdown

logger = get_logger(__name__, "flash_explorer")  # Component-specific level from config.yaml


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _enforce_token_limit(prompt: str, max_chars: int = 24000, run_logger=None) -> str:
    """Enforce token limit on prompt to prevent context overflow.

    Args:
        prompt: The prompt string
        max_chars: Maximum characters (roughly 6k tokens at 4 chars/token)
        run_logger: Optional run logger for logging

    Returns:
        Truncated prompt if necessary
    """
    if len(prompt) <= max_chars:
        return prompt

    if run_logger:
        run_logger._write_to_file(
            f"[TOKEN_LIMIT] Prompt ({len(prompt)} chars) exceeds limit ({max_chars}). Truncating."
        )

    logger.warning(
        f"[TOKEN_LIMIT] Prompt ({len(prompt)} chars) exceeds limit ({max_chars}). Truncating."
    )

    # Truncate and add marker
    return prompt[:max_chars] + "\n\n[...content truncated due to length...]"


def _needs_community_sentiment(query: str) -> bool:
    """Detect if a query needs community sentiment research.

    Returns True if the query is asking about opinions, sentiment,
    or what people/communities think about a topic.

    Args:
        query: The research query to analyze

    Returns:
        True if community_deep_agent should be spawned
    """
    query_lower = query.lower()

    # Direct sentiment/opinion keywords
    sentiment_keywords = [
        "what do people think",
        "what does reddit think",
        "what does the community think",
        "public opinion",
        "community sentiment",
        "user sentiment",
        "how do people feel",
        "what are people saying",
        "opinion on",
        "thoughts on",
        "sentiment",
        "reaction",
        "reactions",
    ]

    for keyword in sentiment_keywords:
        if keyword in query_lower:
            return True

    # Question patterns that suggest sentiment research
    sentiment_patterns = [
        "people think about",
        "users think about",
        "community thinks",
        "everyone thinks",
        "reddit says",
        "twitter says",
        "social media",
        "discussion about",
        "debate about",
    ]

    for pattern in sentiment_patterns:
        if pattern in query_lower:
            return True

    return False


# =============================================================================
# STATE SCHEMAS (Best Practices: Input/Output separation)
# =============================================================================

class ExplorerInputState(TypedDict, total=False):
    """Input state for Flash Explorer Agent - public API.

    These are the inputs that external callers provide to the Flash Explorer.
    total=False makes all fields optional for TypedDict.
    """
    # ===== Required Input from Lab Architect =====
    research_avenue: str  # The broad topic to investigate (e.g., "quantitative employment data")
    session_id: str  # For Context Engine writes
    context_engine: Any  # For writing findings
    yaml_config: Dict[str, Any]  # Full config for LLM/tools (passed to deep agents)

    # ===== Optional Input for RAG =====
    documents: List[str]  # File paths to local documents (PDF, TXT) - enables RAG mode

    # ===== Optional LangGraph Store for memory persistence =====
    store: Optional[Any]  # LangGraph Store (BaseStore) for deep agents to save findings


class ExplorerOutputState(TypedDict):
    """Output state from Flash Explorer Agent - public API.

    These are the outputs that the Flash Explorer returns to callers.
    """
    # ===== Structured Findings =====
    structured_findings: Dict[str, Any]  # Synthesized findings for Lab Architect
    artifacts_created: List[str]  # IDs of artifacts written to Context Engine


class ExplorerPrivateState(TypedDict):
    """Private state for Flash Explorer Agent - internal implementation.

    These fields are NOT part of the public API and are only used internally.
    """
    # ===== Planning =====
    research_plan: List[str]  # List of angles/queries to investigate
    current_angle: int  # Which angle we're currently exploring

    # ===== Best Practices: Proactive recursion handling =====
    # We track remaining steps manually for graceful degradation
    # The value is decremented as the graph progresses
    remaining_steps: int  # Steps remaining until recursion limit

    # ===== Deep Agent Results =====
    deep_agent_results: List[Dict[str, Any]]  # Results from deep agents

    # ===== Evaluation =====
    evaluation: Dict[str, Any]  # Quality assessment of findings
    sufficient_info: bool  # Does internal critic have enough info?

    # ===== Control =====
    next_action: Literal["plan", "search", "evaluate", "synthesize", "done"]


class ExplorerState(ExplorerInputState, ExplorerOutputState, ExplorerPrivateState):
    """Complete state for Flash Explorer Agent.

    Combines input, output, and private state for the graph.
    This follows the best practice of separating public/private interfaces.
    """
    # ===== Internal: RunLogger for observability (not part of public API) =====
    run_logger: Optional[Any] = None  # RunLogger for structured logging


# =============================================================================
# CONFIGURATION
# =============================================================================

# Helper to get best practices config with defaults
def _get_best_practice_config(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get best practices configuration from yaml_config with defaults.

    Default values here match config.yaml defaults. Config values take precedence.
    """
    bp_config = yaml_config.get("best_practices", {})
    return {
        "tool_token_limit_before_evict": bp_config.get("tool_token_limit_before_evict", 20000),
        "low_steps_threshold": bp_config.get("low_steps_threshold", 3),  # Force synthesis when remaining steps <= this
        "streaming_enabled": bp_config.get("streaming_enabled", True),
        "structured_output_enabled": bp_config.get("structured_output_enabled", True),
    }


# =============================================================================
# STREAMING EVENTS
# =============================================================================

class ExplorerEvent:
    """Streaming events from Flash Explorer for real-time updates."""

    @staticmethod
    def planning(research_avenue: str, remaining_steps: int) -> Dict[str, Any]:
        return {
            "type": "explorer_planning",
            "research_avenue": research_avenue,
            "remaining_steps": remaining_steps,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def searching(angle: str, angle_index: int, remaining_steps: int) -> Dict[str, Any]:
        return {
            "type": "explorer_searching",
            "angle": angle,
            "angle_index": angle_index,
            "remaining_steps": remaining_steps,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def evaluating(sufficient: bool, confidence: float, remaining_steps: int) -> Dict[str, Any]:
        return {
            "type": "explorer_evaluating",
            "sufficient": sufficient,
            "confidence": confidence,
            "remaining_steps": remaining_steps,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def synthesizing(avenue: str, loops_completed: int) -> Dict[str, Any]:
        return {
            "type": "explorer_synthesizing",
            "avenue": avenue,
            "loops_completed": loops_completed,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def complete(findings: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "explorer_complete",
            "findings": findings,
            "timestamp": datetime.now().isoformat()
        }


# =============================================================================
# GRAPH NODES
# =============================================================================

async def planning_node(state: ExplorerState, config) -> Dict[str, Any]:
    """Plan how to investigate the research avenue.

    The explorer decides what angles to explore, what queries to run,
    and what would constitute sufficient information.

    This is where the autonomy happens - the explorer figures out HOW
    to research, not just WHAT to research.

    Best Practices:
    - Uses RemainingSteps for proactive recursion handling
    - Uses "Draft then Extract" pattern for structured output
    - Emits streaming events for real-time updates
    """
    from ..graph.helpers.llm import create_llm_from_config
    from ..graph.helpers.deep_agent_helpers import extract_structured_output, get_llm_response_content
    from pydantic import BaseModel, Field, field_validator
    from langchain_core.runnables import RunnableConfig

    research_avenue = state["research_avenue"]
    remaining_steps = state.get("remaining_steps", 100)
    documents = state.get("documents", [])
    yaml_config = state.get("yaml_config", {})
    run_logger = state.get("run_logger")

    # CRITICAL: Initialize new_remaining_steps EARLY so exception handler can use it
    # This prevents UnboundLocalError if exception occurs before the decrement line
    new_remaining_steps = max(0, remaining_steps - 1)

    # Get max_angles from config (for E2E test speed)
    spawner_config = yaml_config.get("spawner_config", {})
    max_angles = spawner_config.get("flash_max_angles", 3)  # Default 3, configurable via yaml

    # Simple extraction model for research angles
    class SimpleResearchPlan(BaseModel):
        """Simple research plan with angle strings."""
        angles: List[str] = Field(
            ...,
            min_length=1,
            max_length=max_angles,
            description=f"Research angles to investigate (max {max_angles})"
        )

        @field_validator('angles')
        @classmethod
        def validate_angles(cls, v: List[str]) -> List[str]:
            """Validate that angles are meaningful research topics, not garbage."""
            import re
            validated = []
            for angle in v:
                if not angle or not angle.strip():
                    continue
                # Reject JSON fragments like "]}", "{", "}"
                if re.match(r'^[\[\]{}(),"`\s]+$', angle.strip()):
                    raise ValueError(
                        f"Invalid research angle: '{angle}' is a JSON fragment, not a research topic"
                    )
                # Reject very short strings
                if len(angle.strip()) < 10:
                    raise ValueError(
                        f"Research angle too short: '{angle}'. Must be at least 10 characters."
                    )
                validated.append(angle.strip())
            if not validated:
                raise ValueError("No valid research angles extracted")
            return validated

    # Get best practices config
    bp_config = _get_best_practice_config(yaml_config)
    low_steps_threshold = bp_config["low_steps_threshold"]

    # Best Practices: Proactive recursion handling using RemainingSteps
    # Force synthesis when approaching limit (graceful degradation)
    if new_remaining_steps <= low_steps_threshold:
        logger.warning(
            f"[EXPLORER] Low remaining steps ({new_remaining_steps}), forcing synthesis. "
            f"Research avenue: {research_avenue}"
        )
        return {
            "research_plan": [],
            "next_action": "evaluate",
            "remaining_steps": new_remaining_steps,  # CRITICAL: Track decrement
        }

    # Emit streaming event
    _emit_event(config, ExplorerEvent.planning(research_avenue, remaining_steps))

    # Use Flash model (fast, no thinking)
    yaml_config = state.get("yaml_config", {})
    flash_llm = await create_llm_from_config(yaml_config, role="worker")

    # Get prompt from config with Jinja2 rendering
    from ..graph.helpers.prompts import get_prompt_from_config
    from ..graph.helpers.message_structure import (
        build_context_messages,
        validate_message_roles,
        get_message_summary,
    )

    # ARCHITECTURE: There is only ONE user message in this system - the original query.
    # Everything else (research_avenue, context, etc.) is internally generated.
    #
    # Message structure:
    #   system: "You are a Research Planner..." (from config - defines role)
    #   system: "The research avenue is: X" (internal context - added as system)
    #   user: "Create 5 tweets about gym discipline" (THE original user query)

    # Get the original user query from state (may be in different keys)
    original_query = state.get("query") or state.get("original_query") or research_avenue

    # Step 1: Get SYSTEM instructions from config
    # Config prompts are SYSTEM messages - they define the agent's role
    system_prompt = get_prompt_from_config(
        yaml_config,
        "flash_explorer.planning.system",
    )

    # Build context sections (SYSTEM-GENERATED, not user!)
    context_sections = []
    context_sections.append(f"Current research avenue to plan: {research_avenue}")
    if documents:
        context_sections.append(f"RAG documents available: {len(documents)}")

    # Build properly structured messages using centralized builders
    messages = build_context_messages(
        system_prompt=system_prompt,
        human_query=original_query,  # THE ONLY USER MESSAGE
        context_sections=context_sections,  # SYSTEM-GENERATED
    )

    # Validate message structure
    issues = validate_message_roles(messages)
    if issues:
        logger.warning(f"[EXPLORER] Message structure issues: {issues}")

    try:
        # Step 2: Get free-form response with proper message structure
        # Enforce token limit on each message
        safe_messages = []
        for msg in messages:
            safe_content = _enforce_token_limit(msg["content"], max_chars=8000, run_logger=run_logger)
            safe_messages.append({"role": msg["role"], "content": safe_content})

        response = await flash_llm.ainvoke(safe_messages)
        agent_output = get_llm_response_content(response, run_logger=run_logger)

        # Step 3: Extract structured output using extract_structured_output
        research_plan = await extract_structured_output(
            llm=flash_llm,
            agent_output=agent_output,
            pydantic_model=SimpleResearchPlan,
            role="research_planner",
            yaml_config=yaml_config,
            run_logger=run_logger
        )

        logger.info(
            f"[EXPLORER] Planned {len(research_plan.angles)} angles for: {research_avenue} "
            f"(remaining_steps: {remaining_steps})"
        )

        return {
            "research_plan": research_plan.angles,
            "current_angle": 0,
            "deep_agent_results": [],  # Clear previous results on re-plan
            "next_action": "search",
            "remaining_steps": new_remaining_steps,  # CRITICAL: Track decrement
        }

    except Exception as e:
        logger.error(f"[EXPLORER] Planning failed: {type(e).__name__}: {e}")

        # Fallback plan (no Pydantic validation)
        return {
            "research_plan": [
                f"{research_avenue} overview",
                f"{research_avenue} recent developments",
                f"{research_avenue} statistics and data",
            ],
            "current_angle": 0,
            "deep_agent_results": [],
            "next_action": "search",
            "remaining_steps": new_remaining_steps,  # CRITICAL: Track decrement
        }


async def _execute_angle_research(
    query: str,
    angle_index: int,
    explorer_id: str,
    yaml_config: Dict[str, Any],
    session_id: str,
    documents: List[str],
    config: Any,
    remaining_steps: int,
    store: Optional[Any] = None,  # LangGraph Store for memory persistence
) -> Dict[str, Any]:
    """Execute deep agents for a single research angle.

    This runs web_deep_agent, pdf_deep_agent (if docs), and community_deep_agent (if needed)
    for a single angle. Multiple angles are executed in parallel via asyncio.gather.

    Args:
        query: The research query for this angle
        angle_index: Index of this angle in the research plan
        explorer_id: Explorer ID for logging
        yaml_config: Configuration dict
        session_id: Session ID
        documents: List of document paths (for RAG)
        config: LangGraph config for streaming events
        remaining_steps: Remaining steps counter
        store: Optional LangGraph Store for memory persistence

    Returns:
        Dict with angle results
    """
    import time
    import asyncio

    # Debug: Log entry to function
    logger.info(f"[EXPLORER] _execute_angle_research starting: angle={angle_index}, yaml_config_provided={yaml_config is not None}")

    # Emit streaming event
    _emit_event(config, ExplorerEvent.searching(query, angle_index, remaining_steps))

    angle_query_id = f"{explorer_id}_angle_{angle_index}"
    logger.info(f"[EXPLORER] Spawning deep agents for angle {angle_index}: {query}")

    # Get TOOL model from yaml_config (for web_deep_agent) - OUTSIDE try block to avoid scoping issues
    # This ensures web agents use the same model as other research workers
    from ..graph.helpers.llm import _resolve_model_slot
    tool_model_info = _resolve_model_slot(yaml_config, role="worker")
    tool_model = tool_model_info.get("model")  # e.g., "gpt-oss-20b-UD-Q4_K_XL.gguf"

    if tool_model:
        logger.info(f"[EXPLORER] Using tool model for web_deep_agent: {tool_model}")

    # Determine which deep agents to spawn based on context
    has_web_deep_agent = True  # Always use web deep agent
    has_pdf_deep_agent = len(documents) > 0  # Use PDF deep agent if documents provided
    has_community_deep_agent = _needs_community_sentiment(query)  # Detect sentiment queries

    agents_used = ["web_deep_agent"]
    if has_pdf_deep_agent:
        agents_used.append("pdf_deep_agent")
    if has_community_deep_agent:
        agents_used.append("community_deep_agent")

    # Get TOOL model from yaml_config (for web_deep_agent) - OUTSIDE try block to avoid scoping issues
    # This ensures web agents use the same model as other research workers
    from ..graph.helpers.llm import _resolve_model_slot
    tool_model_info = _resolve_model_slot(yaml_config, role="worker")
    tool_model = tool_model_info.get("model")  # e.g., "gpt-oss-20b-UD-Q4_K_XL.gguf"

    if tool_model:
        logger.info(f"[EXPLORER] Using tool model for web_deep_agent: {tool_model}")

    logger.info(f"[EXPLORER] Angle {angle_index} using deep agents: {', '.join(agents_used)}")

    # Create a runtime dict that Pydantic can validate as ToolRuntime
    # ToolRuntime fields: state, context, config, stream_writer, tool_call_id, store
    # web_research_agent extracts store via: runtime.state.get("store")
    # Pydantic validation: context must be None, stream_writer must be callable
    runtime = None
    if store:
        # Dummy callable for stream_writer (required by Pydantic)
        def _dummy_stream_writer(msg):
            pass

        runtime = {
            "state": {
                "session_id": session_id,
                "store": store,
                "current_worker": f"{explorer_id}",
            },
            "context": None,  # Must be None, not {}
            "config": {},
            "stream_writer": _dummy_stream_writer,  # Must be callable
            "tool_call_id": None,
            "store": store,  # Also pass at top level for direct access
        }
        logger.info(f"[EXPLORER] Created runtime dict with store for web_deep_agent")

    try:
        from src.shared.tools.ulli_langchain.web.web_agent import web_research_agent
        from langgraph.types import Command

        # =====================================================================
        # PARALLEL EXECUTION: Run web, pdf, and community agents in parallel
        # =====================================================================

        async def run_web_agent():
            """Run web deep agent using the full research graph.

            The web_research_agent tool runs the full graph:
            plan → search → select → scrape → extract → critic → write_results

            The write_results node automatically saves to LangGraph Store,
            so Flash Explorer doesn't need to handle persistence.
            """
            web_start = time.time()
            try:
                # Call the web_research_agent tool directly
                # This runs the full graph with planning, URL selection, critic, and save
                result = await web_research_agent.ainvoke({
                    "query": query,
                    "num_results": 10,
                    "max_urls": 5,
                    "max_steps": 3,  # Limit steps for speed in parallel execution
                    "max_refinements": 1,  # Limit refinements for speed
                    "runtime": runtime,  # Pass runtime with store for persistence
                    "tool_model": tool_model,  # Use TOOL model (local llamacpp)
                })

                web_elapsed = time.time() - web_start

                # Extract content from Command result
                if isinstance(result, Command):
                    # Get the ToolMessage content
                    messages = result.update.get("messages", [])
                    if messages:
                        web_findings = messages[0].content
                    else:
                        web_findings = f"Web research completed for: {query}"

                    # Get raw_sources if available
                    web_raw_sources = result.update.get("raw_sources", {})
                else:
                    web_findings = str(result)
                    web_raw_sources = {}

                return web_findings, web_raw_sources, web_elapsed

            except Exception as e:
                web_elapsed = time.time() - web_start
                logger.warning(f"[EXPLORER] web_research_agent failed: {e}")
                return f"Web research failed: {e}", {}, web_elapsed

        async def run_pdf_agent():
            """Run PDF deep agent if documents available."""
            if not has_pdf_deep_agent:
                return "", [], 0.0
            from src.shared.tools.ulli_langchain.memory.pdf_agent import pdf_deep_agent
            pdf_start = time.time()
            pdf_result = await pdf_deep_agent.ainvoke({
                "query": query,
                "mode": "agentic",
                "top_k": 10,
                "max_attempts": 3,
                "runtime": None
            })
            pdf_elapsed = time.time() - pdf_start
            pdf_findings = pdf_result if isinstance(pdf_result, str) else str(pdf_result).strip()
            return pdf_findings, [], pdf_elapsed

        async def run_community_agent():
            """Run community deep agent if sentiment query detected."""
            if not has_community_deep_agent:
                return "", [], 0.0
            from src.shared.tools.ulli_langchain.web.community_deep_agent import simple_community_research
            community_start = time.time()
            try:
                community_findings, community_meta = await simple_community_research(
                    query=query,
                    platform="auto",
                    limit=20
                )
                community_elapsed = time.time() - community_start
                logger.info(
                    f"[EXPLORER] community_deep_agent complete: "
                    f"{community_meta.get('total_posts', 0)} posts, {community_elapsed:.2f}s"
                )
                return community_findings, [], community_elapsed
            except Exception as e:
                logger.warning(f"[EXPLORER] community_deep_agent failed: {e}")
                return "", [], 0.0

        # Run ALL agents in parallel for this angle
        web_task = asyncio.create_task(run_web_agent())
        pdf_task = asyncio.create_task(run_pdf_agent())
        community_task = asyncio.create_task(run_community_agent())

        (web_findings, web_raw_sources, web_elapsed), \
        (pdf_findings, _, pdf_elapsed), \
        (community_findings, _, community_elapsed) = await asyncio.gather(
            web_task, pdf_task, community_task, return_exceptions=False
        )

        # Combine results from deep agents
        combined_findings = f"WEB RESEARCH:\n{web_findings}\n\n"
        if pdf_findings:
            combined_findings += f"DOCUMENT RESEARCH:\n{pdf_findings}\n\n"
        if community_findings:
            combined_findings += f"COMMUNITY SENTIMENT:\n{community_findings}\n\n"

        # Track what was used
        deep_agents_used = ["web_deep_agent"]
        if has_pdf_deep_agent:
            deep_agents_used.append("pdf_deep_agent")
        if has_community_deep_agent:
            deep_agents_used.append("community_deep_agent")
        total_elapsed = web_elapsed + pdf_elapsed + community_elapsed

        # Store the deep agent result
        deep_agent_result = {
            "angle": query,
            "angle_index": angle_index,
            "query_id": angle_query_id,
            "findings": combined_findings[:1000] if len(combined_findings) > 1000 else combined_findings,
            "findings_full": combined_findings,
            "raw_sources": web_raw_sources,
            "confidence": 0.8,
            "deep_agents_used": deep_agents_used,
            "timing": {"web": web_elapsed, "pdf": pdf_elapsed, "community": community_elapsed, "total": total_elapsed},
            "completed_at": datetime.now().isoformat()
        }

        logger.info(
            f"[EXPLORER] Angle {angle_index} done: {total_elapsed:.2f}s "
            f"(web={web_elapsed:.2f}s, pdf={pdf_elapsed:.2f}s, community={community_elapsed:.2f}s)"
        )

        return deep_agent_result

    except Exception as e:
        logger.error(f"[EXPLORER] Deep agents failed for angle {angle_index}: {type(e).__name__}: {e}")
        deep_agents_attempted = ["web_deep_agent"]
        if documents:
            deep_agents_attempted.append("pdf_deep_agent")
        if has_community_deep_agent:
            deep_agents_attempted.append("community_deep_agent")

        return {
            "angle": query,
            "angle_index": angle_index,
            "query_id": angle_query_id,
            "error": str(e),
            "raw_sources": {},  # Empty sources on error
            "deep_agents_attempted": deep_agents_attempted,
            "completed_at": datetime.now().isoformat()
        }


async def search_node(state: ExplorerState, config) -> Dict[str, Any]:
    """Execute deep agents for ALL research angles in PARALLEL.

    This is where "deep agents all the way down" happens.
    The Flash Explorer spawns deep agents (NO direct tool access):

    Deep Agents:
    - web_deep_agent: For web research (web_search, web_get, web_map, web_crawl)
    - pdf_deep_agent: For RAG research (vector search in documents)
    - community_deep_agent: For community sentiment (reddit, hackernews, lemmy, 4chan)

    PARALLEL EXECUTION:
    - ALL research angles are executed in parallel using asyncio.gather
    - Within each angle, web/pdf/community agents also run in parallel
    - Results are collected and combined before evaluation

    Best Practices:
    - Emits streaming events for real-time updates
    - Parallel execution for speed
    """
    import asyncio

    research_plan = state["research_plan"]
    remaining_steps = state.get("remaining_steps", 100)
    yaml_config = state.get("yaml_config", {})
    session_id = state.get("session_id")
    documents = state.get("documents", [])

    if not research_plan:
        # No angles to explore
        return {"next_action": "evaluate"}

    # Create a unique explorer ID for this exploration
    explorer_id = f"flash_explorer_{state.get('research_avenue', 'unknown')[:20].replace(' ', '_')}"

    logger.info(f"[EXPLORER] Executing {len(research_plan)} research angles in PARALLEL")

    # Create tasks for ALL angles - execute in parallel
    angle_tasks = [
        _execute_angle_research(
            query=angle,
            angle_index=idx,
            explorer_id=explorer_id,
            yaml_config=yaml_config,
            session_id=session_id,
            documents=documents,
            config=config,
            remaining_steps=remaining_steps,
            store=state.get("store"),  # Pass LangGraph Store for memory persistence
        )
        for idx, angle in enumerate(research_plan)
    ]

    # Execute ALL angles in parallel
    angle_results = await asyncio.gather(*angle_tasks, return_exceptions=True)

    # Process results, handling any exceptions
    deep_agent_results = state.get("deep_agent_results", [])
    for result in angle_results:
        if isinstance(result, Exception):
            logger.error(f"[EXPLORER] Angle task failed with exception: {result}")
            deep_agent_results.append({
                "angle": "unknown",
                "angle_index": -1,
                "error": str(result),
                "completed_at": datetime.now().isoformat()
            })
        else:
            deep_agent_results.append(result)

    logger.info(f"[EXPLORER] All {len(research_plan)} angles completed, proceeding to evaluation")

    return {
        "deep_agent_results": deep_agent_results,
        "current_angle": len(research_plan),  # Mark all angles as completed
        "next_action": "evaluate",
        "remaining_steps": max(0, remaining_steps - len(research_plan)),  # Decrement by number of angles
    }


async def evaluate_node(state: ExplorerState, config) -> Dict[str, Any]:
    """Evaluate findings and decide if we have sufficient information.

    This is the INTERNAL CRITIC - it checks:
    - Do we have enough sources?
    - Are the sources recent and credible?
    - Do we have different perspectives?
    - Have we met the sufficiency criteria?

    If not enough info, go back to planning for more angles.

    CRITIC MESSAGE TRACKING:
    - Uses run_logger to track full critic evaluation details
    - Logs evaluation prompt, response, and decision reasoning
    - Provides visibility into critic's quality assessment

    Best Practices:
    - Uses RemainingSteps for proactive recursion handling (graceful degradation)
    - Uses "Draft then Extract" pattern for structured output
    - Emits streaming events for real-time updates
    """
    from ..graph.helpers.llm import create_llm_from_config
    from ..graph.helpers.deep_agent_helpers import extract_structured_output, get_llm_response_content
    from pydantic import BaseModel, Field, field_validator
    from langchain_core.runnables import RunnableConfig

    # Simple evaluation model
    class SimpleEvaluation(BaseModel):
        """Simple evaluation model with 0-100 scale."""
        sufficient: bool = Field(..., description="Whether research is sufficient")
        confidence: int = Field(..., ge=0, le=100, description="Confidence level (0-100)")
        gaps: List[str] = Field(default_factory=list, description="What's missing")
        quality_score: int = Field(..., ge=0, le=100, description="Quality score (0-100)")

        @field_validator('confidence', 'quality_score', mode='before')
        @classmethod
        def normalize_score(cls, v):
            """Normalize scores to 0-100 range using failsafe rules.

            FAILSAFE RULES (same as common.normalize_confidence):
            - If value <= 1, multiply by 100 (0-1 scale -> 0-100)
            - If value is 2-10, multiply by 10 (1-10 scale -> 0-100)
            - If value > 10, assume 0-100 scale (clamp if needed)
            """
            if isinstance(v, (int, float)):
                val = float(v)
                # Handle negative
                if val < 0:
                    return 0
                # FAILSAFE 1: <= 1 means 0-1 scale, multiply by 100
                if val <= 1:
                    return min(100, int(round(val * 100)))
                # FAILSAFE 2: 2-10 means 1-10 scale, multiply by 10
                if 2 <= val <= 10:
                    return int(round(val * 10))
                # FAILSAFE 3: > 10 means 0-100 scale, clamp
                if val > 100:
                    return 100
                return int(round(val))
            return 50  # Default

    research_avenue = state["research_avenue"]
    research_plan = state["research_plan"]
    deep_agent_results = state.get("deep_agent_results", [])
    remaining_steps = state.get("remaining_steps", 100)
    yaml_config = state.get("yaml_config", {})
    run_logger = state.get("run_logger")  # Get run_logger for critic message tracking

    # Get best practices config
    bp_config = _get_best_practice_config(yaml_config)
    low_steps_threshold = bp_config["low_steps_threshold"]

    # Count successful results - raw_sources is a dict {source_id: source_data}
    successful_results = [r for r in deep_agent_results if "error" not in r]
    total_sources = sum(len(r.get("raw_sources", {})) for r in successful_results)

    # Best Practices: Proactive recursion handling using RemainingSteps
    # Force synthesis when approaching limit (graceful degradation)
    if remaining_steps <= low_steps_threshold:
        logger.warning(
            f"[EXPLORER] Low remaining steps ({remaining_steps}), forcing synthesis. "
            f"Found {len(successful_results)} results, {total_sources} sources."
        )
        return {
            "evaluation": {
                "sufficient": True,
                "confidence": 0.5,
                "gaps": ["Approaching recursion limit - synthesizing with available data"],
                "quality_score": 0.5,
                "forced": True
            },
            "sufficient_info": True,
            "next_action": "synthesize"
        }

    # Use Flash model for quick evaluation
    yaml_config = state.get("yaml_config", {})
    flash_llm = await create_llm_from_config(yaml_config, role="worker")

    # Build findings summary for evaluation - use findings_full for better context
    # Context management: Include enough content for proper quality evaluation
    # Raw sources is a dict {source_id: source_data}, so len() counts sources correctly
    findings_summary = "\n\n".join([
        f"Angle: {r['angle']}\nSources: {len(r.get('raw_sources', {}))}\n{r.get('findings_full', r.get('findings', ''))[:2000]}"
        for r in successful_results
    ])

    # Get prompt from config with Jinja2 rendering
    from ..graph.helpers.prompts import get_prompt_from_config

    # Step 1: Deep agent runs freely (no JSON constraints)
    prompt = get_prompt_from_config(
        yaml_config,
        "flash_explorer.evaluation",
        research_avenue=research_avenue,
        searches_performed=len(research_plan),
        successful_agents=len(successful_results),
        total_agents=len(deep_agent_results),
        total_sources=total_sources,
        findings_summary=findings_summary
    )

    try:
        # CRITIC MESSAGE TRACKING: Log evaluation prompt and context
        if run_logger:
            run_logger._write_to_file(f"\n{'='*80}")
            run_logger._write_to_file(f"[CRITIC] INTERNAL CRITIC EVALUATION - FLASH EXPLORER")
            run_logger._write_to_file(f"{'='*80}")
            run_logger._write_to_file(f"[CRITIC] Research Avenue: {research_avenue}")
            run_logger._write_to_file(f"[CRITIC] Searches Performed: {len(research_plan)}")
            run_logger._write_to_file(f"[CRITIC] Successful Deep Agents: {len(successful_results)}/{len(deep_agent_results)}")
            run_logger._write_to_file(f"[CRITIC] Total Sources Found: {total_sources}")
            run_logger._write_to_file(f"[CRITIC] Remaining Steps: {remaining_steps}")
            run_logger._write_to_file(f"\n[CRITIC] FINDINGS SUMMARY:")
            run_logger._write_to_file(findings_summary[:2000])  # Log findings summary

        # Step 2: Get free-form response (with token limit enforcement)
        safe_prompt = _enforce_token_limit(prompt, max_chars=24000, run_logger=run_logger)
        response = await flash_llm.ainvoke(safe_prompt)
        agent_output = get_llm_response_content(response, run_logger=run_logger)

        # Step 3: Extract structured output using extract_structured_output
        evaluation = await extract_structured_output(
            llm=flash_llm,
            agent_output=agent_output,
            pydantic_model=SimpleEvaluation,
            role="research_evaluator",
            yaml_config=yaml_config,
            run_logger=run_logger
        )

        # CRITIC MESSAGE TRACKING: Log evaluation results
        if run_logger:
            run_logger._write_to_file(f"\n[CRITIC] EVALUATION RESULT:")
            run_logger._write_to_file(f"[CRITIC] - Sufficient: {evaluation.sufficient}")
            run_logger._write_to_file(f"[CRITIC] - Confidence: {evaluation.confidence:.2f}")
            run_logger._write_to_file(f"[CRITIC] - Quality Score: {evaluation.quality_score:.2f}")
            if evaluation.gaps:
                run_logger._write_to_file(f"[CRITIC] - Identified Gaps:")
                for gap in evaluation.gaps:
                    run_logger._write_to_file(f"[CRITIC]   * {gap}")
            else:
                run_logger._write_to_file(f"[CRITIC] - No gaps identified (research is complete)")
            run_logger._write_to_file(f"{'='*80}\n")

        logger.info(
            f"[EXPLORER] Evaluation: sufficient={evaluation.sufficient}, confidence={evaluation.confidence}, "
            f"quality={evaluation.quality_score} (remaining_steps: {remaining_steps})"
        )

        # Emit streaming event
        _emit_event(config, ExplorerEvent.evaluating(
            evaluation.sufficient,
            evaluation.confidence,
            remaining_steps
        ))

        if evaluation.gaps:
            logger.info(f"[EXPLORER] Identified gaps: {evaluation.gaps}")

        # CRITICAL FIX: Decrement remaining_steps when returning to plan
        # This prevents infinite loop by ensuring progress is made
        new_remaining_steps = max(0, remaining_steps - 1) if evaluation.sufficient else remaining_steps

        return {
            "evaluation": {
                "sufficient": evaluation.sufficient,
                "confidence": evaluation.confidence,
                "gaps": evaluation.gaps,
                "quality_score": evaluation.quality_score
            },
            "sufficient_info": evaluation.sufficient,
            "next_action": "synthesize" if evaluation.sufficient else "plan",
            "remaining_steps": new_remaining_steps,  # CRITICAL: Track decrement
        }

    except Exception as e:
        logger.error(f"[EXPLORER] Evaluation failed: {type(e).__name__}: {e}")
        # Fallback: if we have at least 3 successful results, proceed
        sufficient = len(successful_results) >= 3
        _emit_event(config, ExplorerEvent.evaluating(sufficient, 0.5, remaining_steps))
        return {
            "evaluation": {"sufficient": sufficient, "confidence": 0.5},
            "sufficient_info": sufficient,
            "next_action": "synthesize" if sufficient else "plan",
            "remaining_steps": max(0, remaining_steps - 1),  # CRITICAL: Track decrement
        }


async def synthesis_node(state: ExplorerState, config) -> Dict[str, Any]:
    """Synthesize findings into structured output.

    Deep agents save their own findings via write_file("/memories/") tool.
    This node only synthesizes the results.

    Best Practices:
    - Uses "Draft then Extract" pattern for structured output
    - Emits streaming events for real-time updates
    """
    import asyncio
    from ..graph.helpers.llm import create_llm_from_config
    from ..graph.helpers.deep_agent_helpers import extract_structured_output, get_llm_response_content
    from ..graph.state_models import FlashExplorerFindings
    from pydantic import BaseModel, Field
    from langchain_core.runnables import RunnableConfig

    # Simple synthesis model for extraction
    class SimpleSynthesis(BaseModel):
        """Simple synthesis model with 0-100 scale."""
        summary: str = Field(..., min_length=1, description="Summary of findings")
        key_findings: List[str] = Field(default_factory=list, description="Key findings")
        confidence: int = Field(default=50, ge=0, le=100, description="Confidence (0-100)")
        gaps: List[str] = Field(default_factory=list, description="Identified gaps")

    research_avenue = state["research_avenue"]
    deep_agent_results = state.get("deep_agent_results", [])
    session_id = state.get("session_id")
    yaml_config = state.get("yaml_config", {})
    documents = state.get("documents", [])
    run_logger = state.get("run_logger")

    # Filter successful results
    successful_results = [r for r in deep_agent_results if "error" not in r]

    # Calculate loops completed based on remaining steps vs initial
    remaining_steps = state.get("remaining_steps", 100)
    loops_completed = max(1, (100 - remaining_steps) // 10)  # Rough estimate

    # Emit streaming event
    _emit_event(config, ExplorerEvent.synthesizing(research_avenue, loops_completed))

    # Use Flash model for synthesis
    flash_llm = await create_llm_from_config(yaml_config, role="worker")

    # Build combined findings for synthesis prompt
    combined_findings = "\n\n".join([
        "=== FROM ANGLE: {} ===\n{}".format(
            r['angle'],
            r.get('findings_full', r.get('findings', ''))[:2000]
        ) for r in successful_results
    ])

    # Build document context if RAG was used
    doc_context = ""
    if documents:
        doc_context = f"\n\nDOCUMENTS ANALYZED:\n{chr(10).join(f'- {doc}' for doc in documents)}"

    # Get prompt from config with Jinja2 rendering
    from ..graph.helpers.prompts import get_prompt_from_config

    # Build findings summary for the synthesis template
    # The template expects num_explorers and findings_summary
    # Raw sources is a dict {source_id: source_data}
    findings_summary = "\n\n".join([
        f"=== FROM ANGLE: {r['angle']} ===\n"
        f"Sources: {len(r.get('raw_sources', {}))}\n"
        f"{r.get('findings_full', r.get('findings', ''))[:2000]}"
        for r in successful_results
    ])

    # Add document context if RAG was used
    if documents:
        doc_context = f"\n\nDOCUMENTS ANALYZED:\n{chr(10).join(f'- {doc}' for doc in documents)}"
        findings_summary += doc_context

    num_explorers = len(successful_results)

    # Step 1: Deep agent runs freely (no JSON constraints)
    synthesis_prompt = get_prompt_from_config(
        yaml_config,
        "flash_explorer.synthesis",
        num_explorers=num_explorers,
        findings_summary=findings_summary,
    )

    # =====================================================================
    # Run synthesis (deep agents save their own findings via write_file)
    # =====================================================================

    # Step 2: Get free-form response (with token limit enforcement)
    safe_synthesis_prompt = _enforce_token_limit(synthesis_prompt, max_chars=24000, run_logger=run_logger)
    response = await flash_llm.ainvoke(safe_synthesis_prompt)
    agent_output = get_llm_response_content(response, run_logger=run_logger)

    # Step 3: Extract structured output using extract_structured_output
    synthesis = await extract_structured_output(
        llm=flash_llm,
        agent_output=agent_output,
        pydantic_model=SimpleSynthesis,
        role="research_synthesizer",
        yaml_config=yaml_config,
        run_logger=run_logger
    )

    # =====================================================================
    # Process results
    # =====================================================================

    # Collect all unique sources from deep agents - raw_sources is a dict {source_id: source_data}
    all_sources = []
    for r in successful_results:
        raw_sources = r.get("raw_sources", {})
        if isinstance(raw_sources, dict):
            all_sources.extend(raw_sources.keys())
        elif isinstance(raw_sources, list):
            all_sources.extend(raw_sources)

    # Dedupe and limit sources
    synthesis_sources = list(set(all_sources))[:20]

    # Create final FlashExplorerFindings model for output
    findings = FlashExplorerFindings(
        avenue=research_avenue,
        summary=synthesis.summary,
        key_findings=synthesis.key_findings,
        sources=synthesis_sources,
        confidence=synthesis.confidence,
        gaps=synthesis.gaps,
        deep_agents_run=len(deep_agent_results),
        successful_agents=len(successful_results),
        loops_completed=loops_completed,
        rag_enabled=len(documents) > 0,
        documents_analyzed=len(documents),
    )

    structured_findings = findings.model_dump()

    logger.info(
        f"[EXPLORER] Synthesis complete for: {research_avenue} "
        f"({loops_completed} loops, {len(successful_results)}/{len(deep_agent_results)} agents succeeded)"
    )

    # Emit completion event
    _emit_event(config, ExplorerEvent.complete(structured_findings))

    return {
        "structured_findings": structured_findings,
        "next_action": "done"
    }


def _emit_event(config, event: Dict[str, Any]) -> None:
    """Emit a streaming event if config supports it.

    Args:
        config: LangGraph RunnableConfig (may contain stream_writer)
        event: Event dictionary to emit
    """
    try:
        if config and "configurable" in config:
            stream_writer = config["configurable"].get("__stream_writer__")
            if stream_writer:
                # Write event to stream
                import asyncio
                if asyncio.iscoroutinefunction(stream_writer):
                    asyncio.create_task(stream_writer(event))
                else:
                    stream_writer(event)
    except Exception as e:
        # Don't let streaming errors break the main flow
        logger.debug(f"[EXPLORER] Failed to emit event: {e}")


# =============================================================================
# GRAPH WITH CHECKPOINTING AND STREAMING
# =============================================================================

def create_flash_explorer_graph(checkpointer=None):
    """Create the Flash Explorer agent graph with checkpointing and streaming.

    Args:
        checkpointer: Optional LangGraph checkpointer for observability and resume

    This is a LangGraph that implements autonomous research:
    1. Plan: Figure out how to investigate the avenue
    2. Search: Execute deep agents for ALL angles IN PARALLEL (DEEP AGENTS ALL THE WAY DOWN)
    3. Evaluate: Internal critic checks if we have enough info
    4. Synthesis: Combine findings into structured output
    5. Loop back to planning if evaluation says insufficient (with step tracking protection)

    PARALLEL EXECUTION:
    - All research angles are executed in parallel using asyncio.gather
    - Within each angle, web/pdf/community deep agents also run in parallel
    - This provides maximum speed for research execution

    Best Practices Implemented:
    - Proactive Recursion Handling: Manual step tracking with configurable threshold
    - Structured Output: Pydantic validation with with_structured_output()
    - Input/Output Schemas: Clean API boundaries
    - Streaming: Real-time progress updates via events
    - Checkpointing: Full LangGraph checkpointing support
    - Parallel Execution: All angles and deep agents run in parallel
    """
    from langgraph.graph import StateGraph, END

    # Use ExplorerState directly (remaining_steps is tracked as a regular int)
    builder = StateGraph(ExplorerState)

    # DIAGNOSTIC: Wrap nodes to log entry/exit
    async def diagnostic_planning_node(state, config):
        avenue = state.get("research_avenue", "unknown")[:30]
        logger.info(f"[FLASH_GRAPH] >>> ENTER planning_node for: {avenue}")
        result = await planning_node(state, config)
        logger.info(f"[FLASH_GRAPH] <<< EXIT planning_node: next_action={result.get('next_action')}, remaining_steps={result.get('remaining_steps')}")
        return result

    async def diagnostic_search_node(state, config):
        avenue = state.get("research_avenue", "unknown")[:30]
        logger.info(f"[FLASH_GRAPH] >>> ENTER search_node for: {avenue}")
        result = await search_node(state, config)
        logger.info(f"[FLASH_GRAPH] <<< EXIT search_node: next_action={result.get('next_action')}, results_count={len(state.get('research_plan', []))}")
        return result

    async def diagnostic_evaluate_node(state, config):
        avenue = state.get("research_avenue", "unknown")[:30]
        logger.info(f"[FLASH_GRAPH] >>> ENTER evaluate_node for: {avenue}")
        result = await evaluate_node(state, config)
        logger.info(f"[FLASH_GRAPH] <<< EXIT evaluate_node: next_action={result.get('next_action')}, sufficient={result.get('sufficient_info')}")
        return result

    async def diagnostic_synthesis_node(state, config):
        avenue = state.get("research_avenue", "unknown")[:30]
        logger.info(f"[FLASH_GRAPH] >>> ENTER synthesis_node for: {avenue}")
        result = await synthesis_node(state, config)
        logger.info(f"[FLASH_GRAPH] <<< EXIT synthesis_node: has_structured_findings={'structured_findings' in result}")
        return result

    # Add nodes (wrapped for diagnostics)
    builder.add_node("plan", diagnostic_planning_node)
    builder.add_node("search", diagnostic_search_node)
    builder.add_node("evaluate", diagnostic_evaluate_node)
    builder.add_node("synthesize", diagnostic_synthesis_node)

    # Define edges
    builder.set_entry_point("plan")

    # Plan → Search (search now processes ALL angles in parallel)
    builder.add_edge("plan", "search")

    # Search → Evaluate (all angles done in one parallel batch)
    builder.add_edge("search", "evaluate")

    # Evaluate → loop back to plan for more angles, or go to synthesis
    def route_after_evaluate(state: ExplorerState) -> str:
        if state.get("sufficient_info", False):
            return "synthesize"
        return "plan"

    builder.add_conditional_edges("evaluate", route_after_evaluate, {
        "plan": "plan",
        "synthesize": "synthesize"
    })

    # Synthesis → END
    builder.add_edge("synthesize", END)

    # Add checkpointer if provided
    if checkpointer:
        logger.info("[EXPLORER] Graph compiled with checkpointer for observability")
        return builder.compile(checkpointer=checkpointer)
    else:
        return builder.compile()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def run_flash_explorer(
    research_avenue: str,
    session_id: str,
    context_engine: Any = None,
    yaml_config: Dict[str, Any] = None,
    documents: List[str] = None,
    checkpointer=None,
    recursion_limit: int = 50,  # Max LLM calls per Flash Explorer (configurable via spawner_config.flash_recursion_limit)
    stream_handler=None,
    run_logger: Any = None,  # RunLogger for critic message tracking
    store: Optional[Any] = None,  # LangGraph Store for memory persistence
) -> Dict[str, Any]:
    """Run a Flash Explorer agent for a research avenue.

    Args:
        research_avenue: The broad topic to investigate (e.g., "quantitative employment data")
        session_id: Session ID for Context Engine writes
        context_engine: Context Engine instance for storing findings
        yaml_config: Full config for LLM/tools (passed to deep agents)
        documents: Optional list of file paths to local documents (enables RAG mode)
        checkpointer: Optional LangGraph checkpointer for observability
        recursion_limit: Maximum recursion limit for graph execution
        stream_handler: Optional async function to handle streaming events
        run_logger: Optional RunLogger for critic message tracking
        store: Optional LangGraph Store for memory persistence (passed to web_research_agent)

    Returns:
        Structured findings from the explorer
    """
    # DEBUG: Log store reception at entry point
    logger.info(f"[EXPLORER] run_flash_explorer received store: {type(store).__name__ if store else 'None'}")

    graph = create_flash_explorer_graph(checkpointer=checkpointer)

    # Get best practices config for initial remaining_steps
    bp_config = _get_best_practice_config(yaml_config or {})

    # Build config with streaming support
    config = {}
    if checkpointer:
        config["configurable"] = {
            "thread_id": f"flash_explorer_{session_id}",
            "checkpoint_ns": f"explorer_{research_avenue[:20].replace(' ', '_')}"
        }

    # Add stream handler if provided
    if stream_handler:
        config["configurable"]["__stream_writer__"] = stream_handler

    config["recursion_limit"] = recursion_limit

    logger.info(f"[EXPLORER] Running with recursion_limit={recursion_limit}")
    if checkpointer:
        logger.info(f"[EXPLORER] Running with checkpointing (thread_id: {config['configurable']['thread_id']})")

    logger.info(f"[EXPLORER] Starting research for avenue: {research_avenue}")
    if documents:
        logger.info(f"[EXPLORER] RAG mode enabled with {len(documents)} documents")
    if store:
        logger.info(f"[EXPLORER] LangGraph Store provided for memory persistence")

    result = await graph.ainvoke({
        "research_avenue": research_avenue,
        "session_id": session_id,
        "context_engine": context_engine,
        "yaml_config": yaml_config or {},
        "documents": documents or [],
        "research_plan": [],
        "current_angle": 0,
        "remaining_steps": recursion_limit,  # Initialize with full recursion limit
        "deep_agent_results": [],
        "evaluation": {},
        "sufficient_info": False,
        "structured_findings": {},
        "artifacts_created": [],
        "next_action": "plan",
        "run_logger": run_logger,  # Pass run_logger for critic message tracking
        "store": store,  # Pass LangGraph Store for memory persistence (accessible in nodes)
    }, config=config)

    logger.info(f"[EXPLORER] Completed research for avenue: {research_avenue}")

    # Return both structured findings and artifacts created for validation
    return {
        "structured_findings": result.get("structured_findings", {}),
        "artifacts_created": result.get("artifacts_created", []),
    }
