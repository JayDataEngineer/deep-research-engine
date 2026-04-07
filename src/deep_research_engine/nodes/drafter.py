"""Drafter Node - LangGraph node for drafting using subagents.

Spawns flash subagents and specialist sub-sub agents to gather research and write a draft.
"""

from typing import Any

from deepagents import SubAgent, create_deep_agent

from ..config import get_phase_config
from ..shared import get_langchain_llm
from ..state import EngineState
from ..tools.storage import graph_query, initialize_storage_tools, vector_search_datastore
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..workers.base import WorkerRegistry

logger = get_loguru_logger("nodes.drafter")


@node_context("drafter")
async def drafter_node(state: EngineState) -> dict[str, Any]:
    """Drafter node - The 'King' that orchestrates research via 'Flash' subagents."""
    logger.info("Starting King Drafter node (DeepAgents Orchestration)")

    # 1. Get workflow config and check if enabled
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "drafter")
    if not phase_config:
        # Check "boss" for legacy compatibility
        phase_config = get_phase_config(config_data, "boss")

    node_enabled = phase_config.get("enabled", True)

    if not node_enabled:
        logger.info("Drafter node disabled. Skipping.")
        return {"__goto__": "critic"}

    # Initialize storage tools with current config
    initialize_storage_tools(config_data)

    # 2. Map workers to SubAgent specs
    subagents: list[SubAgent] = []

    # Add workers from config
    config_workers = phase_config.get("workers", [])
    if isinstance(config_workers, list):
        for w_cfg in config_workers:
            if not w_cfg.get("enabled", True):
                continue

            w_type = w_cfg.get("type")
            if not w_type:
                continue

            try:
                # Use WorkerRegistry to get worker class and its metadata
                worker_class = WorkerRegistry.get(w_type)

                # Convert our generic worker config to deepagents SubAgent spec
                subagents.append(
                    {
                        "name": w_type,
                        "description": f"{worker_class.description}. Task: {w_cfg.get('task')}",
                        "system_prompt": (
                            f"You are a specialist in {w_type}. "
                            f"Your goal is: {w_cfg.get('task')}. "
                            f"Be thorough but concise."
                        ),
                        "model": w_cfg.get("model"),
                        "tools": worker_class.required_tools,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to register subagent for worker type '{w_type}': {e}")
                # Fallback to a generic subagent if registry fails
                subagents.append(
                    {
                        "name": w_type,
                        "description": f"Specialist worker for {w_type}. Task: {w_cfg.get('task')}",
                        "system_prompt": f"You are a specialist in {w_type}. Your goal is: {w_cfg.get('task')}. Be thorough but concise.",
                        "model": None,
                    }
                )

    # Add default social archive specialist if applicable
    if (
        state.get("workflow_name") == "social_archive"
        or state.get("context", {}).get("big_text_dump")
    ) and not any(s["name"] == "social_archive" for s in subagents):
        subagents.append(
            {
                "name": "social_archive",
                "description": "Expert at analyzing social archive data dumps to extract relationship profiles and key memories.",
                "system_prompt": "You are a Social Archive Flash Agent. Your job is to analyze the social data to extract friend profiles, relationship vibes, and key memories.",
                "model": "openai:gpt-4o-mini",
            }
        )

    # 3. Create the King Drafter
    king_prompt = (
        "You are the King Drafter, a master orchestrator of research and drafting.\n"
        "Your goal is to oversee a complex research task and produce a high-quality initial draft.\n\n"
        "CORE RESPONSIBILITIES:\n"
        "1. DELEGATE: You don't do the primary research yourself. You delegate specific tasks to your Flash Subagents.\n"
        "2. SYNTHESIZE: Once subagents return their findings, you must integrate them into a coherent, comprehensive report.\n"
        "3. FORMAT: Ensure the output is well-structured markdown with clear headings, citations, and insights.\n\n"
        f"Current Goal: {state.get('input_data', {}).get('query', 'Perform comprehensive research and analysis.')}\n"
    )

    # Get model from config or use local LiteLLM
    king_model_name = phase_config.get("model", None)

    if king_model_name:
        # Use model string directly (deepagents format)
        king_model = king_model_name
    else:
        # Use shared DRY method for LiteLLM-backed ChatOpenAI
        king_model = get_langchain_llm(config_data=config_data)

    king = create_deep_agent(
        model=king_model,
        system_prompt=king_prompt,
        subagents=subagents,
        tools=[vector_search_datastore, graph_query],
        debug=True,
    )

    # 4. Invoke the King
    logger.info(f"Invoking King Drafter with {len(subagents)} subagents...")

    # Build user message from ingestion output or original query
    user_query = state.get("input_data", {}).get("query", "Start the research process.")

    # Check for ingestion output (big_text_dump from telegram/pipeline)
    big_dump_path = state.get("context", {}).get("big_text_dump")
    if big_dump_path:
        from pathlib import Path

        dump_file = Path(big_dump_path)
        if dump_file.exists():
            dump_content = dump_file.read_text()
            logger.info(f"Loaded ingestion output: {len(dump_content)} chars from {dump_file.name}")

            # Try sort+extract sub-agents first (uses DeepAgents pattern)
            try:
                from ..agents.subagents import run_sort_then_extract
                sorted_result = await run_sort_then_extract(
                    dump_content, model=king_model_name if king_model_name else "gemma4-e4b-vision-fast",
                    config_data=config_data,
                )
                logger.info(
                    f"Sort Agent: {len(sorted_result['sorted_content'])} chars, "
                    f"Entity Extractor: {len(sorted_result['entities'])} chars"
                )
                # Write sorted output for reference
                sorted_path = dump_file.parent / "sorted_content.md"
                sorted_path.write_text(sorted_result["sorted_content"])
                entities_path = dump_file.parent / "extracted_entities.md"
                entities_path.write_text(sorted_result["entities"])

                # Use sorted content for the king (it's organized and often shorter)
                analysis_content = sorted_result["sorted_content"]
            except Exception as e:
                logger.warning(f"Sort/Extract sub-agents failed, using raw dump: {e}")
                analysis_content = dump_content

            # Truncate to fit context (32K = ~100K chars for gemma4-e4b)
            max_chars = 90000
            if len(analysis_content) > max_chars:
                logger.info(f"Truncating analysis content from {len(analysis_content)} to {max_chars} chars")
                analysis_content = analysis_content[:max_chars] + "\n\n...[content truncated]..."

            user_message = (
                f"Execute the research plan for: {user_query}\n\n"
                f"## INGESTION DATA (from pipeline)\n\n"
                f"{analysis_content}\n\n"
                f"Use the above data to produce your analysis."
            )
        else:
            logger.warning(f"Ingestion dump file not found: {big_dump_path}")
            user_message = f"Execute the research plan for: {user_query}"
    else:
        user_message = f"Execute the research plan for: {user_query}"

    result = await king.ainvoke({"messages": [("user", user_message)]})

    # Extract the final response from the King
    final_message = result["messages"][-1].content

    # Create result for accumulation
    king_result = {
        "success": True,
        "output": final_message,
        "output_type": "markdown",
        "worker_id": "king_drafter",
    }

    logger.info("King Drafter has completed the orchestration.")
    return {"worker_results": [king_result], "__goto__": "critic"}
