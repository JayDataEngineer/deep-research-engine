"""Drafter Node - LangGraph node for drafting using subagents.

Spawns flash subagents and specialist sub-sub agents to gather research and write a draft.
"""
import asyncio
import os
from typing import Any, List, Dict
from ..state import EngineState, WorkerResult
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config
from deepagents import create_deep_agent, SubAgent

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

    # 2. Map workers to SubAgent specs
    subagents: List[SubAgent] = []
    
    # Check for Social Archive specialist
    if state.get("workflow_name") == "social_archive" or state.get("context", {}).get("big_text_dump"):
        subagents.append({
            "name": "social_specialist",
            "description": "Expert at analyzing social archive data dumps to extract relationship profiles and key memories.",
            "system_prompt": "You are a Social Archive Flash Agent. Your job is to read the 'big_text_dump.md' located in the output directory and synthesize a comprehensive report of friend profiles and vibes. Use file tools to find and read the data.",
            "model": "openai:gpt-4o-mini" # Fast flash model
        })

    # Add other workers from config
    config_workers = phase_config.get("workers", [])
    if isinstance(config_workers, list):
        for w in config_workers:
            if not w.get("enabled", True):
                continue
            
            # Convert our generic worker config to deepagents SubAgent spec
            subagents.append({
                "name": w.get("type", "generic_researcher"),
                "description": f"Specialist worker for {w.get('type')}. Task: {w.get('task')}",
                "system_prompt": f"You are a specialist in {w.get('type')}. Your goal is: {w.get('task')}. Be thorough but concise.",
                "model": "openai:gpt-4o-mini"
            })

    # 3. Create the King Drafter
    king_prompt = (
        "You are the King Drafter. Your goal is to oversee a complex research task.\n"
        "You don't do the dirty work yourself. You delegate specific research tasks to your Flash Agents.\n"
        "Once you have gathered all the necessary context from them, you write the final comprehensive draft.\n\n"
        f"Current Goal: {state.get('input_data', {}).get('query', 'Perform comprehensive research and analysis.')}\n"
    )
    
    # Note: LiteLLM URL should be set via environment variable LITELLM_API_BASE
    
    king = create_deep_agent(
        model="openai:gpt-4o", # Smart King model
        system_prompt=king_prompt,
        subagents=subagents,
        debug=True
    )

    # 4. Invoke the King
    logger.info(f"Invoking King Drafter with {len(subagents)} subagents...")
    
    # We use a simple message for the initial invocation
    result = await king.ainvoke({"messages": [("user", "Start the research process.")]})
    
    # Extract the final response from the King
    final_message = result["messages"][-1].content
    
    # Create result for accumulation
    king_result = {
        "success": True,
        "output": final_message,
        "output_type": "markdown",
        "worker_id": "king_drafter"
    }

    logger.info("King Drafter has completed the orchestration.")
    return {
        "worker_results": [king_result],
        "__goto__": "critic"
    }

