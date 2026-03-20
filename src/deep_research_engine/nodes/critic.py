"""Critic Node - Evaluates the draft and decides if it passes.
"""

from typing import Any
from ..state import EngineState
from ..utils.logging import get_loguru_logger

from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config

logger = get_loguru_logger("nodes.critic")

@node_context("critic")
async def critic_node(state: EngineState) -> dict[str, Any]:
    """Critic node for LangGraph workflow."""
    logger.info("Starting critic node")

    # Get config and check if enabled
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "critic")
    critic_enabled = phase_config.get("enabled", True)

    if not critic_enabled:
        logger.info("Critic node disabled. Skipping.")
        return {"__goto__": "neo4j_builder"}

    # (Placeholder for real critic logic)
    score = 0.8
    passed = score >= 0.7
    
    # Count critic iterations (including this one)
    iteration = len([p for p in state.get("phase_history", []) if p == "critic"])
    if not any(p == "critic" for p in state.get("phase_history", [])):
        iteration += 1 # Account for the current one if not already in history
    
    max_retries = phase_config.get("max_retries", 3)
    
    if passed or iteration >= max_retries:
        logger.info(f"Critic passed or max retries ({iteration}/{max_retries}) reached.")
        return {"__goto__": "neo4j_builder", "critic_passed": True}
    else:
        logger.info(f"Critic failed ({iteration}/{max_retries}). Retrying drafter.")
        return {"__goto__": "drafter", "critic_passed": False}

