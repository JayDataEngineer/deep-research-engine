"""Graph routing functions.

This module contains conditional edge functions that determine the flow
of execution through the research graph.

Routes:
- route_from_synthesis: Determines next node after synthesis (boss for iteration, critic for evaluation)
- route_from_critic: Determines next node after critic (boss for iteration, formatter for completion)
"""

from typing import Dict, Any

# Add backend to path for absolute imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=5)

from langgraph.graph import END
from ..logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level

from src.extensions.content.deep_research.graph.state import GlobalState


def route_from_synthesis(state: GlobalState) -> str:
    """Route after synthesis creates draft content.

    Synthesis node uses Command API internally to set current_phase.
    This conditional edge reads that phase and routes accordingly.

    Args:
        state: Current global state

    Returns:
        Next node name ("boss" for iteration, "critic" for evaluation)
    """
    current_phase = state.get("current_phase", "critic")
    if current_phase == "boss":
        logger.debug("route_from_synthesis: -> boss (iteration)")
        return "boss"
    else:
        logger.debug("route_from_synthesis: -> critic (proceed)")
        return "critic"


def route_from_critic(state: GlobalState) -> str:
    """Route after critic evaluates draft.

    Critic node uses Command API internally to set current_phase.
    This conditional edge reads that phase and routes accordingly.

    When critic approves, route to formatter (not directly to END).
    Formatter will then route to END after formatting the output.

    Args:
        state: Current global state

    Returns:
        Next node name ("boss" for iteration, "formatter" for completion)
    """
    current_phase = state.get("current_phase", "end")
    if current_phase == "boss":
        logger.debug("route_from_critic: -> boss (iteration)")
        return "boss"
    else:
        logger.debug("route_from_critic: -> formatter (format output)")
        return "formatter"


def route_from_boss(state: GlobalState) -> str:
    """Route after boss makes decision.

    Boss node uses Command API internally to set boss_decision.
    This conditional edge reads that decision and routes accordingly.

    Args:
        state: Current global state

    Returns:
        Next node name ("lab_architect" for research, "synthesis" for simple queries)
    """
    boss_decision = state.get("boss_decision", "synthesis")
    if boss_decision == "spawn_architects":
        workers = state.get("workers_to_spawn", [])
        if workers:
            logger.debug(f"route_from_boss: -> lab_architect (spawn workers: {workers})")
            return "lab_architect"
        else:
            logger.debug("route_from_boss: -> synthesis (no workers specified)")
            return "synthesis"
    else:
        logger.debug("route_from_boss: -> synthesis (direct synthesis)")
        return "synthesis"
