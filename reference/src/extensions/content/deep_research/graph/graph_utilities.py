"""Graph utility functions for checkpoint management and debugging.

This module contains helper functions for:
- Getting execution history (time-travel debugging)
- Formatting checkpoint summaries
- Getting checkpoints at specific steps
- Resuming execution from checkpoints
- Debugging alternative paths
- Printing execution trees
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add backend to path for absolute imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=5)

from ..logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


def get_execution_history(graph, config: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Get checkpoint history for time-travel debugging.

    Args:
        graph: Compiled LangGraph with checkpointer
        config: RunnableConfig containing thread_id
        limit: Max number of checkpoints to return (most recent first)

    Returns:
        List of checkpoint states with metadata

    Example:
        >>> history = get_execution_history(graph, {"configurable": {"thread_id": "xyz"}})
        >>> for state in history:
        ...     checkpoint_id = state.config["configurable"]["checkpoint_id"]
        ...     next_node = state.next[0] if state.next else "END"
        ...     print(f"{checkpoint_id}: {next_node}")
    """
    try:
        state_history = list(graph.get_state_history(config))
        # Return most recent checkpoints first
        return state_history[:limit]
    except Exception as e:
        logger.error(f"[TIME_TRAVEL] Failed to get execution history: {e}")
        return []


def format_checkpoint_summary(state, index: int) -> str:
    """Format a checkpoint state for human-readable display.

    Args:
        state: Checkpoint state from get_state_history
        index: Checkpoint index (for display)

    Returns:
        Formatted string with checkpoint info
    """
    checkpoint_id = state.config["configurable"].get("checkpoint_id", "unknown")[:12]
    next_node = state.next[0] if state.next else "END"
    timestamp = state.config.get("configurable", {}).get("step", 0)

    return f"  [{index}] Checkpoint {checkpoint_id}... -> {next_node} (step {timestamp})"


def get_checkpoint_at_step(graph, config: Dict[str, Any], step: int) -> Optional[Any]:
    """Get a specific checkpoint by step number.

    Args:
        graph: Compiled LangGraph with checkpointer
        config: RunnableConfig containing thread_id
        step: Step number to retrieve

    Returns:
        Checkpoint state at the given step, or None if not found
    """
    try:
        state_history = list(graph.get_state_history(config))
        for state in state_history:
            if state.config.get("configurable", {}).get("step") == step:
                return state
        return None
    except Exception as e:
        logger.error(f"[TIME_TRAVEL] Failed to get checkpoint at step {step}: {e}")
        return None


def resume_from_checkpoint(graph, checkpoint_config: Dict[str, Any]) -> Dict[str, Any]:
    """Resume execution from a specific checkpoint.

    Args:
        graph: Compiled LangGraph with checkpointer
        checkpoint_config: Config containing thread_id and checkpoint_id

    Returns:
        Final state after resuming execution

    Example:
        >>> # Get a checkpoint from history
        >>> history = get_execution_history(graph, config)
        >>> checkpoint_state = history[2]
        >>> # Resume from that checkpoint
        >>> result = resume_from_checkpoint(graph, checkpoint_state.config)
    """
    try:
        logger.info(f"[TIME_TRAVEL] Resuming from checkpoint: {checkpoint_config.get('checkpoint_id', 'unknown')}")
        return graph.invoke(None, checkpoint_config)
    except Exception as e:
        logger.error(f"[TIME_TRAVEL] Failed to resume from checkpoint: {e}")
        raise


def debug_alternative_path(
    graph,
    checkpoint_config: Dict[str, Any],
    state_update: Dict[str, Any]
) -> Dict[str, Any]:
    """Test alternative paths by modifying state at checkpoint and continuing execution.

    This creates a NEW checkpoint (fork) with the modified state, allowing you to
    explore "what if" scenarios without affecting the original execution.

    Args:
        graph: Compiled LangGraph with checkpointer
        checkpoint_config: Config containing thread_id and checkpoint_id
        state_update: State modifications to apply

    Returns:
        Final state after exploring alternative path

    Example:
        >>> # Say we want to change the topic at checkpoint 2
        >>> history = get_execution_history(graph, config)
        >>> checkpoint_state = history[2]
        >>> # Modify the topic and see what happens
        >>> result = debug_alternative_path(
        ...     graph,
        ...     checkpoint_state.config,
        ...     {"topic": "chickens"}
        ... )
    """
    try:
        # Update state at checkpoint (creates new checkpoint - fork in history)
        new_config = graph.update_state(checkpoint_config, values=state_update)
        logger.info(f"[TIME_TRAVEL] Created alternative path from checkpoint: "
                   f"{new_config['configurable'].get('checkpoint_id', 'unknown')}")

        # Resume execution with modified state
        return graph.invoke(None, new_config)
    except Exception as e:
        logger.error(f"[TIME_TRAVEL] Failed to debug alternative path: {e}")
        raise


def print_execution_tree(graph, config: Dict[str, Any]) -> None:
    """Print a visual tree of the execution history.

    Args:
        graph: Compiled LangGraph with checkpointer
        config: RunnableConfig containing thread_id

    Example:
        >>> print_execution_tree(graph, {"configurable": {"thread_id": "xyz"}})

    Output:
        [0] START -> worker
        [1] worker -> boss
        [2] boss -> lab_architect
        [3] lab_architect -> critic
        [4] critic -> synthesis
        [5] synthesis -> END
    """
    state_history = list(graph.get_state_history(config))
    logger.info(f"[TIME_TRAVEL] Execution Tree for thread_id={config['configurable'].get('thread_id', 'unknown')}")
    logger.info("=" * 60)

    for i, state in enumerate(reversed(state_history)):
        checkpoint_id = state.config["configurable"].get("checkpoint_id", "unknown")[:8]
        next_node = state.next[0] if state.next else "END"
        logger.info(f"[{i}] {checkpoint_id}... -> {next_node}")

    logger.info("=" * 60)
