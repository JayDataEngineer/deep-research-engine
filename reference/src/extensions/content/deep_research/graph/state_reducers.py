"""State Reducers - Reducer functions for LangGraph state management.

This module contains the reducer functions used for merging state updates
in LangGraph. These functions define how to combine state updates from
different nodes.

Separated from state.py to reduce file size and improve modularity.
"""

from typing import List, Any, Dict
from langchain_core.messages import BaseMessage
import operator


# =============================================================================
# MESSAGE REDUCER
# =============================================================================

def merge_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Merge message lists by concatenation.

    This reducer is used for chat_history fields where new messages
    are appended to existing history.

    Args:
        left: First message list
        right: Second message list (typically new messages)

    Returns:
        Concatenated message list
    """
    if not left:
        return right
    if not right:
        return left
    return left + right


# =============================================================================
# WORKER RESULTS REDUCER
# =============================================================================

def merge_worker_results(left: List[Any], right: List[Any]) -> List[Any]:
    """Merge worker results by concatenation.

    This reducer is used for worker_outputs fields where results from
    different workers are accumulated.

    Args:
        left: First worker results list
        right: Second worker results list

    Returns:
        Concatenated worker results list
    """
    if not left:
        return right
    if not right:
        return left
    return left + right


# =============================================================================
# CITATIONS REDUCER
# =============================================================================

def merge_citations(left: List[Any], right: List[Any]) -> List[Any]:
    """Merge citation lists by concatenation.

    This reducer is used for all_citations fields where citations from
    different workers are accumulated.

    Args:
        left: First citations list
        right: Second citations list

    Returns:
        Concatenated citations list
    """
    if not left:
        return right
    if not right:
        return left
    return left + right


# =============================================================================
# OVERWRITE REDUCER
# =============================================================================

def overwrite_reducer(left: Any, right: Any) -> Any:
    """Overwrite reducer - right side always wins.

    This reducer is used for fields that should be completely replaced
    on each update rather than merged.

    Args:
        left: Current value
        right: New value (always wins)

    Returns:
        The new value (right)
    """
    return right


# =============================================================================
# PROMPT ADJUSTMENTS REDUCER
# =============================================================================

def merge_prompt_adjustments(
    left: List[Any],
    right: List[Any]
) -> List[Any]:
    """Merge prompt adjustments by concatenation.

    This reducer accumulates prompt adjustments across iterations.

    Args:
        left: First prompt adjustments list (PromptAdjustment objects)
        right: Second prompt adjustments list (PromptAdjustment objects)

    Returns:
        Concatenated prompt adjustments list

    Note:
        Using List[Any] instead of List[PromptAdjustment] to avoid circular import
        and forward reference issues. The actual type is enforced by GlobalState.
    """
    if not left:
        return right
    if not right:
        return left
    return left + right


# =============================================================================
# DICT MERGE REDUCER
# =============================================================================

def merge_dict_reducer(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dictionaries by combining their key-value pairs.

    This reducer is used for _worker_results_partial field where multiple
    workers write their results to separate keys in the same dict.

    Args:
        left: First dict
        right: Second dict (values from right override left for conflicts)

    Returns:
        Merged dictionary with combined keys from both inputs

    Example:
        left = {"worker_0": result0}
        right = {"worker_1": result1}
        returns = {"worker_0": result0, "worker_1": result1}
    """
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}
