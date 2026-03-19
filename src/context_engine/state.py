"""Minimal state for Context Engine.

Most context lives in LangGraph Store (auto-vectorized).
State only tracks orchestration data.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import operator


class ContextEngineState(TypedDict, total=False):
    """Minimal state for Context Engine.

    Most context lives in LangGraph Store (auto-vectorized).
    State only tracks orchestration data.

    Total fields: 6 (not 18+)
    """

    # Messages (standard LangGraph pattern)
    messages: Annotated[List[BaseMessage], operator.add]

    # Session tracking
    session_id: str
    run_id: str

    # Orchestration
    current_phase: str  # "ingest" | "build" | "critic" | "complete"
    iteration_count: int
    max_iterations: int

    # Critic feedback (for loop)
    critic_score: float
    critic_passed: bool
    critic_issues: List[str]
    critic_suggestions: List[str]


def create_initial_state(
    session_id: str,
    run_id: str,
    max_iterations: int = 3,
) -> ContextEngineState:
    """Create initial state for a pipeline run.

    Args:
        session_id: Session ID for tracing
        run_id: Unique run identifier
        max_iterations: Max critic loops

    Returns:
        Initial ContextEngineState
    """
    return {
        "messages": [],
        "session_id": session_id,
        "run_id": run_id,
        "current_phase": "ingest",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "critic_score": 0.0,
        "critic_passed": False,
        "critic_issues": [],
        "critic_suggestions": [],
    }
