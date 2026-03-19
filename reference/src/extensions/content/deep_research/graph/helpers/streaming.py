"""Helper for emitting custom SSE events from research nodes.

This module provides utilities for deep research nodes to emit real-time
progress events that are picked up by the frontend for UI updates.

Usage:
    from ..helpers.streaming import get_writer, emit_phase_change

    # In node function (LangGraph passes writer as second positional arg):
    async def my_node(state, writer):
        emit_phase_change(writer, "worker", "Query Analysis", "Analyzing...", "active")

    # Or if writer is passed through config:
    writer = get_writer_from_config(config)
    emit_phase_change(writer, "worker", "Query Analysis", "Analyzing...", "active")

The events are captured by stream_adapter.py and converted to SSE data parts
that the frontend handles in useResearchHandlers.ts.
"""

from typing import Any, Dict, Optional

# Import safe_get_logger for consistent logging
from ...logging_utils import safe_get_logger

logger = safe_get_logger(__name__, "streaming")

# Health thresholds for streaming failures
MAX_EMIT_FAILURES = 10
WARN_EMIT_FAILURES = 3


def get_writer_from_config(config: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Extract the stream writer from LangGraph config.

    LangGraph passes the writer through config["configurable"]["__stream_writer__"]
    when using stream_mode=["custom"].

    Args:
        config: LangGraph RunnableConfig dict

    Returns:
        StreamWriter function or None if not available
    """
    if not config:
        return None

    configurable = config.get("configurable", {})
    if not configurable:
        return None

    return configurable.get("__stream_writer__")


def get_writer(writer: Optional[Any], config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Get the stream writer from either direct arg or config.

    LangGraph can pass the writer either:
    1. As a second positional argument to the node function
    2. Through config["configurable"]["__stream_writer__"]

    This function checks both sources.

    Args:
        writer: Direct writer argument (may be None)
        config: LangGraph RunnableConfig dict (optional)

    Returns:
        StreamWriter function or None if not available
    """
    # First check if writer was passed directly
    if writer is not None:
        return writer

    # Fall back to extracting from config
    return get_writer_from_config(config)


def emit_custom_event(writer: Optional[Any], event_type: str, data: Dict[str, Any]) -> bool:
    """Emit a custom event for SSE streaming.

    Args:
        writer: LangGraph StreamWriter from config (can be None in non-streaming context)
        event_type: Type of event (phase_change, worker_summary, etc.)
        data: Event data payload

    Returns:
        True if emission succeeded, False otherwise
    """
    if writer is None:
        # Not an error - just not in streaming context
        return True

    try:
        writer({"type": event_type, **data})
        return True
    except Exception as e:
        # Track failure in observability context
        try:
            from ...observability.context import increment_stream_failure
            failure_count = increment_stream_failure()
        except ImportError:
            failure_count = 0

        # Log based on severity
        if failure_count >= MAX_EMIT_FAILURES:
            logger.error(
                f"Stream emit critical failure ({failure_count} failures): "
                f"{event_type}: {e}. Consider disabling streaming."
            )
        elif failure_count >= WARN_EMIT_FAILURES:
            logger.warning(
                f"Stream emit repeated failures ({failure_count}): {event_type}: {e}"
            )
        else:
            logger.debug(f"Stream emit failed: {event_type}: {e}")

        return False


def get_streaming_health() -> Dict[str, Any]:
    """Get streaming health status for current context.

    Returns:
        Dict with failure_count, status, recommendation
    """
    try:
        from ...observability.context import get_health_status
        return get_health_status()
    except ImportError:
        return {"status": "unknown", "stream_emit_failures": 0}


def emit_phase_change(
    writer: Optional[Any],
    phase: str,
    phase_name: str,
    message: str,
    status: str = "active"
) -> None:
    """Emit a phase change event.

    Args:
        writer: LangGraph StreamWriter from config
        phase: Phase identifier (worker, boss, lab, critic, synthesis)
        phase_name: Human-readable phase name
        message: Status message
        status: Phase status (active, complete, error)
    """
    emit_custom_event(writer, "phase_change", {
        "phase": phase,
        "phase_name": phase_name,
        "message": message,
        "status": status
    })


def emit_worker_summary(
    writer: Optional[Any],
    complexity: float,
    approach: str,
    entities_found: list,
    research_vectors: list
) -> None:
    """Emit a worker phase summary event.

    Args:
        writer: LangGraph StreamWriter from config
        complexity: Complexity estimate (0-1)
        approach: Recommended approach
        entities_found: List of key entities
        research_vectors: List of research vectors
    """
    emit_custom_event(writer, "worker_summary", {
        "data": {
            "complexity": complexity,
            "approach": approach,
            "entities_found": entities_found,
            "research_vectors": research_vectors
        }
    })


def emit_boss_summary(
    writer: Optional[Any],
    decision: str,
    num_workers: int,
    worker_types: list,
    reasoning: str = ""
) -> None:
    """Emit a boss phase summary event.

    Args:
        writer: LangGraph StreamWriter from config
        decision: Boss decision (research, direct, etc.)
        num_workers: Number of workers to spawn
        worker_types: List of worker types
        reasoning: Reasoning for the decision
    """
    emit_custom_event(writer, "boss_summary", {
        "data": {
            "decision": decision,
            "num_workers": num_workers,
            "worker_types": worker_types,
            "reasoning": reasoning
        }
    })


def emit_worker_task_start(
    writer: Optional[Any],
    worker_id: str,
    task: str,
    description: str = ""
) -> None:
    """Emit a worker task start event.

    Args:
        writer: LangGraph StreamWriter from config
        worker_id: Unique worker identifier
        task: Task description
        description: Additional description
    """
    emit_custom_event(writer, "worker_task_start", {
        "data": {
            "worker_id": worker_id,
            "task": task,
            "description": description
        }
    })


def emit_worker_task_complete(
    writer: Optional[Any],
    worker_id: str,
    sources: Optional[list] = None,
    confidence: Optional[float] = None,
    summary: str = ""
) -> None:
    """Emit a worker task complete event.

    Args:
        writer: LangGraph StreamWriter from config
        worker_id: Unique worker identifier
        sources: List of sources found
        confidence: Confidence score (0-1)
        summary: Task summary
    """
    emit_custom_event(writer, "worker_task_complete", {
        "data": {
            "worker_id": worker_id,
            "sources": sources or [],
            "confidence": confidence,
            "summary": summary
        }
    })


def emit_lab_summary(
    writer: Optional[Any],
    workers_completed: int,
    worker_summaries: list
) -> None:
    """Emit a lab phase summary event.

    Args:
        writer: LangGraph StreamWriter from config
        workers_completed: Number of workers completed
        worker_summaries: List of worker summary dicts
    """
    emit_custom_event(writer, "lab_summary", {
        "data": {
            "workers_completed": workers_completed,
            "worker_summaries": worker_summaries
        }
    })


def emit_quality_metrics(
    writer: Optional[Any],
    quality: float,
    completeness: float,
    passed: bool,
    feedback: str = "",
    issues: Optional[list] = None,
    missing_info: Optional[list] = None
) -> None:
    """Emit a quality metrics event from critic.

    Args:
        writer: LangGraph StreamWriter from config
        quality: Quality score (0-1)
        completeness: Completeness score (0-1)
        passed: Whether the research passed quality gate
        feedback: Critic feedback
        issues: List of issues identified
        missing_info: List of missing information
    """
    emit_custom_event(writer, "quality_metrics", {
        "data": {
            "quality": quality,
            "completeness": completeness,
            "passed": passed,
            "feedback": feedback,
            "issues": issues or [],
            "missing_info": missing_info or []
        }
    })


def emit_research_complete(
    writer: Optional[Any],
    final_response: str,
    sources: Optional[list] = None,
    citations: Optional[list] = None
) -> None:
    """Emit a research complete event.

    Args:
        writer: LangGraph StreamWriter from config
        final_response: The final synthesized response
        sources: List of sources
        citations: List of citations
    """
    emit_custom_event(writer, "research_complete", {
        "data": {
            "final_response": final_response,
            "sources": sources or [],
            "citations": citations or []
        }
    })
