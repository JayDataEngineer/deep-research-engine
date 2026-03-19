"""Evaluation score helpers for LangFuse.

This module provides utilities for adding evaluation scores to
LangFuse traces, enabling quality gates and performance tracking.

This replaces:
- RunLogger.add_critique (quality gates)
- WorkerMetrics.confidence (worker confidence scores)

Usage:
    from src.extensions.integrations.langfuse import (
        score_research_quality,
        score_worker_confidence,
    )

    # Add quality gate score
    score_research_quality(
        trace_id="trace-123",
        score=0.85,
        passed=True,
        comment="Meets all quality criteria",
    )

    # Add worker confidence score
    score_worker_confidence(
        trace_id="trace-123",
        worker_id="researcher_0",
        confidence=0.92,
    )
"""

from typing import Optional

import structlog

from .client import get_langfuse_client, is_langfuse_enabled

logger = structlog.get_logger(__name__)


def score_research_quality(
    trace_id: str,
    score: float,
    passed: bool,
    comment: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """Add a quality score to a research trace.

    Replaces: RunLogger.add_critique

    This is typically called by the Critic phase to record
    whether the research meets quality gates.

    Args:
        trace_id: LangFuse trace ID
        score: Quality score (0.0 to 1.0)
        passed: Whether the quality gate passed
        comment: Optional comment explaining the score
        session_id: Optional session ID (for logging)

    Returns:
        True if score was added successfully, False otherwise

    Example:
        score_research_quality(
            trace_id=trace.id,
            score=0.75,
            passed=True,
            comment="Good coverage but could use more diverse sources",
        )
    """
    client = get_langfuse_client()
    if client is None:
        return False

    try:
        client.score(
            trace_id=trace_id,
            name="quality_gate",
            value=score,
            comment=comment or f"Passed: {passed}",
            data_type="NUMERIC",
        )

        logger.debug(
            "langfuse_quality_score_added",
            trace_id=trace_id,
            score=score,
            passed=passed,
            session_id=session_id,
        )

        return True

    except Exception as e:
        logger.warning(
            "langfuse_quality_score_failed",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
            session_id=session_id,
        )
        return False


def score_worker_confidence(
    trace_id: str,
    worker_id: str,
    confidence: float,
    comment: Optional[str] = None,
) -> bool:
    """Add a worker confidence score.

    Replaces: WorkerMetrics.confidence

    This tracks how confident a worker/researcher is in their findings.

    Args:
        trace_id: LangFuse trace ID
        worker_id: Worker identifier
        confidence: Confidence score (0.0 to 1.0)
        comment: Optional comment

    Returns:
        True if score was added successfully, False otherwise

    Example:
        score_worker_confidence(
            trace_id=trace.id,
            worker_id="researcher_0",
            confidence=0.88,
            comment="Strong sources, recent data",
        )
    """
    client = get_langfuse_client()
    if client is None:
        return False

    try:
        client.score(
            trace_id=trace_id,
            name=f"worker_confidence_{worker_id}",
            value=confidence,
            comment=comment,
            data_type="NUMERIC",
        )

        logger.debug(
            "langfuse_worker_confidence_added",
            trace_id=trace_id,
            worker_id=worker_id,
            confidence=confidence,
        )

        return True

    except Exception as e:
        logger.warning(
            "langfuse_worker_confidence_failed",
            trace_id=trace_id,
            worker_id=worker_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


def score_phase_completion(
    trace_id: str,
    phase_name: str,
    success: bool,
    duration_seconds: Optional[float] = None,
    comment: Optional[str] = None,
) -> bool:
    """Add a phase completion score.

    This tracks whether a research phase completed successfully
    and optionally its duration.

    Args:
        trace_id: LangFuse trace ID
        phase_name: Name of the phase (boss, worker, critic, etc.)
        success: Whether the phase completed successfully
        duration_seconds: Optional duration in seconds
        comment: Optional comment

    Returns:
        True if score was added successfully, False otherwise

    Example:
        score_phase_completion(
            trace_id=trace.id,
            phase_name="critic",
            success=True,
            duration_seconds=45.2,
            comment="Critic validated research quality",
        )
    """
    client = get_langfuse_client()
    if client is None:
        return False

    try:
        # Use 1.0 for success, 0.0 for failure
        value = 1.0 if success else 0.0

        score_comment = comment or f"Phase {'completed' if success else 'failed'}"
        if duration_seconds is not None:
            score_comment += f" ({duration_seconds:.2f}s)"

        client.score(
            trace_id=trace_id,
            name=f"phase_{phase_name}",
            value=value,
            comment=score_comment,
            data_type="NUMERIC",
        )

        logger.debug(
            "langfuse_phase_score_added",
            trace_id=trace_id,
            phase_name=phase_name,
            success=success,
            duration_seconds=duration_seconds,
        )

        return True

    except Exception as e:
        logger.warning(
            "langfuse_phase_score_failed",
            trace_id=trace_id,
            phase_name=phase_name,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


def score_source_quality(
    trace_id: str,
    source_count: int,
    high_quality_count: int,
    comment: Optional[str] = None,
) -> bool:
    """Add a source quality score.

    This tracks the quality and quantity of sources used in research.

    Args:
        trace_id: LangFuse trace ID
        source_count: Total number of sources
        high_quality_count: Number of high-quality sources
        comment: Optional comment

    Returns:
        True if score was added successfully, False otherwise
    """
    client = get_langfuse_client()
    if client is None:
        return False

    try:
        # Calculate quality ratio
        quality_ratio = high_quality_count / source_count if source_count > 0 else 0.0

        client.score(
            trace_id=trace_id,
            name="source_quality",
            value=quality_ratio,
            comment=comment or f"{high_quality_count}/{source_count} high-quality sources",
            data_type="NUMERIC",
        )

        logger.debug(
            "langfuse_source_quality_added",
            trace_id=trace_id,
            source_count=source_count,
            high_quality_count=high_quality_count,
            quality_ratio=quality_ratio,
        )

        return True

    except Exception as e:
        logger.warning(
            "langfuse_source_quality_failed",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


def score_token_efficiency(
    trace_id: str,
    total_tokens: int,
    output_length: int,
    comment: Optional[str] = None,
) -> bool:
    """Add a token efficiency score.

    This tracks how efficiently tokens were used (output per token).

    Args:
        trace_id: LangFuse trace ID
        total_tokens: Total tokens consumed
        output_length: Length of the final output
        comment: Optional comment

    Returns:
        True if score was added successfully, False otherwise
    """
    client = get_langfuse_client()
    if client is None:
        return False

    try:
        # Calculate efficiency (output chars per token)
        efficiency = output_length / total_tokens if total_tokens > 0 else 0.0

        client.score(
            trace_id=trace_id,
            name="token_efficiency",
            value=efficiency,
            comment=comment or f"{output_length} chars from {total_tokens} tokens",
            data_type="NUMERIC",
        )

        logger.debug(
            "langfuse_token_efficiency_added",
            trace_id=trace_id,
            total_tokens=total_tokens,
            output_length=output_length,
            efficiency=efficiency,
        )

        return True

    except Exception as e:
        logger.warning(
            "langfuse_token_efficiency_failed",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False
