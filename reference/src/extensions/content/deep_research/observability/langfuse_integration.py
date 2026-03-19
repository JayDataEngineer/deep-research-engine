"""LangFuse integration helpers for deep research.

This module provides the bridge between the deep research system and LangFuse
observability. It handles:
- Manual tracking via LangfuseResearchTracker
- Converting session_id to LangFuse trace context
- Integrating diagnostic scores with LangFuse

NOTE: LLM tracing is now automatic via LiteLLM's built-in Langfuse integration.
You do NOT need to create callback handlers - just pass session_id/tags to get_llm().

Usage:
    from .observability.langfuse_integration import LangfuseResearchTracker

    # For manual score/phase tracking (optional)
    tracker = LangfuseResearchTracker(session_id="research-123")
    tracker.start_research(query="What is AI?")
    # ... research execution ...
    tracker.end_research(response="AI is...", success=True)

    # For automatic LLM tracing (recommended):
    from src.shared import get_llm
    llm = await get_llm(
        model="qwen-35-27-fast",
        session_id="research-123",
        tags=["deep_research"]
    )
    # All LLM calls are automatically traced to Langfuse via LiteLLM
"""

import os
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import structlog

from src.extensions.integrations.langfuse.client import (
    get_langfuse_client,
    is_langfuse_enabled,
    reset_langfuse_client,
)

logger = structlog.get_logger(__name__)


def get_langfuse_callback_for_research(
    session_id: str,
    use_case: str = "rag",
    variant: str = "default",
    query: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[Any]:
    """DEPRECATED: Langfuse tracing is now automatic via LiteLLM.

    This function is kept for backward compatibility but no longer returns
    a callback handler. Instead, pass session_id and tags directly to get_llm():

        # OLD (deprecated):
        callback = get_langfuse_callback_for_research(session_id="research-123")
        result = await graph.ainvoke(state, config={"callbacks": [callback]})

        # NEW (recommended):
        from src.shared import get_llm
        llm = await get_llm(
            model="qwen-35-27-fast",
            session_id="research-123",
            tags=["deep_research", "rag"]
        )
        # All calls are automatically traced via LiteLLM

    Args:
        session_id: Research session ID (maps to LangFuse session_id)
        use_case: Research use case (rag, podcast, trading, etc.)
        variant: Config variant
        query: Optional query for trace naming
        user_id: Optional user identifier

    Returns:
        None (tracing is automatic via LiteLLM)
    """
    import warnings
    warnings.warn(
        "get_langfuse_callback_for_research() is deprecated. "
        "Langfuse tracing is now automatic via LiteLLM. "
        "Pass session_id and tags to get_llm() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    logger.debug(
        "langfuse_callback_deprecated",
        session_id=session_id,
        use_case=use_case,
        message="Use get_llm(session_id=..., tags=[...]) instead",
    )

    return None


class LangfuseResearchTracker:
    """Comprehensive LangFuse tracker for deep research runs.

    This class provides a higher-level API for tracking research runs
    with LangFuse, including:
    - Start/end research tracking
    - Phase tracking (boss, critic, synthesis, etc.)
    - Quality score integration
    - Diagnostic score integration

    Example:
        tracker = LangfuseResearchTracker(
            session_id="research-123",
            use_case="rag",
        )

        tracker.start_research(query="What is AI?")

        # Track phases
        with tracker.phase("boss"):
            result = boss_node(state)

        # Add quality scores
        tracker.add_quality_score(
            score=0.85,
            passed=True,
            comment="Good coverage",
        )

        tracker.end_research(
            response="AI is...",
            success=True,
        )
    """

    def __init__(
        self,
        session_id: str,
        use_case: str = "rag",
        variant: str = "default",
    ):
        """Initialize the tracker.

        Args:
            session_id: Research session ID
            use_case: Research use case
            variant: Config variant
        """
        self.session_id = session_id
        self.use_case = use_case
        self.variant = variant
        self._trace = None
        self._client = get_langfuse_client()
        self._phase_spans: Dict[str, Any] = {}

    @property
    def is_enabled(self) -> bool:
        """Check if LangFuse tracking is enabled."""
        return self._client is not None

    def start_research(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking a research run.

        Args:
            query: Research query
            metadata: Optional additional metadata
        """
        if not self.is_enabled:
            return

        self._trace = self._client.trace(
            name=f"research_{self.use_case}",
            session_id=self.session_id,
            input={"query": query},
            metadata={
                "use_case": self.use_case,
                "variant": self.variant,
                **(metadata or {}),
            },
            tags=["deep_research", self.use_case],
        )

        logger.debug(
            "langfuse_research_started",
            session_id=self.session_id,
            trace_id=self._trace.id if self._trace else None,
        )

    def end_research(
        self,
        response: str,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End tracking a research run.

        Args:
            response: Final research response
            success: Whether the research succeeded
            error: Optional error message if failed
            metadata: Optional additional metadata
        """
        if not self.is_enabled or not self._trace:
            return

        self._trace.update(
            output={"response": response, "success": success},
            metadata={
                "error": error,
                **(metadata or {}),
            },
        )

        logger.debug(
            "langfuse_research_ended",
            session_id=self.session_id,
            success=success,
        )

    @contextmanager
    def phase(self, phase_name: str, input_data: Optional[Dict[str, Any]] = None):
        """Context manager for tracking a research phase.

        Args:
            phase_name: Name of the phase (boss, critic, synthesis, etc.)
            input_data: Optional input data for the phase

        Yields:
            The span object (or None if not enabled)

        Example:
            with tracker.phase("boss", {"query": query}) as span:
                result = boss_node(state)
                if span:
                    span.update(output={"goto": result.goto})
        """
        if not self.is_enabled or not self._trace:
            yield None
            return

        span = self._trace.span(
            name=phase_name,
            input=input_data,
        )

        self._phase_spans[phase_name] = span

        try:
            yield span
        except Exception as e:
            span.update(
                level="ERROR",
                status_message=str(e),
                metadata={"error_type": type(e).__name__},
            )
            raise
        finally:
            if phase_name in self._phase_spans:
                del self._phase_spans[phase_name]

    def add_quality_score(
        self,
        score: float,
        passed: bool,
        comment: Optional[str] = None,
    ) -> None:
        """Add a quality gate score to the research trace.

        This integrates with the critic's quality evaluation.

        Args:
            score: Quality score (0.0 to 1.0)
            passed: Whether the quality gate passed
            comment: Optional comment
        """
        if not self.is_enabled or not self._trace:
            return

        self._client.score(
            trace_id=self._trace.id,
            name="quality_gate",
            value=score,
            comment=comment or f"Passed: {passed}",
            data_type="NUMERIC",
        )

    def add_diagnostic_score(
        self,
        diagnostic_type: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """Add a diagnostic score to the research trace.

        This integrates with diagnostic_logging.py for tracking issues.

        Args:
            diagnostic_type: Type of diagnostic (llm_relevance, flash_explorer, etc.)
            value: Score value (typically 0.0 or 1.0)
            comment: Optional comment
        """
        if not self.is_enabled or not self._trace:
            return

        self._client.score(
            trace_id=self._trace.id,
            name=f"diagnostic_{diagnostic_type}",
            value=value,
            comment=comment,
            data_type="NUMERIC",
        )

    def track_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Track an LLM call within the research.

        Args:
            model: Model name
            prompt: Prompt text
            response: Response text
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        if not self.is_enabled or not self._trace:
            return

        self._trace.generation(
            name=f"llm_{model}",
            model=model,
            input=prompt,
            output=response,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    def track_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Track a tool call within the research.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input
            tool_output: Tool output
            duration_seconds: Optional duration
        """
        if not self.is_enabled or not self._trace:
            return

        metadata = {}
        if duration_seconds is not None:
            metadata["duration_seconds"] = duration_seconds

        self._trace.span(
            name=f"tool_{tool_name}",
            input=tool_input,
            output=tool_output,
            metadata=metadata,
        )

    def get_callback(self) -> Optional[Any]:
        """Get a LangFuse callback for graph invocation.

        Returns:
            LangFuse CallbackHandler or None
        """
        return get_langfuse_callback_for_research(
            session_id=self.session_id,
            use_case=self.use_case,
            variant=self.variant,
        )


def integrate_langfuse_with_run_logger(
    run_logger: Any,
    session_id: str,
    use_case: str = "rag",
    variant: str = "default",
) -> Optional[LangfuseResearchTracker]:
    """Integrate LangFuse with an existing RunLogger.

    This creates a LangfuseResearchTracker that can be used alongside
    the legacy RunLogger during the migration period.

    Args:
        run_logger: Existing RunLogger instance
        session_id: Research session ID
        use_case: Research use case
        variant: Config variant

    Returns:
        LangfuseResearchTracker or None if LangFuse is not enabled

    Example:
        # In run_v4_spawner or similar
        langfuse_tracker = integrate_langfuse_with_run_logger(
            run_logger=run_logger,
            session_id=session_id,
            use_case=use_case,
            variant=variant,
        )

        if langfuse_tracker:
            langfuse_tracker.start_research(query=query)
            callback = langfuse_tracker.get_callback()
            # Add callback to graph invocation
    """
    if not is_langfuse_enabled():
        logger.debug(
            "langfuse_integration_skipped",
            session_id=session_id,
            run_id=run_logger.run_id if run_logger else None,
        )
        return None

    tracker = LangfuseResearchTracker(
        session_id=session_id,
        use_case=use_case,
        variant=variant,
    )

    logger.info(
        "langfuse_integrated_with_run_logger",
        session_id=session_id,
        run_id=run_logger.run_id if run_logger else None,
        use_case=use_case,
        variant=variant,
    )

    return tracker
