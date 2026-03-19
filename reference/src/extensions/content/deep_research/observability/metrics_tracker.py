"""Metrics tracker for research runs.

.. deprecated:: 2.0.0
    Use LangFuse for metrics tracking instead. See:
    src/extensions/integrations/langfuse/

    LangFuse automatically tracks:
    - LLM call tokens and timing
    - Tool call duration
    - Custom scores for quality gates

    Migration:
    - MetricsTracker.track_call() -> LangFuse Generation (automatic via callback)
    - MetricsTracker.track_tool_call() -> LangFuse Span (automatic via callback)
    - MetricsTracker.add_critique() -> score_research_quality()

This module provides metrics tracking for:
- LLM calls (tokens, timing, per-phase breakdown)
- Tool usage (calls, duration, output size)
- Worker performance (per-worker metrics)
- Quality scores and critique history

Usage:
    from .metrics_tracker import MetricsTracker

    tracker = MetricsTracker(run_id="my_run", use_case="rag", variant="default")
    tracker.start_run()
    tracker.track_call("worker", prompt_tokens=100, completion_tokens=200, time_seconds=2.5)
    tracker.track_tool_call("web_search", duration=1.5, output_size=5000)
    tracker.end_run()
    metrics = tracker.metrics
"""

import warnings

warnings.warn(
    "MetricsTracker is deprecated. Use LangFuse for metrics tracking instead. "
    "See: src/extensions/integrations/langfuse/",
    DeprecationWarning,
    stacklevel=2,
)

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from .models import WorkerMetrics, RunMetrics

# Get logger - safe_get_logger always succeeds
from ..logging_utils import safe_get_logger
logger = safe_get_logger(__name__, "metrics_tracker")


class MetricsTracker:
    """Tracks metrics for a research run.

    Features:
    - Per-phase LLM call tracking
    - Tool usage statistics
    - Worker performance metrics
    - Automatic performance calculation
    - Quality score tracking
    """

    def __init__(
        self,
        session_id: str,
        use_case: str = "rag",
        variant: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the metrics tracker.

        Args:
            session_id: Unique identifier for this session (links all artifacts)
            use_case: Use case type (rag, soc_media_gen, etc.)
            variant: Configuration variant
            config: Optional configuration dictionary
        """
        config = config or {}
        self.metrics = RunMetrics(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            use_case=use_case,
            variant=variant,
            config=config,
        )
        self._worker_start_times: Dict[str, float] = {}
        self._run_start_time: float = 0.0
        self._tool_start_times: Dict[str, float] = {}

    def start_run(self) -> None:
        """Mark the start of the run."""
        self._run_start_time = time.time()
        self.metrics.start_time = self._run_start_time

    def end_run(self) -> None:
        """Mark the end of the run and calculate final metrics."""
        self.metrics.end_time = time.time()
        self.metrics.total_time = self.metrics.end_time - self.metrics.start_time

        # Calculate performance metrics
        total_tokens = self.metrics.total_prompt_tokens + self.metrics.total_completion_tokens
        if self.metrics.total_time > 0 and total_tokens > 0:
            self.metrics.tokens_per_sec = total_tokens / self.metrics.total_time

        total_calls = (
            self.metrics.worker_calls +
            self.metrics.boss_calls +
            self.metrics.worker_calls +
            self.metrics.critic_calls +
            self.metrics.synthesis_calls
        )
        if total_calls > 0:
            self.metrics.avg_call_time = self.metrics.total_time / total_calls

        # Calculate worker metrics
        for w in self.metrics.worker_metrics:
            w.calculate_performance_metrics()

    def track_call(
        self,
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        time_seconds: float,
        tool_calls: int = 0,
        worker_id: Optional[str] = None,
    ) -> None:
        """Track an LLM call.

        Args:
            phase: Phase name (worker, boss, worker/lab, critic, synthesis)
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            time_seconds: Duration of the call
            tool_calls: Number of tool calls made
            worker_id: Optional worker identifier
        """
        self.metrics.calls.append({
            "phase": phase,
            "worker_id": worker_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "time_seconds": time_seconds,
            "tool_calls": tool_calls,
            "timestamp": datetime.now().isoformat()
        })

        # Update totals
        self.metrics.total_prompt_tokens += prompt_tokens
        self.metrics.total_completion_tokens += completion_tokens

        # Update phase-specific metrics
        phase_prefix = phase.lower()
        if phase_prefix == "worker":
            self.metrics.worker_calls += 1
            self.metrics.worker_time += time_seconds
            self.metrics.worker_prompt_tokens += prompt_tokens
            self.metrics.worker_completion_tokens += completion_tokens
        elif phase_prefix == "boss":
            self.metrics.boss_calls += 1
            self.metrics.boss_time += time_seconds
            self.metrics.boss_prompt_tokens += prompt_tokens
            self.metrics.boss_completion_tokens += completion_tokens
        elif phase_prefix in ("worker", "lab", "researcher"):
            self.metrics.worker_calls += 1
            self.metrics.lab_time += time_seconds
            self.metrics.worker_prompt_tokens += prompt_tokens
            self.metrics.worker_completion_tokens += completion_tokens
        elif phase_prefix == "critic":
            self.metrics.critic_calls += 1
            self.metrics.critic_time += time_seconds
            self.metrics.critic_prompt_tokens += prompt_tokens
            self.metrics.critic_completion_tokens += completion_tokens
        elif phase_prefix == "synthesis":
            self.metrics.synthesis_calls += 1
            self.metrics.synthesis_time += time_seconds
            self.metrics.synthesis_prompt_tokens += prompt_tokens
            self.metrics.synthesis_completion_tokens += completion_tokens

    def track_tool_call(
        self,
        tool_name: str,
        duration: float,
        output_size: int,
        worker_id: Optional[str] = None,
    ) -> None:
        """Track a tool call.

        Args:
            tool_name: Name of the tool called
            duration: Duration of the call in seconds
            output_size: Size of the output in bytes/characters
            worker_id: Optional worker who made the call
        """
        self.metrics.tool_calls.append({
            "tool": tool_name,
            "duration": duration,
            "output_size": output_size,
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat()
        })

        # Update tool stats
        if tool_name not in self.metrics.tool_stats:
            self.metrics.tool_stats[tool_name] = {
                "count": 0,
                "total_duration": 0.0,
                "total_output_size": 0,
                "avg_duration": 0.0,
                "avg_output_size": 0.0,
            }

        stats = self.metrics.tool_stats[tool_name]
        stats["count"] += 1
        stats["total_duration"] += duration
        stats["total_output_size"] += output_size
        stats["avg_duration"] = stats["total_duration"] / stats["count"]
        stats["avg_output_size"] = stats["total_output_size"] / stats["count"]

        # Track per-worker tool stats
        if worker_id:
            if "researcher_tool_stats" not in self.metrics.__dict__:
                self.metrics.researcher_tool_stats = {}

            if worker_id not in self.metrics.researcher_tool_stats:
                self.metrics.researcher_tool_stats[worker_id] = {}

            if tool_name not in self.metrics.researcher_tool_stats[worker_id]:
                self.metrics.researcher_tool_stats[worker_id][tool_name] = 0

            self.metrics.researcher_tool_stats[worker_id][tool_name] += 1

    def start_worker(self, worker_id: str) -> None:
        """Mark the start of a worker execution.

        Args:
            worker_id: Unique worker identifier
        """
        self._worker_start_times[worker_id] = time.time()

    def end_worker(
        self,
        worker_id: str,
        worker_type: str,
        task_description: str = "",
        output_length: int = 0,
        source_count: int = 0,
        confidence: float = 0.0,
        status: str = "success",
        error_message: str = "",
    ) -> None:
        """Mark the end of a worker execution and record metrics.

        Args:
            worker_id: Unique worker identifier
            worker_type: Type of worker (generic, deep_agent, etc.)
            task_description: Description of the task
            output_length: Length of the output
            source_count: Number of sources found
            confidence: Confidence score (0-1)
            status: Status (success, error, timeout)
            error_message: Error message if failed
        """
        start_time = self._worker_start_times.pop(worker_id, 0)
        end_time = time.time()
        execution_time = end_time - start_time if start_time > 0 else 0

        worker_metric = WorkerMetrics(
            worker_id=worker_id,
            worker_type=worker_type,
            task_description=task_description,
            start_time=start_time,
            end_time=end_time,
            execution_time=execution_time,
            output_length=output_length,
            source_count=source_count,
            confidence=confidence,
            status=status,
            error_message=error_message,
        )

        self.metrics.worker_metrics.append(worker_metric)
        self.metrics.workers_spawned = len(self.metrics.worker_metrics)

    def start_tool(self, tool_name: str) -> None:
        """Mark the start of a tool call.

        Args:
            tool_name: Name of the tool
        """
        self._tool_start_times[tool_name] = time.time()

    def end_tool(self, tool_name: str) -> float:
        """Mark the end of a tool call and return duration.

        Args:
            tool_name: Name of the tool

        Returns:
            Duration in seconds
        """
        start_time = self._tool_start_times.pop(tool_name, 0)
        return time.time() - start_time if start_time > 0 else 0

    def add_quality_score(self, score: float, passed: bool) -> None:
        """Add a quality score from the critic.

        Args:
            score: Quality score (0-1)
            passed: Whether the quality gate passed
        """
        self.metrics.quality_score = score
        self.metrics.passed_quality_gate = passed

    def add_critique(
        self,
        score: float,
        passed: bool,
        feedback: str,
        gaps: List[str] = None,
    ) -> None:
        """Add a critique entry.

        Args:
            score: Quality score (0-1)
            passed: Whether the critique passed
            feedback: Feedback message
            gaps: Optional list of identified gaps
        """
        self.metrics.critique_history.append({
            "score": score,
            "passed": passed,
            "feedback": feedback,
            "gaps": gaps or [],
            "timestamp": datetime.now().isoformat(),
        })
        self.metrics.iterations = len(self.metrics.critique_history)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metrics.

        Returns:
            Dictionary with key metrics
        """
        return {
            "run_id": self.metrics.run_id,
            "total_time": self.metrics.total_time,
            "total_tokens": self.metrics.total_prompt_tokens + self.metrics.total_completion_tokens,
            "tokens_per_sec": self.metrics.tokens_per_sec,
            "calls": {
                "worker": self.metrics.worker_calls,
                "boss": self.metrics.boss_calls,
                "workers": self.metrics.worker_calls,
                "critic": self.metrics.critic_calls,
                "synthesis": self.metrics.synthesis_calls,
                "total": sum([
                    self.metrics.worker_calls,
                    self.metrics.boss_calls,
                    self.metrics.worker_calls,
                    self.metrics.critic_calls,
                    self.metrics.synthesis_calls,
                ]),
            },
            "workers_spawned": self.metrics.workers_spawned,
            "iterations": self.metrics.iterations,
            "quality_score": self.metrics.quality_score,
            "passed_quality_gate": self.metrics.passed_quality_gate,
        }


__all__ = ["MetricsTracker"]
