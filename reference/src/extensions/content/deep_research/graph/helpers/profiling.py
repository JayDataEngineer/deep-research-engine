"""Deep profiling system for deep research lab phase.

This module provides detailed timing instrumentation to identify exactly where
time is being spent during worker execution.

Key metrics tracked:
- Total worker time
- LLM call time (per call, aggregated, with/without thinking mode)
- Tool call time (per tool type, aggregated)
- Network latency (web_search, web_get, etc.)
- Token processing time
- Overhead (prompt building, artifact creation, etc.)
"""

import time
import json
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class TimingMetric:
    """A single timing measurement."""
    name: str
    duration_ms: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMMetric(TimingMetric):
    """LLM-specific timing metric."""
    model: str = ""
    temperature: float = 0.0
    thinking_enabled: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    chunks: int = 0
    phase: str = ""  # worker, boss, worker, synthesis, critic


@dataclass
class ToolMetric(TimingMetric):
    """Tool-specific timing metric."""
    tool_name: str = ""
    success: bool = True
    error: str = ""
    network_latency_ms: float = 0.0
    processing_time_ms: float = 0.0


@dataclass
class WorkerProfile:
    """Complete profiling data for a single worker execution."""
    worker_id: str
    worker_type: str
    start_time: float
    end_time: float
    total_duration_ms: float

    # Timing breakdowns
    setup_time_ms: float = 0.0
    prompt_building_time_ms: float = 0.0
    llm_total_time_ms: float = 0.0
    tool_total_time_ms: float = 0.0
    post_processing_time_ms: float = 0.0

    # Detailed metrics
    llm_calls: List[LLMMetric] = field(default_factory=list)
    tool_calls: List[ToolMetric] = field(default_factory=list)

    # Summary stats
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    thinking_mode_calls: int = 0
    total_tokens_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["llm_calls"] = [call.to_dict() for call in self.llm_calls]
        data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return data

    def get_summary(self) -> str:
        """Get a human-readable summary of the profile."""
        lines = [
            f"\n{'='*70}",
            f"PROFILING SUMMARY: {self.worker_id} ({self.worker_type})",
            f"{'='*70}",
            f"Total Duration: {self.total_duration_ms/1000:.2f}s",
            f"",
            f"Timing Breakdown:",
            f"  - Setup: {self.setup_time_ms:.0f}ms ({self.setup_time_ms/self.total_duration_ms*100:.1f}%)",
            f"  - Prompt Building: {self.prompt_building_time_ms:.0f}ms ({self.prompt_building_time_ms/self.total_duration_ms*100:.1f}%)",
            f"  - LLM Calls: {self.llm_total_time_ms:.0f}ms ({self.llm_total_time_ms/self.total_duration_ms*100:.1f}%)",
            f"  - Tool Calls: {self.tool_total_time_ms:.0f}ms ({self.tool_total_time_ms/self.total_duration_ms*100:.1f}%)",
            f"  - Post-Processing: {self.post_processing_time_ms:.0f}ms ({self.post_processing_time_ms/self.total_duration_ms*100:.1f}%)",
            f"",
            f"Call Statistics:",
            f"  - Total LLM Calls: {self.total_llm_calls}",
            f"  - Thinking Mode Calls: {self.thinking_mode_calls}",
            f"  - Total Tool Calls: {self.total_tool_calls}",
            f"  - Total Tokens: {self.total_tokens_processed:,}",
            f"",
        ]

        if self.llm_calls:
            avg_llm_time = sum(c.duration_ms for c in self.llm_calls) / len(self.llm_calls)
            avg_thinking_time = sum(c.duration_ms for c in self.llm_calls if c.thinking_enabled) / max(1, len([c for c in self.llm_calls if c.thinking_enabled]))
            lines.extend([
                f"LLM Call Details:",
                f"  - Avg LLM Call: {avg_llm_time:.0f}ms",
                f"  - Avg Thinking Mode Call: {avg_thinking_time:.0f}ms",
                f"  - Slowest LLM Call: {max(c.duration_ms for c in self.llm_calls):.0f}ms",
                f"  - Fastest LLM Call: {min(c.duration_ms for c in self.llm_calls):.0f}ms",
                f"",
            ])

        if self.tool_calls:
            tool_times = {}
            tool_counts = {}
            for call in self.tool_calls:
                if call.tool_name not in tool_times:
                    tool_times[call.tool_name] = 0
                    tool_counts[call.tool_name] = 0
                tool_times[call.tool_name] += call.duration_ms
                tool_counts[call.tool_name] += 1

            lines.extend([
                f"Tool Call Details:",
            ])
            for tool_name in sorted(tool_times.keys(), key=lambda x: tool_times[x], reverse=True):
                total_time = tool_times[tool_name]
                count = tool_counts[tool_name]
                avg_time = total_time / count
                lines.append(f"  - {tool_name}: {total_time:.0f}ms ({count} calls, avg {avg_time:.0f}ms)")
            lines.append("")

        lines.append(f"{'='*70}")
        return "\n".join(lines)


class Profiler:
    """Main profiler for tracking execution metrics."""

    def __init__(self, worker_id: str, worker_type: str, run_logger=None):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.run_logger = run_logger

        self.start_time = time.time()
        self.current_profile = WorkerProfile(
            worker_id=worker_id,
            worker_type=worker_type,
            start_time=self.start_time,
            end_time=0,
            total_duration_ms=0,
        )

        # Timing stack for nested operations
        self._timing_stack: List[str] = []
        self._section_start: Dict[str, float] = {}

    def record_section(self, section_name: str) -> Callable:
        """Decorator to record timing for a code section."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.time_section(section_name):
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.time_section(section_name):
                    return func(*args, **kwargs)

            # Return appropriate wrapper based on whether function is async
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @contextmanager
    def time_section(self, section_name: str):
        """Context manager to time a code section."""
        start = time.time()
        self._timing_stack.append(section_name)
        self._section_start[section_name] = start

        if self.run_logger:
            self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] >>> {section_name}")

        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self._timing_stack.pop()
            self._section_start.pop(section_name, None)

            if self.run_logger:
                self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] <<< {section_name}: {duration_ms:.0f}ms")

            # Record to appropriate section
            if section_name == "setup":
                self.current_profile.setup_time_ms = duration_ms
            elif section_name == "prompt_building":
                self.current_profile.prompt_building_time_ms = duration_ms
            elif section_name == "post_processing":
                self.current_profile.post_processing_time_ms = duration_ms

    def record_llm_call(
        self,
        model: str,
        temperature: float,
        thinking_enabled: bool,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        chunks: int = 0,
        phase: str = "worker",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator to record LLM call metrics."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                if self.run_logger:
                    self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] >>> LLM call (model={model}, thinking={thinking_enabled})")

                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    metric = LLMMetric(
                        name=f"llm_call_{len(self.current_profile.llm_calls)}",
                        duration_ms=duration_ms,
                        timestamp=start,
                        model=model,
                        temperature=temperature,
                        thinking_enabled=thinking_enabled,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        chunks=chunks,
                        phase=phase,
                        metadata=metadata or {},
                    )
                    self.current_profile.llm_calls.append(metric)
                    self.current_profile.llm_total_time_ms += duration_ms
                    self.current_profile.total_llm_calls += 1
                    if thinking_enabled:
                        self.current_profile.thinking_mode_calls += 1
                    self.current_profile.total_tokens_processed += input_tokens + output_tokens + reasoning_tokens

                    if self.run_logger:
                        self.run_logger._write_to_file(
                            f"[PROFILE {self.worker_id}] <<< LLM call: {duration_ms:.0f}ms "
                            f"(in={input_tokens}, out={output_tokens}, reasoning={reasoning_tokens})"
                        )

                    return result
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    if self.run_logger:
                        self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] <<< LLM call FAILED: {duration_ms:.0f}ms - {e}")
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                if self.run_logger:
                    self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] >>> LLM call (model={model}, thinking={thinking_enabled})")

                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    metric = LLMMetric(
                        name=f"llm_call_{len(self.current_profile.llm_calls)}",
                        duration_ms=duration_ms,
                        timestamp=start,
                        model=model,
                        temperature=temperature,
                        thinking_enabled=thinking_enabled,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        chunks=chunks,
                        phase=phase,
                        metadata=metadata or {},
                    )
                    self.current_profile.llm_calls.append(metric)
                    self.current_profile.llm_total_time_ms += duration_ms
                    self.current_profile.total_llm_calls += 1
                    if thinking_enabled:
                        self.current_profile.thinking_mode_calls += 1
                    self.current_profile.total_tokens_processed += input_tokens + output_tokens + reasoning_tokens

                    if self.run_logger:
                        self.run_logger._write_to_file(
                            f"[PROFILE {self.worker_id}] <<< LLM call: {duration_ms:.0f}ms "
                            f"(in={input_tokens}, out={output_tokens}, reasoning={reasoning_tokens})"
                        )

                    return result
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    if self.run_logger:
                        self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] <<< LLM call FAILED: {duration_ms:.0f}ms - {e}")
                    raise

            # Return appropriate wrapper
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def record_tool_call(
        self,
        tool_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator to record tool call metrics."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                network_latency = 0
                processing_time = 0
                success = True
                error = ""

                if self.run_logger:
                    self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] >>> Tool: {tool_name}")

                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    metric = ToolMetric(
                        name=f"tool_call_{tool_name}_{len(self.current_profile.tool_calls)}",
                        duration_ms=duration_ms,
                        timestamp=start,
                        tool_name=tool_name,
                        success=success,
                        error=error,
                        network_latency_ms=network_latency,
                        processing_time_ms=processing_time,
                        metadata=metadata or {},
                    )
                    self.current_profile.tool_calls.append(metric)
                    self.current_profile.tool_total_time_ms += duration_ms
                    self.current_profile.total_tool_calls += 1

                    if self.run_logger:
                        self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] <<< Tool {tool_name}: {duration_ms:.0f}ms")

                    return result
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    success = False
                    error = str(e)

                    metric = ToolMetric(
                        name=f"tool_call_{tool_name}_{len(self.current_profile.tool_calls)}",
                        duration_ms=duration_ms,
                        timestamp=start,
                        tool_name=tool_name,
                        success=success,
                        error=error,
                        network_latency_ms=network_latency,
                        processing_time_ms=processing_time,
                        metadata=metadata or {},
                    )
                    self.current_profile.tool_calls.append(metric)
                    self.current_profile.tool_total_time_ms += duration_ms
                    self.current_profile.total_tool_calls += 1

                    if self.run_logger:
                        self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] <<< Tool {tool_name} FAILED: {duration_ms:.0f}ms - {error}")

                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                if self.run_logger:
                    self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] >>> Tool: {tool_name}")

                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    metric = ToolMetric(
                        name=f"tool_call_{tool_name}_{len(self.current_profile.tool_calls)}",
                        duration_ms=duration_ms,
                        timestamp=start,
                        tool_name=tool_name,
                        success=success,
                        error=error,
                        network_latency_ms=network_latency,
                        processing_time_ms=processing_time,
                        metadata=metadata or {},
                    )
                    self.current_profile.tool_calls.append(metric)
                    self.current_profile.tool_total_time_ms += duration_ms
                    self.current_profile.total_tool_calls += 1

                    if self.run_logger:
                        self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] <<< Tool {tool_name}: {duration_ms:.0f}ms")

                    return result
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    success = False
                    error = str(e)

                    metric = ToolMetric(
                        name=f"tool_call_{tool_name}_{len(self.current_profile.tool_calls)}",
                        duration_ms=duration_ms,
                        timestamp=start,
                        tool_name=tool_name,
                        success=success,
                        error=error,
                        network_latency_ms=network_latency,
                        processing_time_ms=processing_time,
                        metadata=metadata or {},
                    )
                    self.current_profile.tool_calls.append(metric)
                    self.current_profile.tool_total_time_ms += duration_ms
                    self.current_profile.total_tool_calls += 1

                    if self.run_logger:
                        self.run_logger._write_to_file(f"[PROFILE {self.worker_id}] <<< Tool {tool_name} FAILED: {duration_ms:.0f}ms - {error}")

                    raise

            # Return appropriate wrapper
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def finalize(self) -> WorkerProfile:
        """Finalize the profile and return it."""
        self.current_profile.end_time = time.time()
        self.current_profile.total_duration_ms = (self.current_profile.end_time - self.current_profile.start_time) * 1000

        if self.run_logger:
            self.run_logger._write_to_file(self.current_profile.get_summary())

        return self.current_profile


class LabProfiler:
    """Profiler for the entire lab phase across all workers."""

    def __init__(self, run_logger=None):
        self.run_logger = run_logger
        self.start_time = time.time()
        self.worker_profiles: List[WorkerProfile] = []
        self.active_workers: Dict[str, Profiler] = {}

    def start_worker(self, worker_id: str, worker_type: str) -> Profiler:
        """Start profiling a worker."""
        profiler = Profiler(worker_id, worker_type, self.run_logger)
        self.active_workers[worker_id] = profiler

        if self.run_logger:
            self.run_logger._write_to_file(f"\n[LAB PROFILE] ===== START WORKER: {worker_id} =====\n")

        return profiler

    def end_worker(self, worker_id: str) -> Optional[WorkerProfile]:
        """End profiling for a worker."""
        if worker_id not in self.active_workers:
            return None

        profiler = self.active_workers.pop(worker_id)
        profile = profiler.finalize()
        self.worker_profiles.append(profile)

        if self.run_logger:
            self.run_logger._write_to_file(f"\n[LAB PROFILE] ===== END WORKER: {worker_id} =====\n")
            self.run_logger.save_profiling_json(profile)

        return profile

    def finalize(self) -> Dict[str, Any]:
        """Finalize lab-level profiling and return summary."""
        total_duration_ms = (time.time() - self.start_time) * 1000

        # Aggregate statistics
        total_llm_time = sum(p.llm_total_time_ms for p in self.worker_profiles)
        total_tool_time = sum(p.tool_total_time_ms for p in self.worker_profiles)
        total_llm_calls = sum(p.total_llm_calls for p in self.worker_profiles)
        total_tool_calls = sum(p.total_tool_calls for p in self.worker_profiles)
        thinking_calls = sum(p.thinking_mode_calls for p in self.worker_profiles)

        summary = {
            "lab_phase": {
                "total_duration_ms": total_duration_ms,
                "total_duration_s": total_duration_ms / 1000,
                "worker_count": len(self.worker_profiles),
            },
            "aggregate_timing": {
                "total_llm_time_ms": total_llm_time,
                "total_llm_time_s": total_llm_time / 1000,
                "total_tool_time_ms": total_tool_time,
                "total_tool_time_s": total_tool_time / 1000,
                "llm_percentage": (total_llm_time / total_duration_ms * 100) if total_duration_ms > 0 else 0,
                "tool_percentage": (total_tool_time / total_duration_ms * 100) if total_duration_ms > 0 else 0,
            },
            "call_statistics": {
                "total_llm_calls": total_llm_calls,
                "total_tool_calls": total_tool_calls,
                "thinking_mode_calls": thinking_calls,
                "avg_llm_time_ms": total_llm_time / total_llm_calls if total_llm_calls > 0 else 0,
                "avg_tool_time_ms": total_tool_time / total_tool_calls if total_tool_calls > 0 else 0,
            },
            "worker_summaries": [
                {
                    "worker_id": p.worker_id,
                    "worker_type": p.worker_type,
                    "duration_s": p.total_duration_ms / 1000,
                    "llm_calls": p.total_llm_calls,
                    "tool_calls": p.total_tool_calls,
                }
                for p in self.worker_profiles
            ],
        }

        if self.run_logger:
            self._write_lab_summary(summary)

        return summary

    def _write_lab_summary(self, summary: Dict[str, Any]):
        """Write lab-level summary to log."""
        lines = [
            f"\n{'='*80}",
            f"LAB PHASE PROFILING SUMMARY",
            f"{'='*80}",
            f"Total Duration: {summary['lab_phase']['total_duration_s']:.2f}s",
            f"Workers: {summary['lab_phase']['worker_count']}",
            f"",
            f"Aggregate Timing:",
            f"  - LLM Calls: {summary['aggregate_timing']['total_llm_time_s']:.2f}s ({summary['aggregate_timing']['llm_percentage']:.1f}%)",
            f"  - Tool Calls: {summary['aggregate_timing']['total_tool_time_s']:.2f}s ({summary['aggregate_timing']['tool_percentage']:.1f}%)",
            f"  - Other: {summary['lab_phase']['total_duration_s'] - summary['aggregate_timing']['total_llm_time_s'] - summary['aggregate_timing']['total_tool_time_s']:.2f}s",
            f"",
            f"Call Statistics:",
            f"  - Total LLM Calls: {summary['call_statistics']['total_llm_calls']}",
            f"  - Thinking Mode Calls: {summary['call_statistics']['thinking_mode_calls']}",
            f"  - Total Tool Calls: {summary['call_statistics']['total_tool_calls']}",
            f"  - Avg LLM Call: {summary['call_statistics']['avg_llm_time_ms']:.0f}ms",
            f"  - Avg Tool Call: {summary['call_statistics']['avg_tool_time_ms']:.0f}ms",
            f"",
            f"Worker Summaries:",
        ]

        for worker_summary in summary["worker_summaries"]:
            lines.append(
                f"  - {worker_summary['worker_id']}: {worker_summary['duration_s']:.2f}s "
                f"({worker_summary['llm_calls']} LLM calls, {worker_summary['tool_calls']} tool calls)"
            )

        lines.append(f"{'='*80}\n")

        self.run_logger._write_to_file("\n".join(lines))
        self.run_logger.save_lab_profiling_json(summary)
