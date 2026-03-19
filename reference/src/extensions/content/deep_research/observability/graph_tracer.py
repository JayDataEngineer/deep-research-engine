"""Graph Execution Tracer - Real-time visibility into LangGraph execution.

.. deprecated:: 2.0.0
    Use LangFuse tracing instead. See:
    src/extensions/integrations/langfuse/tracing.py

    LangFuse automatically traces LangGraph node execution via callbacks.
    For custom spans, use trace_research_phase().

    Migration:
    - GraphExecutionTracer() -> LangFuse CallbackHandler (automatic)
    - wrap_node_with_tracing() -> Not needed, handled by callback

This module provides deep observability into the Deep Research graph by:
1. Tracking every node execution with timing
2. Logging Command API decisions (goto targets)
3. Counting steps per node to detect loops
4. Providing real-time console output for debugging

USAGE:
    The tracer is automatically enabled when DEEP_RESEARCH_TRACE=1 is set.

    export DEEP_RESEARCH_TRACE=1
    uv run python test_deep_research_e2e.py

OUTPUT FORMAT:
    [TRACE] ══════════════════════════════════════════════════════════
    [TRACE] NODE #1: boss (START)
    [TRACE]   └─ Duration: 2.34s
    [TRACE]   └─ Decision: goto=lab_architect
    [TRACE]   └─ State: iteration_count=1, boss_iteration_count=0
    [TRACE] ══════════════════════════════════════════════════════════
    [TRACE] NODE #2: lab_architect
    [TRACE]   └─ Workers spawned: 3
    [TRACE]   └─ Duration: 15.67s
    ...

This helps answer:
- Which nodes are being executed?
- How many times has Boss run?
- How many workers spawned?
- Why isn't the graph terminating?
"""

import warnings

warnings.warn(
    "GraphExecutionTracer is deprecated. Use LangFuse tracing instead. "
    "See: src/extensions/integrations/langfuse/tracing.py",
    DeprecationWarning,
    stacklevel=2,
)

import os
import sys
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, Callable, List
from functools import wraps
from dataclasses import dataclass, field

from ..logging_utils import safe_get_logger

logger = safe_get_logger(__name__, "graph_tracer")


def _is_tracing_enabled() -> bool:
    """Check if tracing is enabled (runtime check)."""
    return os.environ.get("DEEP_RESEARCH_TRACE", "0") == "1"


@dataclass
class NodeExecution:
    """Record of a single node execution."""
    step: int
    node_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    goto_target: Optional[str] = None
    state_keys: List[str] = field(default_factory=list)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    workers_spawned: int = 0
    is_start: bool = False
    is_end: bool = False


class GraphExecutionTracer:
    """Tracks and logs graph execution in real-time.

    This tracer provides visibility into LangGraph execution by:
    1. Wrapping node functions to track entry/exit
    2. Extracting Command decisions (goto targets)
    3. Taking state snapshots at each node
    4. Counting executions per node

    Usage:
        tracer = GraphExecutionTracer()

        # Wrap a node function
        wrapped_node = tracer.wrap_node("boss", unified_boss_node)

        # Get summary
        print(tracer.get_summary())
    """

    def __init__(self, max_traces: int = 1000):
        self.max_traces = max_traces
        self.executions: List[NodeExecution] = []
        self.node_counts: Dict[str, int] = {}
        self.step_counter = 0
        self.start_time = time.time()
        self._current_node_start: Optional[float] = None

    def _should_trace(self) -> bool:
        """Check if tracing is enabled."""
        return _is_tracing_enabled()

    def _print(self, message: str):
        """Print trace message if enabled."""
        if self._should_trace():
            print(f"[TRACE] {message}", flush=True)

    def _extract_goto_target(self, result: Any) -> Optional[str]:
        """Extract goto target from Command result.

        LangGraph nodes return Command objects with goto parameter.
        """
        # Check for langgraph.types.Command
        if hasattr(result, 'goto'):
            goto = result.goto
            if isinstance(goto, str):
                return goto
            elif hasattr(goto, '__name__'):
                return goto.__name__
            return str(goto)

        # Check for dict with goto key
        if isinstance(result, dict):
            if 'goto' in result:
                goto = result['goto']
                if isinstance(goto, str):
                    return goto
                elif hasattr(goto, '__name__'):
                    return goto.__name__
                return str(goto)

        return None

    def _extract_state_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key state values for logging."""
        snapshot = {}

        # Key state keys to track
        important_keys = [
            "iteration_count",
            "boss_iteration_count",
            "workers_to_spawn",
            "worker_outputs",
            "critique_history",
            "current_phase",
            "is_refinement",
            "query",
            "session_id",
        ]

        for key in important_keys:
            if key in state:
                value = state[key]
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    snapshot[key] = value[:100] + "..."
                elif isinstance(value, list):
                    snapshot[key] = f"[{len(value)} items]"
                elif isinstance(value, dict):
                    snapshot[key] = f"{{{len(value)} keys}}"
                else:
                    snapshot[key] = value

        return snapshot

    def wrap_node(self, node_name: str, node_func: Callable) -> Callable:
        """Wrap a node function to trace its execution.

        Args:
            node_name: Name of the node (e.g., "boss", "critic")
            node_func: The async node function to wrap

        Returns:
            Wrapped function that traces execution
        """
        @wraps(node_func)
        async def traced_node(state, config=None, runtime=None, **kwargs):
            # Increment step counter
            self.step_counter += 1
            step = self.step_counter

            # Track node execution count
            self.node_counts[node_name] = self.node_counts.get(node_name, 0) + 1
            node_count = self.node_counts[node_name]

            # Start timing
            start_time = time.time()

            # Create execution record
            execution = NodeExecution(
                step=step,
                node_name=node_name,
                start_time=start_time,
                state_keys=list(state.keys()) if isinstance(state, dict) else [],
                state_snapshot=self._extract_state_snapshot(state) if isinstance(state, dict) else {},
            )

            # Log node entry
            is_start = node_name == "boss" and node_count == 1
            execution.is_start = is_start

            self._print("═" * 60)
            if is_start:
                self._print(f"NODE #{step}: {node_name} (START)")
            else:
                self._print(f"NODE #{step}: {node_name} (count: {node_count})")

            # Log state snapshot
            if execution.state_snapshot:
                self._print(f"  └─ State: {execution.state_snapshot}")

            # Execute the node - pass all kwargs through
            result = None
            try:
                # Build kwargs for the node function
                node_kwargs = {}
                if config is not None:
                    node_kwargs['config'] = config
                if runtime is not None:
                    node_kwargs['runtime'] = runtime
                node_kwargs.update(kwargs)

                result = await node_func(state, **node_kwargs)

                # Extract goto target if Command was returned
                goto_target = self._extract_goto_target(result)
                execution.goto_target = goto_target

                # Log result
                end_time = time.time()
                execution.end_time = end_time
                execution.duration = end_time - start_time

                if goto_target:
                    self._print(f"  └─ Decision: goto={goto_target}")
                else:
                    self._print(f"  └─ Duration: {execution.duration:.2f}s")

                # Check for workers spawned
                if isinstance(result, dict) and "workers_to_spawn" in result:
                    workers = result["workers_to_spawn"]
                    execution.workers_spawned = len(workers) if isinstance(workers, list) else 0
                    if execution.workers_spawned > 0:
                        self._print(f"  └─ Workers spawned: {execution.workers_spawned}")

                return result

            except Exception as e:
                execution.error = str(e)
                self._print(f"  └─ ERROR: {e}")
                raise

            finally:
                # Store execution record
                if len(self.executions) < self.max_traces:
                    self.executions.append(execution)

        return traced_node

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        total_time = time.time() - self.start_time

        # Count by node
        node_summary = {}
        for name, count in self.node_counts.items():
            node_execs = [e for e in self.executions if e.node_name == name]
            total_duration = sum(e.duration or 0 for e in node_execs)
            node_summary[name] = {
                "count": count,
                "total_time": total_duration,
                "avg_time": total_duration / count if count > 0 else 0,
            }

        # Find loops
        loops = []
        seen_sequences = set()
        for i in range(len(self.executions) - 3):
            seq = tuple(e.node_name for e in self.executions[i:i+4])
            if seq in seen_sequences:
                loops.append(seq)
            seen_sequences.add(seq)

        return {
            "total_steps": self.step_counter,
            "total_time": total_time,
            "node_counts": self.node_counts,
            "node_summary": node_summary,
            "potential_loops": loops[:5],  # Top 5 detected loops
            "execution_order": [e.node_name for e in self.executions[-20:]],  # Last 20 nodes
        }

    def print_summary(self):
        """Print execution summary to console."""
        summary = self.get_summary()

        self._print("")
        self._print("=" * 60)
        self._print("GRAPH EXECUTION SUMMARY")
        self._print("=" * 60)
        self._print(f"Total Steps: {summary['total_steps']}")
        self._print(f"Total Time: {summary['total_time']:.2f}s")
        self._print("")
        self._print("Node Execution Counts:")
        for name, stats in summary['node_summary'].items():
            self._print(f"  {name}: {stats['count']}x ({stats['total_time']:.2f}s total)")
        self._print("")
        self._print("Recent Execution Order:")
        self._print(f"  {' → '.join(summary['execution_order'])}")

        if summary['potential_loops']:
            self._print("")
            self._print("⚠️  Potential Loops Detected:")
            for loop in summary['potential_loops']:
                self._print(f"  {' → '.join(loop)}")

        self._print("=" * 60)


# Global tracer instance (fallback when no context is set)
_global_tracer: Optional[GraphExecutionTracer] = None


def get_tracer() -> GraphExecutionTracer:
    """Get or create the tracer instance.

    First checks the observability context (thread-safe), then falls back
    to the global tracer for backward compatibility.

    Returns:
        GraphExecutionTracer instance
    """
    # First, try to get from context
    from .context import get_context_or_none
    ctx = get_context_or_none()
    if ctx:
        if ctx.tracer is None:
            ctx.tracer = GraphExecutionTracer()
        return ctx.tracer

    # Fall back to global tracer
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = GraphExecutionTracer()
    return _global_tracer


def reset_tracer():
    """Reset the tracer for a new run.

    Resets in the observability context if available, otherwise resets
    the global tracer for backward compatibility.
    """
    # Try to reset in context first
    from .context import get_context_or_none
    ctx = get_context_or_none()
    if ctx:
        ctx.tracer = GraphExecutionTracer()
        return

    # Fall back to global tracer
    global _global_tracer
    _global_tracer = GraphExecutionTracer()


def wrap_node_with_tracing(node_name: str, node_func: Callable) -> Callable:
    """Convenience function to wrap a node with tracing.

    Args:
        node_name: Name of the node
        node_func: The node function to wrap

    Returns:
        Wrapped function (or original if tracing disabled)
    """
    if not _is_tracing_enabled():
        return node_func

    # Log that we're wrapping this node
    print(f"[TRACE] Wrapping node: {node_name}", flush=True)
    tracer = get_tracer()
    return tracer.wrap_node(node_name, node_func)
