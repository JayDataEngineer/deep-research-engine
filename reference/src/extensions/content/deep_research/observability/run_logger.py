"""Run logger for research execution tracking.

.. deprecated:: 2.0.0
    Use LangFuse integration instead. See:
    src/extensions/integrations/langfuse/

    Migration guide:
    - RunLogger(session_id) -> langfuse.trace(session_id=session_id)
    - run_logger.log_query(query) -> trace.update(input={"query": query})
    - run_logger.log_phase_call(...) -> trace.generation(...)
    - run_logger.log_tool_call(...) -> trace.span(name=f"tool_{name}")
    - run_logger.add_critique(...) -> client.score(name="quality_gate", ...)

This module provides the main RunLogger class that coordinates:
- File logging via FileHandler
- Metrics tracking via MetricsTracker
- Structured event logging
- Session-based tracking

Usage:
    from .run_logger import RunLogger

    async with RunLogger("my_run", config, base_dir) as logger:
        logger.log_query("What is X?")
        logger.log_phase_call("worker", tokens=100, time=2.5)
        logger.set_response(response_text)

The RunLogger uses session IDs for Context Engine integration, ensuring
all research artifacts are linked to a single session.
"""

import warnings

warnings.warn(
    "RunLogger is deprecated. Use LangFuse integration instead. "
    "See: src/extensions/integrations/langfuse/",
    DeprecationWarning,
    stacklevel=2,
)

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .file_handler import FileHandler, FileHandlerFactory
from .metrics_tracker import MetricsTracker
from .models import WorkerMetrics, save_worker_outputs_json, save_raw_sources_json
from .diagnostic_logging import DiagnosticLogger

# Get logger - safe_get_logger always succeeds
from ..logging_utils import safe_get_logger
logger = safe_get_logger(__name__, "run_logger")


class RunLogger:
    """Main logging class for research runs.

    Coordinates file logging, metrics tracking, and diagnostic logging.
    All artifacts are linked via session_id for Context Engine integration.

    Attributes:
        run_name: Name of the run
        session_id: Primary identifier for the research session (links all artifacts)
        base_dir: Base directory for log files
        file_handler: FileHandler instance for file I/O
        metrics_tracker: MetricsTracker instance for metrics
        diagnostic: DiagnosticLogger for issue detection
    """

    def __init__(
        self,
        run_name: str,
        config: Dict[str, Any],
        base_dir: Path,
        session_id: Optional[str] = None,
    ):
        """Initialize the RunLogger.

        Args:
            run_name: Name of the research run
            config: Full configuration dictionary
            base_dir: Base directory for log files
            session_id: Optional session ID (generated if not provided)
        """
        self.run_name = run_name
        self.config = config
        self.base_dir = base_dir

        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Session management for Context Engine integration
        # IMPORTANT: session_id links all artifacts in Context Engine
        import uuid
        self.session_id = session_id or str(uuid.uuid4())

        # Extract use case and variant from config
        use_case_config = config.get("use_case_config", {})
        self.use_case = use_case_config.get("use_case", "rag")
        self.variant = use_case_config.get("variant", "default")

        # Get observability config
        obs_config = config.get("observability", {})

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            session_id=self.session_id,
            use_case=self.use_case,
            variant=self.variant,
            config=config,
        )

        # Initialize file handler with config
        output_config = obs_config.get("output", {})
        self._file_handler = None
        self._encoding = output_config.get("encoding", "utf-8")
        self._max_file_size = output_config.get("max_file_size", 0)
        self._backup_count = output_config.get("backup_count", 5)

        # Get domain keywords for diagnostic
        diagnostic_config = obs_config.get("diagnostics", {})
        self._domain_keywords = self._get_domain_keywords(diagnostic_config)

        # Initialize components (lazy)
        self._diagnostic = None

        # State tracking
        self._is_open = False

    @property
    def file_handler(self) -> FileHandler:
        """Get or create the file handler."""
        if self._file_handler is None:
            self._file_handler = FileHandlerFactory.create_file_handler(
                base_dir=self.base_dir,
                run_name=self.run_name,
                timestamp=self.timestamp,
                encoding=self._encoding,
                max_file_size=self._max_file_size,
                backup_count=self._backup_count,
            )
        return self._file_handler

    @property
    def diagnostic(self) -> DiagnosticLogger:
        """Get or create the diagnostic logger."""
        if self._diagnostic is None:
            self._diagnostic = DiagnosticLogger(self)
        return self._diagnostic

    @property
    def metrics(self):
        """Get the metrics object."""
        return self.metrics_tracker.metrics

    @property
    def run_id(self) -> str:
        """Alias for session_id for backward compatibility."""
        return self.session_id

    def _get_domain_keywords(self, diagnostic_config: Dict[str, Any]) -> List[str]:
        """Get domain keywords for relevance checking.

        Args:
            diagnostic_config: Diagnostic configuration

        Returns:
            List of domain keywords
        """
        domain_keywords_config = diagnostic_config.get("domain_keywords", {})

        # Check for use-case specific keywords
        use_case_keywords = domain_keywords_config.get(self.use_case)
        if use_case_keywords:
            return use_case_keywords

        # Fall back to default keywords
        return domain_keywords_config.get("default", [])

    def ensure_directory(self) -> None:
        """Ensure run directory exists."""
        self.file_handler.ensure_directory()

    def open(self) -> None:
        """Open log files."""
        if self._is_open:
            return

        self.file_handler.open_log_files()
        self._is_open = True

    def close(self) -> None:
        """Close log files and save final metrics."""
        if not self._is_open:
            return

        try:
            # Save final metrics
            self._save_metrics()

            # Save worker outputs
            self._save_worker_outputs()

            # Save raw sources
            self._save_raw_sources()
        except Exception as e:
            logger.error(f"Error saving final data: {e}")
        finally:
            self.file_handler.close_log_files()
            self._is_open = False

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        if exc_type is not None:
            self._write_to_file(f"[ERROR] {exc_type.__name__}: {exc_val}")
        self.close()
        return False  # Don't suppress exceptions

    def _write_to_file(self, message: str) -> None:
        """Write to app.log."""
        if self._is_open:
            self.file_handler.write_to_log(message)

    def log_error(self, phase: str, error: Exception, context: str = "") -> None:
        """Log an error with context.

        Args:
            phase: The phase where the error occurred
            error: The exception that was raised
            context: Additional context about the error
        """
        error_msg = f"[ERROR] [{phase}] {type(error).__name__}: {error}"
        if context:
            error_msg += f" (context: {context})"
        self._write_to_file(error_msg)
        logger.error(f"[{phase}] {error}")

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_data = {
            "session_id": self.session_id,
            "run_id": self.session_id,  # Backward compat alias
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "use_case": self.use_case,
            "variant": self.variant,
            **self.metrics_tracker.get_summary(),
        }
        self.file_handler.write_json("metrics.json", metrics_data)

    def _save_worker_outputs(self) -> None:
        """Save worker outputs to context_engine storage via save_output tool."""
        if self.metrics.worker_outputs:
            # Use context_engine save_output tool - NO direct file I/O
            save_worker_outputs_json(
                session_id=self.session_id,
                worker_outputs=self.metrics.worker_outputs
            )

    def _save_raw_sources(self) -> None:
        """Save raw sources to context_engine storage via save_output tool."""
        if hasattr(self.metrics, "raw_sources") and self.metrics.raw_sources:
            # Use context_engine save_output tool - NO direct file I/O
            save_raw_sources_json(
                session_id=self.session_id,
                raw_sources=self.metrics.raw_sources
            )

    def save_worker_outputs_json(self, worker_outputs: List[Any]) -> None:
        """Save worker outputs to context_engine storage.

        This method is called by the graph to save worker outputs mid-run.
        Updates the internal metrics and saves to context_engine.

        Args:
            worker_outputs: List of worker output objects
        """
        # Update metrics
        self.metrics.worker_outputs = worker_outputs

        # Save to context_engine immediately (don't wait for close())
        save_worker_outputs_json(
            session_id=self.session_id,
            worker_outputs=worker_outputs
        )

    def save_raw_sources_json(self, raw_sources: Dict[str, Any]) -> None:
        """Save raw sources to context_engine storage.

        This method is called by the graph to save raw sources mid-run.
        Updates the internal metrics and saves to context_engine.

        Args:
            raw_sources: Dictionary of source objects
        """
        # Update metrics
        self.metrics.raw_sources = raw_sources

        # Save to context_engine immediately
        save_raw_sources_json(
            session_id=self.session_id,
            raw_sources=raw_sources
        )

    # ========================================================================
    # LOGGING METHODS
    # ========================================================================

    def log_query(self, query: str) -> None:
        """Log the initial research query.

        Args:
            query: The research query
        """
        self._write_to_file(f"[QUERY] {query}")
        self.metrics_tracker.metrics.query = query

    def log_phase_start(self, phase: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log the start of a phase.

        Args:
            phase: Phase name (worker, boss, lab, critic, synthesis)
            details: Optional metadata
        """
        details = details or {}
        metadata_str = f" ({details})" if details else ""
        self._write_to_file(f"[{phase.upper()}] Starting{metadata_str}")

    def log_phase_end(self, phase: str, elapsed: float, details: Optional[Dict[str, Any]] = None) -> None:
        """Log the end of a phase.

        Args:
            phase: Phase name
            elapsed: Elapsed time in seconds
            details: Optional metadata
        """
        details = details or {}
        metadata_str = f" ({details})" if details else ""
        self._write_to_file(f"[{phase.upper()}] Completed in {elapsed:.2f}s{metadata_str}")

    def log_phase_call(
        self,
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        time_seconds: float,
        tool_calls: int = 0,
        worker_id: Optional[str] = None,
    ) -> None:
        """Log an LLM call during a phase.

        Args:
            phase: Phase name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            time_seconds: Duration
            tool_calls: Number of tool calls
            worker_id: Optional worker ID
        """
        self.metrics_tracker.track_call(
            phase=phase,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_seconds=time_seconds,
            tool_calls=tool_calls,
            worker_id=worker_id,
        )
        self._write_to_file(
            f"[{phase.upper()}] LLM Call: "
            f"{prompt_tokens + completion_tokens} tokens, {time_seconds:.2f}s"
        )

    def log_tool_call(
        self,
        tool_name: str,
        duration: float,
        output_size: int,
        worker_id: Optional[str] = None,
    ) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the tool
            duration: Duration in seconds
            output_size: Size of output
            worker_id: Optional worker ID
        """
        self.metrics_tracker.track_tool_call(
            tool_name=tool_name,
            duration=duration,
            output_size=output_size,
            worker_id=worker_id,
        )
        self._write_to_file(
            f"[TOOL] {tool_name}: {duration:.2f}s, {output_size} bytes"
        )

    def log_worker_start(self, worker_id: str, task: str) -> None:
        """Log worker start.

        Args:
            worker_id: Worker identifier
            task: Task description
        """
        self.metrics_tracker.start_worker(worker_id)
        self._write_to_file(f"[WORKER] {worker_id} started: {task[:100]}...")

    def log_worker_end(
        self,
        worker_id: str,
        worker_type: str,
        task: str,
        output_length: int,
        source_count: int,
        confidence: float,
    ) -> None:
        """Log worker completion.

        Args:
            worker_id: Worker identifier
            worker_type: Type of worker
            task: Task description
            output_length: Length of output
            source_count: Number of sources
            confidence: Confidence score
        """
        self.metrics_tracker.end_worker(
            worker_id=worker_id,
            worker_type=worker_type,
            task_description=task,
            output_length=output_length,
            source_count=source_count,
            confidence=confidence,
            status="success",
        )
        self._write_to_file(
            f"[WORKER] {worker_id} completed: "
            f"{output_length} chars, {source_count} sources, {confidence:.2f} confidence"
        )

    def set_response(self, response: str) -> None:
        """Set the final response.

        Args:
            response: Final response text
        """
        self.metrics_tracker.metrics.response_length = len(response)

        # Save response to file
        self.file_handler.write_json("response.json", {"response": response})

    def add_critique(self, score: float, passed: bool, feedback: str, gaps: List[str] = None) -> None:
        """Add a critique entry.

        Args:
            score: Quality score (0-1)
            passed: Whether quality gate passed
            feedback: Feedback message
            gaps: Optional list of gaps
        """
        self.metrics_tracker.add_critique(score, passed, feedback, gaps)
        self._write_to_file(
            f"[CRITIC] Score: {score:.2f}, Passed: {passed}, Feedback: {feedback[:200]}..."
        )

    def get_run_dir(self) -> Path:
        """Get the run directory path.

        Returns:
            Path to the run directory
        """
        return self.file_handler.run_dir

    def set_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Set tool calls breakdown from callback handler.

        Args:
            tool_calls: List of tool call dictionaries with tool name,
                       duration, output_length, and optional researcher ID
        """
        self.metrics.tool_calls = tool_calls

    @classmethod
    def from_config(cls, config: Dict[str, Any], base_dir: Path, session_id: Optional[str] = None) -> "RunLogger":
        """Create a RunLogger from configuration.

        Args:
            config: Configuration dictionary
            base_dir: Base directory for logs
            session_id: Optional session ID

        Returns:
            Configured RunLogger instance
        """
        run_name = config.get("run_name", "research_run")
        return cls(run_name, config, base_dir, session_id)


__all__ = ["RunLogger", "from_config"]
