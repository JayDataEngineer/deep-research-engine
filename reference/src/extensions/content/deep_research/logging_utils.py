"""Centralized logging for Deep Research extension.

Uses standard Python logging with centralized configuration from src/shared/logging_config.py.
Supports config-based level filtering (normal, quiet, silent, debug).

Features:
- Component-specific log levels from config.yaml
- Timing/profiling helpers with breakdown support
- Structured JSON logging for production monitoring
- Unified logging facade for consistent patterns

Usage:
    from .logging_utils import get_logger, time_operation, DeepResearchLogger
    logger = get_logger(__name__)

    # Simple logging
    logger.info("Research complete")

    # Timing operations
    async with time_operation(logger, "web_search") as timer:
        results = await search(query)
        timer.add_checkpoint("search_done")
        content = await scrape(urls)
        timer.add_checkpoint("scrape_done")
    # Logs: [web_search] completed in 2.50s (search_done: 1.00s, scrape_done: 1.50s)

    # Structured logging
    structured = DeepResearchLogger(session_id="abc-123", format="json")
    structured.log_event("research_started", query="What is X?")
"""
from typing import Any, Dict, List, Optional, Literal
import logging
import time
import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


# =============================================================================
# CONFIG LEVELS MAPPING
# =============================================================================

# Map config levels to standard logging levels
LEVEL_MAPPING = {
    "debug": logging.DEBUG,      # Show everything
    "normal": logging.INFO,       # Show key progress (default)
    "quiet": logging.WARNING,     # Show warnings/errors only
    "silent": logging.ERROR,      # Show errors only
}


# =============================================================================
# LOGGING CONSTANTS - CONSISTENT NAMING CONVENTIONS
# =============================================================================

# Component names - use these for get_logger(name, COMPONENT_X)
COMPONENT_BOSS = "boss"
COMPONENT_LAB = "lab"
COMPONENT_LAB_ARCHITECT = "lab_architect"
COMPONENT_WORKER = "worker"  # Research workers
COMPONENT_CRITIC = "critic"
COMPONENT_SYNTHESIS = "synthesis"
COMPONENT_FLASH_EXPLORER = "flash_explorer"
COMPONENT_WEB_AGENT = "web_agent"
COMPONENT_CONTEXT_ENGINE = "context_engine"

# Log message prefixes - ALWAYS uppercase with brackets
# Usage: logger.info(f"{PREFIX_BOSS} Starting...")
PREFIX_BOSS = "[BOSS]"
PREFIX_LAB = "[LAB]"
PREFIX_LAB_ARCHITECT = "[LAB_ARCHITECT]"
PREFIX_WORKER = "[WORKER]"  # Research workers
PREFIX_CRITIC = "[CRITIC]"
PREFIX_SYNTHESIS = "[SYNTHESIS]"
PREFIX_FLASH_EXPLORER = "[FLASH_EXPLORER]"
PREFIX_WEB_AGENT = "[WEB_AGENT]"
PREFIX_CONTEXT_ENGINE = "[CONTEXT_ENGINE]"

# Timing format - ALWAYS use .2f for seconds
TIMING_FORMAT = "{elapsed:.2f}s"
TIMING_FORMAT_WITH_NAME = "{name}: {elapsed:.2f}s"


def format_timing(elapsed: float) -> str:
    """Format elapsed time consistently."""
    return TIMING_FORMAT.format(elapsed=elapsed)


def format_timing_breakdown(elapsed: float, breakdown: Dict[str, float]) -> str:
    """Format timing with breakdown consistently.

    Args:
        elapsed: Total elapsed time in seconds
        breakdown: Dict of step name -> time in seconds

    Returns:
        Formatted string like "2.50s (search: 1.00s, scrape: 1.50s)"
    """
    if not breakdown:
        return format_timing(elapsed)
    parts = [f"{k}: {v:.2f}s" for k, v in breakdown.items()]
    return f"{elapsed:.2f}s ({', '.join(parts)})"


# =============================================================================
# RUN_LOGGER HELPERS (for output.md logging)
# =============================================================================

def log_component(run_logger: Any, component: str, message: str) -> None:
    """Log to run_logger with automatic component prefix.

    This is the preferred way to write to output.md - ensures consistent
    formatting across all nodes and tools.

    Args:
        run_logger: RunLogger instance (can be None - will be handled gracefully)
        component: Component name (e.g., "BOSS", "CRITIC", "WORKER")
        message: Log message (without prefix - will be added automatically)

    Usage:
        from ...logging_utils import log_component
        log_component(run_logger, "BOSS", f"Query: {query[:100]}...")
        # Output: "[BOSS] Query: what is..."
    """
    if run_logger is not None:
        run_logger._write_to_file(f"[{component.upper()}] {message}")


def log_component_warning(run_logger: Any, component: str, message: str) -> None:
    """Log warning to run_logger with automatic component prefix."""
    log_component(run_logger, component, f"WARNING: {message}")


def log_component_error(run_logger: Any, component: str, message: str) -> None:
    """Log error to run_logger with automatic component prefix."""
    log_component(run_logger, component, f"ERROR: {message}")


def log_phase_start(logger: logging.Logger, prefix: str, phase_details: str = "") -> None:
    """Log phase start consistently.

    Args:
        logger: Logger instance
        prefix: Component prefix (e.g., PREFIX_WORKER)
        phase_details: Optional details about the phase
    """
    if phase_details:
        logger.info(f"{prefix} Starting: {phase_details}")
    else:
        logger.info(f"{prefix} Starting")


def log_phase_end(logger: logging.Logger, prefix: str, elapsed: float, details: str = "") -> None:
    """Log phase end with timing consistently.

    Args:
        logger: Logger instance
        prefix: Component prefix (e.g., PREFIX_WORKER)
        elapsed: Time elapsed in seconds
        details: Optional details about the results
    """
    timing = format_timing(elapsed)
    if details:
        logger.info(f"{prefix} Completed in {timing} - {details}")
    else:
        logger.info(f"{prefix} Completed in {timing}")


def log_error_with_context(
    logger: logging.Logger,
    prefix: str,
    error: Exception,
    context: str = "",
    level: int = logging.ERROR
) -> None:
    """Log error with context consistently.

    Args:
        logger: Logger instance
        prefix: Component prefix (e.g., PREFIX_WORKER)
        error: The exception that occurred
        context: Additional context about where the error occurred
        level: Log level (default: ERROR)
    """
    context_str = f" in {context}" if context else ""
    logger.log(level, f"{prefix} ERROR{context_str}: {type(error).__name__}: {error}")


def _get_config_level() -> str:
    """Get the configured log level from config.yaml."""
    try:
        import yaml
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("logging", {}).get("level", "normal")
    except Exception:
        pass
    return "normal"


def _get_component_level(component_name: str) -> str | None:
    """Get component-specific log level from config.yaml."""
    try:
        import yaml
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("logging", {}).get("components", {}).get(component_name)
    except Exception:
        pass
    return None


def get_logger(name: str, component_name: str | None = None) -> logging.Logger:
    """Get a logger that respects config levels.

    Args:
        name: Logger name (typically __name__)
        component_name: Optional component name for component-specific levels

    Returns:
        A standard Python logger that respects normal/quiet/silent/debug levels

    Examples:
        logger = get_logger(__name__)  # Use global level
        logger = get_logger(__name__, "flash_explorer")  # Component-specific level
    """
    # Determine effective level
    effective_level_str = _get_component_level(component_name) or _get_config_level()
    effective_level = LEVEL_MAPPING.get(effective_level_str, logging.INFO)

    # Get standard library logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(effective_level)

    return logger


def safe_get_logger(name: str, component_name: str | None = None) -> logging.Logger:
    """Get a logger with graceful fallback - ALWAYS succeeds.

    This function should be used for module-level logger initialization
    where try/except blocks are undesirable. It will always return a
    working logger, even if the config system is unavailable.

    Args:
        name: Logger name (typically __name__)
        component_name: Optional component name for component-specific levels

    Returns:
        A standard Python logger (with config levels if available, basic if not)

    Examples:
        # At module level (no try/except needed)
        from ..logging_utils import safe_get_logger
        logger = safe_get_logger(__name__, "run_logger")
    """
    try:
        return get_logger(name, component_name)
    except Exception:
        # Fall back to basic logger
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# =============================================================================
# COMPONENT-SPECIFIC LOGGERS
# =============================================================================

def get_worker_logger() -> logging.Logger:
    """Get a logger for worker component with component-specific filtering."""
    return get_logger(COMPONENT_WORKER, COMPONENT_WORKER)


def get_boss_logger() -> logging.Logger:
    """Get a logger for boss component with component-specific filtering."""
    return get_logger(COMPONENT_BOSS, COMPONENT_BOSS)


def get_lab_logger() -> logging.Logger:
    """Get a logger for lab component with component-specific filtering."""
    return get_logger(COMPONENT_LAB, COMPONENT_LAB)


def get_lab_architect_logger() -> logging.Logger:
    """Get a logger for lab_architect component with component-specific filtering."""
    return get_logger(COMPONENT_LAB_ARCHITECT, COMPONENT_LAB_ARCHITECT)


def get_critic_logger() -> logging.Logger:
    """Get a logger for critic component with component-specific filtering."""
    return get_logger(COMPONENT_CRITIC, COMPONENT_CRITIC)


def get_synthesis_logger() -> logging.Logger:
    """Get a logger for synthesis component with component-specific filtering."""
    return get_logger(COMPONENT_SYNTHESIS, COMPONENT_SYNTHESIS)


def get_flash_explorer_logger() -> logging.Logger:
    """Get a logger for flash_explorer component with component-specific filtering."""
    return get_logger(COMPONENT_FLASH_EXPLORER, COMPONENT_FLASH_EXPLORER)


def get_web_agent_logger() -> logging.Logger:
    """Get a logger for web_agent component with component-specific filtering."""
    return get_logger(COMPONENT_WEB_AGENT, COMPONENT_WEB_AGENT)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def log_exception(logger: logging.Logger, message: str, exc_info: Any = None) -> None:
    """Log an exception with traceback if available."""
    if exc_info:
        logger.error(message, exc_info=exc_info)
    else:
        logger.error(message)


def log_with_context(logger: logging.Logger, message: str, **context) -> None:
    """Log a message with additional context."""
    logger.info(f"{message} {context}" if context else message)


# =============================================================================
# TIMING/PROFILING HELPERS
# =============================================================================

@dataclass
class TimingCheckpoint:
    """A single timing checkpoint with name and elapsed time."""
    name: str
    elapsed_ms: float  # Time since operation start in milliseconds


@dataclass
class OperationTimer:
    """Timer for tracking operation duration with checkpoints.

    Use with time_operation() context manager for clean timing.

    Attributes:
        operation_name: Name of the operation being timed
        start_time: When the operation started (monotonic time)
        checkpoints: List of timing checkpoints
    """
    operation_name: str
    start_time: float = field(default_factory=time.monotonic)
    checkpoints: List[TimingCheckpoint] = field(default_factory=list)
    _last_checkpoint_time: float = field(default_factory=time.monotonic, repr=False)

    def add_checkpoint(self, name: str) -> float:
        """Add a timing checkpoint.

        Args:
            name: Checkpoint name (e.g., "search_done", "scrape_done")

        Returns:
            Time since last checkpoint in seconds (for optional use)
        """
        now = time.monotonic()
        elapsed_ms = (now - self.start_time) * 1000
        step_ms = (now - self._last_checkpoint_time) * 1000
        self.checkpoints.append(TimingCheckpoint(name=name, elapsed_ms=elapsed_ms))
        self._last_checkpoint_time = now
        return step_ms / 1000

    @property
    def total_seconds(self) -> float:
        """Get total elapsed time in seconds."""
        return time.monotonic() - self.start_time

    @property
    def total_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return self.total_seconds * 1000

    def format_breakdown(self) -> str:
        """Format timing breakdown as a readable string.

        Returns:
            String like "2.5s (search: 1.0s, scrape: 1.5s)"
        """
        if not self.checkpoints:
            return f"{self.total_seconds:.2f}s"

        # Calculate durations between checkpoints
        parts = []
        prev_time = 0.0
        for cp in self.checkpoints:
            duration_s = (cp.elapsed_ms - prev_time) / 1000
            parts.append(f"{cp.name}: {duration_s:.2f}s")
            prev_time = cp.elapsed_ms

        return f"{self.total_seconds:.2f}s ({', '.join(parts)})"


@asynccontextmanager
async def time_operation(
    logger: logging.Logger,
    operation_name: str,
    level: int = logging.INFO,
    log_on_start: bool = False
):
    """Context manager for timing async operations with checkpoint support.

    Args:
        logger: Logger instance to use
        operation_name: Name of the operation being timed
        level: Log level for the completion message (default: INFO)
        log_on_start: If True, log when operation starts

    Yields:
        OperationTimer: Timer object for adding checkpoints

    Example:
        async with time_operation(logger, "web_research") as timer:
            results = await search(query)
            timer.add_checkpoint("search_done")
            content = await scrape(urls)
            timer.add_checkpoint("scrape_done")
        # Logs: [web_research] completed in 2.50s (search_done: 1.00s, scrape_done: 1.50s)
    """
    timer = OperationTimer(operation_name=operation_name)

    if log_on_start:
        logger.log(level, f"[{operation_name}] Starting...")

    try:
        yield timer
    finally:
        breakdown = timer.format_breakdown()
        logger.log(level, f"[{operation_name}] completed in {breakdown}")


def log_timing_breakdown(
    logger: logging.Logger,
    operation_name: str,
    timings: Dict[str, float],
    level: int = logging.INFO
) -> None:
    """Log a timing breakdown for an operation.

    This is useful when you have timing data from multiple sources
    and want to log it in a consistent format.

    Args:
        logger: Logger instance to use
        operation_name: Name of the operation
        timings: Dict of checkpoint name -> time in seconds
        level: Log level (default: INFO)

    Example:
        timings = {"search": 1.2, "scrape": 3.5, "synthesize": 0.8}
        log_timing_breakdown(logger, "web_research", timings)
        # Logs: [web_research] timing breakdown - total: 5.50s, search: 1.20s, scrape: 3.50s, synthesize: 0.80s
    """
    total = sum(timings.values())
    parts = [f"{k}: {v:.2f}s" for k, v in timings.items()]
    logger.log(level, f"[{operation_name}] timing breakdown - total: {total:.2f}s, {', '.join(parts)}")


class ProfilingLogger:
    """Logger wrapper that adds automatic timing to method calls.

    Useful for profiling complex operations with multiple steps.

    Example:
        profiler = ProfilingLogger(logger, "web_agent")
        await profiler.time_step("search", lambda: web_search(query))
        await profiler.time_step("scrape", lambda: web_scrape(urls))
        profiler.log_summary()
    """

    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.timings: Dict[str, float] = {}
        self._start_time = time.monotonic()

    async def time_step(self, step_name: str, coro_factory, *args, **kwargs) -> Any:
        """Time a single step and store the result.

        Args:
            step_name: Name of the step
            coro_factory: Async callable to execute
            *args, **kwargs: Arguments to pass to callable

        Returns:
            Result of the callable
        """
        start = time.monotonic()
        try:
            result = await coro_factory(*args, **kwargs)
            return result
        finally:
            elapsed = time.monotonic() - start
            self.timings[step_name] = elapsed
            self.logger.debug(f"[{self.operation_name}] {step_name}: {elapsed:.2f}s")

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log a summary of all timed steps."""
        if not self.timings:
            return

        total = time.monotonic() - self._start_time
        parts = [f"{k}: {v:.2f}s" for k, v in self.timings.items()]
        self.logger.log(
            level,
            f"[{self.operation_name}] summary - total: {total:.2f}s, steps: {', '.join(parts)}"
        )


# =============================================================================
# STRUCTURED LOGGING SUPPORT
# =============================================================================

class LogFormat(str, Enum):
    """Log output format options."""
    TEXT = "text"
    JSON = "json"


class LogLevel(str, Enum):
    """Standard log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StructuredLogEntry:
    """A single structured log entry."""
    timestamp: str
    level: str
    session_id: str
    component: str
    event: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))

    def to_text(self) -> str:
        """Convert to human-readable text."""
        data_str = f" | {self.data}" if self.data else ""
        return f"[{self.timestamp}] [{self.level}] [{self.component}] {self.message}{data_str}"


class DeepResearchLogger:
    """Unified logging facade for Deep Research.

    Provides both text and JSON structured logging with automatic
    session tracking, component tagging, and event categorization.

    Usage:
        # Text logging (default)
        logger = DeepResearchLogger(session_id="abc-123")
        logger.info("Research started", component="BOSS")

        # JSON structured logging
        logger = DeepResearchLogger(session_id="abc-123", format="json")
        logger.log_event("research_started", query="What is X?", component="BOSS")

        # With RunLogger integration
        logger = DeepResearchLogger(session_id="abc-123", run_logger=run_logger)
        logger.info("Worker spawned", component="LAB")
    """

    def __init__(
        self,
        session_id: str,
        format: Literal["text", "json"] = "text",
        run_logger: Any = None,
        python_logger: Optional[logging.Logger] = None,
    ):
        """Initialize the structured logger.

        Args:
            session_id: Research session identifier
            format: Output format ("text" or "json")
            run_logger: Optional RunLogger for file output
            python_logger: Optional Python logger for console output
        """
        self.session_id = session_id
        self.format = LogFormat(format)
        self.run_logger = run_logger
        self.python_logger = python_logger or logging.getLogger(__name__)
        self._entries: List[StructuredLogEntry] = []

    def _create_entry(
        self,
        level: str,
        component: str,
        event: str,
        message: str,
        **data
    ) -> StructuredLogEntry:
        """Create a structured log entry."""
        return StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            session_id=self.session_id,
            component=component.upper(),
            event=event,
            message=message,
            data=data,
        )

    def _write_entry(self, entry: StructuredLogEntry) -> None:
        """Write entry to configured outputs."""
        self._entries.append(entry)

        # Write to RunLogger if available
        if self.run_logger:
            if self.format == LogFormat.JSON:
                self.run_logger._write_to_file(entry.to_json())
            else:
                self.run_logger._write_to_file(entry.to_text())

        # Write to Python logger if available
        if self.python_logger:
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            self.python_logger.log(
                level_map.get(entry.level, logging.INFO),
                entry.to_text()
            )

    def log_event(
        self,
        event: str,
        message: str,
        component: str = "GENERAL",
        level: str = "info",
        **data
    ) -> None:
        """Log a structured event.

        Args:
            event: Event name (e.g., "research_started", "worker_spawned")
            message: Human-readable message
            component: Component name (BOSS, WORKER, CRITIC, etc.)
            level: Log level (debug, info, warning, error, critical)
            **data: Additional structured data
        """
        entry = self._create_entry(level, component, event, message, **data)
        self._write_entry(entry)

    def debug(self, message: str, component: str = "GENERAL", **data) -> None:
        """Log debug message."""
        self.log_event("debug", message, component, "debug", **data)

    def info(self, message: str, component: str = "GENERAL", **data) -> None:
        """Log info message."""
        self.log_event("info", message, component, "info", **data)

    def warning(self, message: str, component: str = "GENERAL", **data) -> None:
        """Log warning message."""
        self.log_event("warning", message, component, "warning", **data)

    def error(self, message: str, component: str = "GENERAL", **data) -> None:
        """Log error message."""
        self.log_event("error", message, component, "error", **data)

    def critical(self, message: str, component: str = "GENERAL", **data) -> None:
        """Log critical message."""
        self.log_event("critical", message, component, "critical", **data)

    def log_phase_start(
        self,
        phase: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log phase start."""
        self.log_event(
            "phase_start",
            f"Starting: {phase}",
            component=phase.upper(),
            level="info",
            **(details or {})
        )

    def log_phase_end(
        self,
        phase: str,
        elapsed: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log phase end with timing."""
        self.log_event(
            "phase_end",
            f"Completed in {elapsed:.2f}s",
            component=phase.upper(),
            level="info",
            elapsed_seconds=elapsed,
            **(details or {})
        )

    def log_llm_call(
        self,
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed: float,
        model: str = "unknown"
    ) -> None:
        """Log LLM call with metrics."""
        self.log_event(
            "llm_call",
            f"LLM call in {phase}: {prompt_tokens + completion_tokens} tokens",
            component=phase.upper(),
            level="info",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            model=model,
        )

    def log_tool_call(
        self,
        tool_name: str,
        duration: float,
        output_size: int,
        worker_id: str = "unknown"
    ) -> None:
        """Log tool call."""
        self.log_event(
            "tool_call",
            f"Tool {tool_name} completed in {duration:.2f}s",
            component="TOOL",
            level="info",
            tool_name=tool_name,
            duration_seconds=duration,
            output_size=output_size,
            worker_id=worker_id,
        )

    def log_error(
        self,
        error: Exception,
        component: str,
        context: str = "",
        **data
    ) -> None:
        """Log error with exception details."""
        self.log_event(
            "error",
            f"{type(error).__name__}: {error}",
            component=component.upper(),
            level="error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **data
        )

    def get_entries(
        self,
        component: Optional[str] = None,
        level: Optional[str] = None,
        event: Optional[str] = None
    ) -> List[StructuredLogEntry]:
        """Get log entries with optional filtering.

        Args:
            component: Filter by component name
            level: Filter by log level
            event: Filter by event name

        Returns:
            List of matching log entries
        """
        entries = self._entries
        if component:
            entries = [e for e in entries if e.component == component.upper()]
        if level:
            entries = [e for e in entries if e.level == level.upper()]
        if event:
            entries = [e for e in entries if e.event == event]
        return entries

    def export_json(self) -> str:
        """Export all logs as JSON array."""
        return json.dumps([asdict(e) for e in self._entries], indent=2)

    def export_text(self) -> str:
        """Export all logs as text."""
        return "\n".join(e.to_text() for e in self._entries)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core logging
    "get_logger",
    "safe_get_logger",
    "LogFormat",
    "LogLevel",
    # Component loggers
    "get_worker_logger",
    "get_boss_logger",
    "get_lab_logger",
    "get_lab_architect_logger",
    "get_critic_logger",
    "get_synthesis_logger",
    "get_flash_explorer_logger",
    "get_web_agent_logger",
    # RunLogger helpers
    "log_component",
    "log_component_warning",
    "log_component_error",
    "log_phase_start",
    "log_phase_end",
    "log_error_with_context",
    # Convenience
    "log_exception",
    "log_with_context",
    # Timing
    "format_timing",
    "format_timing_breakdown",
    "TimingCheckpoint",
    "OperationTimer",
    "time_operation",
    "log_timing_breakdown",
    "ProfilingLogger",
    # Structured logging
    "StructuredLogEntry",
    "DeepResearchLogger",
]

