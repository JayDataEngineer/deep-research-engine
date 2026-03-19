"""FlashExplorerLogger - Simple logger wrapper using central logging configuration.

This is a lightweight wrapper that uses the centralized logging configuration
from logging_utils.py for consistent formatting and level control.

NOTE: This logger is primarily used for run_logger file output (writing to
run-specific log files), not console logging. For console logging, use the
standard logger from logging_utils.get_logger().
"""
from typing import Any, Optional
from pathlib import Path

# Import central logging configuration
from src.shared.logging_config import get_logger as get_central_logger


class FlashExplorerLogger:
    """Logger wrapper for flash_explorer component with file output support.

    This provides two capabilities:
    1. Standard logging via central configuration (console/files as configured)
    2. Run-specific file logging via _write_to_file() for observability

    The file logging is used by the observability system to write run-specific
    logs to files like app.log and llm_calls.md.

    Usage:
        logger = FlashExplorerLogger("flash_explorer")
        logger.info("Message")  # Goes to central logging
        logger._write_to_file("Raw message")  # Goes to run-specific file
    """

    def __init__(self, name: str, run_output_dir: Optional[Path] = None):
        """Initialize the logger.

        Args:
            name: Logger name (typically __name__ or component name)
            run_output_dir: Optional directory for run-specific file output
        """
        self.name = name
        # Use central logging configuration for console/file logging
        self.logger = get_central_logger(name)
        self._run_output_dir = run_output_dir
        self._log_file = None
        self._session_id = None  # Set via setter for context_engine integration

        if run_output_dir:
            self._log_file = run_output_dir / "explorer.log"

    def set_session_id(self, session_id: str) -> None:
        """Set session_id for context_engine integration.

        Args:
            session_id: Research session ID
        """
        self._session_id = session_id

    def _write_to_file(self, message: str) -> None:
        """Write a message to context_engine storage via log_observation tool.

        NO direct file I/O - all logging goes through context_engine.
        Falls back to local file only if context_engine is unavailable.

        Args:
            message: The message to write (will have newline appended)
        """
        # Try context_engine log_observation tool first
        if self._session_id:
            try:
                from src.shared.tools.registry import get_tool
                log_observation = get_tool("log_observation")
                if log_observation:
                    # Use context_engine tool
                    log_observation.invoke({
                        "session_id": self._session_id,
                        "log_type": "flash_explorer",
                        "content": message,
                        "metadata": {"logger": self.name}
                    })
                    return  # Success - skip local file
            except Exception:
                pass  # Fall through to local file

        # Fallback: Local file (only if context_engine unavailable)
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"{message}\n")
            except Exception:
                pass  # Don't break execution if file logging fails

    # Delegate standard logging methods to central logger
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)
