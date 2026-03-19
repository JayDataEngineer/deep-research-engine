"""Structured JSON logging configuration for the application.

This module provides a custom JSON formatter for structured logging,
making logs easier to parse and analyze with log aggregation tools.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any

# JSON serializable types
JsonSerializable = str | int | float | bool | None | dict[str, Any] | list[Any]


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON.

    Each log entry is a JSON object with the following fields:
    - timestamp: ISO 8601 timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Logger name
    - message: Log message
    - module: Module name
    - function: Function name
    - line: Line number
    - thread: Thread ID
    - process: Process ID
    - extra: Any additional fields from the log record
    - exception: Exception details if an exception occurred
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Create base log entry
        log_entry: dict[str, JsonSerializable] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_entry["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }

        # Add any extra fields from the log record
        # These are added via logger.info("msg", extra={"key": "value"})
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "exc_info", "exc_text", "stack_info",
            }:
                # Skip if already in log_entry or None
                if value is not None and key not in log_entry:
                    log_entry["extra"] = log_entry.get("extra", {})
                    log_entry["extra"][key] = value

        return json.dumps(log_entry, default=str)


def configure_logging(
    level: str | int = logging.INFO,
    format_json: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level (e.g., "INFO", logging.INFO)
        format_json: Whether to use JSON formatting (True) or text formatting (False)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Set formatter
    if format_json:
        handler.setFormatter(JSONFormatter())
    else:
        # Use a simple text format for development
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Log helpers for common patterns
def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float | None = None,
    **extra: Any,
) -> None:
    """Log an HTTP request.

    Args:
        logger: Logger instance
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        **extra: Additional fields to include in the log
    """
    log_data = {
        "http_method": method,
        "http_path": path,
        "http_status": status_code,
        **extra,
    }
    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    logger.info("HTTP Request", extra=log_data)


def log_error(
    logger: logging.Logger,
    message: str,
    exception: Exception | None = None,
    **extra: Any,
) -> None:
    """Log an error with optional exception.

    Args:
        logger: Logger instance
        message: Error message
        exception: Exception that occurred
        **extra: Additional fields to include in the log
    """
    log_data = {"error_message": message, **extra}
    if exception:
        logger.error(message, exc_info=(type(exception), exception, exception.__traceback__), extra=log_data)
    else:
        logger.error(message, extra=log_data)
