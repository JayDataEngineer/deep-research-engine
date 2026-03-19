"""Standardized logging utilities.

Provides consistent logging patterns and helper functions for:
- Request/response logging
- Error logging with context
- Operation logging
- Extension lifecycle logging
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Optional
import json as json_module

# Get log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


# =============================================================================
# Standardized Log Tags
# =============================================================================

class LogTag:
    """Standard log tags for consistent log message prefixes."""
    # API endpoints
    API = "API"
    CHAT = "CHAT"
    THREADS = "THREADS"
    MESSAGES = "MESSAGES"
    
    # Extension operations
    EXTENSION = "EXTENSION"
    COMPANION = "COMPANION"
    ROLEPLAY = "ROLEPLAY"
    STANDARD = "STANDARD"
    DEEP_RESEARCH = "DEEP_RESEARCH"
    TTS = "TTS"
    
    # Operations
    VALIDATION = "VALIDATION"
    DATABASE = "DATABASE"
    CHECKPOINTER = "CHECKPOINTER"
    LLM = "LLM"
    TOOLS = "TOOLS"
    
    # Media
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    
    # System
    SYSTEM = "SYSTEM"
    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"


# =============================================================================
# Log Message Helpers
# =============================================================================

def log_request(logger: logging.Logger, method: str, path: str, **kwargs) -> None:
    """Log an incoming request with standard format.
    
    Args:
        logger: Logger instance
        method: HTTP method (GET, POST, etc.)
        path: Request path
        **kwargs: Additional context (query params, etc.)
    """
    extra = f" {kwargs}" if kwargs else ""
    logger.info(f"[{LogTag.API}] {method} {path}{extra}")


def log_response(logger: logging.Logger, method: str, path: str, 
                 status_code: int, duration_ms: float) -> None:
    """Log a response with standard format.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
    """
    logger.info(f"[{LogTag.API}] {method} {path} -> {status_code} ({duration_ms:.0f}ms)")


def log_operation(logger: logging.Logger, operation: str, **context) -> None:
    """Log an operation with context.
    
    Args:
        logger: Logger instance
        operation: Operation name
        **context: Additional context (thread_id, user_id, etc.)
    """
    ctx_str = f" {context}" if context else ""
    logger.info(f"[{operation}]{ctx_str}")


def log_operation_success(logger: logging.Logger, operation: str, **context) -> None:
    """Log a successful operation.
    
    Args:
        logger: Logger instance
        operation: Operation name
        **context: Additional context
    """
    ctx_str = f" {context}" if context else ""
    logger.info(f"[{operation}] Success{ctx_str}")


def log_operation_error(logger: logging.Logger, operation: str, error: Exception, **context) -> None:
    """Log an operation error.
    
    Args:
        logger: Logger instance
        operation: Operation name
        error: The exception
        **context: Additional context
    """
    ctx_str = f" {context}" if context else ""
    logger.error(f"[{operation}] Failed{ctx_str}: {error}")


def log_validation_error(logger: logging.Logger, field: str, message: str) -> None:
    """Log a validation error.
    
    Args:
        logger: Logger instance
        field: Field that failed validation
        message: Error message
    """
    logger.warning(f"[{LogTag.VALIDATION}] {field}: {message}")


def log_database_query(logger: logging.Logger, query_type: str, table: str, 
                      count: Optional[int] = None, **context) -> None:
    """Log a database operation.
    
    Args:
        logger: Logger instance
        query_type: Type of query (SELECT, INSERT, UPDATE, DELETE)
        table: Table name
        count: Number of rows affected
        **context: Additional context
    """
    count_str = f" {count} rows" if count is not None else ""
    ctx_str = f" {context}" if context else ""
    logger.debug(f"[{LogTag.DATABASE}] {query_type} {table}{count_str}{ctx_str}")


def log_extension_event(logger: logging.Logger, extension: str, event: str, **details) -> None:
    """Log an extension lifecycle event.
    
    Args:
        logger: Logger instance
        extension: Extension name
        event: Event type (loaded, unloaded, error, etc.)
        **details: Additional details
    """
    logger.info(f"[{LogTag.EXTENSION}] {extension}: {event}{' ' + str(details) if details else ''}")


# =============================================================================
# Timing Context Managers
# =============================================================================

@contextmanager
def log_duration(logger: logging.Logger, operation: str, level: int = INFO):
    """Context manager to log operation duration.
    
    Args:
        logger: Logger instance
        operation: Operation name
        level: Log level to use (default: INFO)
        
    Yields:
        None
    """
    start = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start) * 1000
        logger.log(level, f"[{operation}] Completed in {duration_ms:.0f}ms")


@contextmanager
def log_request_context(logger: logging.Logger, method: str, path: str):
    """Context manager to log request/response with timing.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        
    Yields:
        None
    """
    log_request(logger, method, path)
    start = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start) * 1000
        # Response status will be logged separately
        logger.debug(f"[{LogTag.API}] {method} {path} completed in {duration_ms:.0f}ms")


# =============================================================================
# Data Logging Helpers
# =============================================================================

def log_data(logger: logging.Logger, tag: str, data: Any, 
               max_length: int = 500, truncate: bool = True) -> None:
    """Log data with optional truncation.
    
    Args:
        logger: Logger instance
        tag: Log tag for categorization
        data: Data to log (will be JSON serialized)
        max_length: Maximum length before truncation
        truncate: Whether to truncate long data
    """
    try:
        if isinstance(data, (dict, list)):
            json_str = json_module.dumps(data, indent=None, ensure_ascii=False, default=str)
        else:
            json_str = str(data)
            
        if truncate and len(json_str) > max_length:
            json_str = json_str[:max_length] + f"... (truncated from {len(json_str)} chars)"
        
        logger.debug(f"[{tag}] {json_str}")
    except Exception as e:
        logger.warning(f"[{tag}] Could not serialize data for logging: {e}")


def log_large_data(logger: logging.Logger, tag: str, data: Any, 
                   max_length: int = 2000) -> None:
    """Log potentially large data with safe truncation.
    
    Args:
        logger: Logger instance
        tag: Log tag
        data: Data to log
        max_length: Maximum length
    """
    log_data(logger, tag, data, max_length=max_length, truncate=True)


# =============================================================================
# Extension Logger Factory
# =============================================================================

def get_extension_logger(name: str) -> logging.Logger:
    """Get a logger for an extension with standardized naming.
    
    Args:
        name: Extension name (e.g., "companion", "roleplay")
        
    Returns:
        Logger instance with name formatted as "extensions.{name}"
    """
    return logging.getLogger(f"extensions.{name}")


def get_api_logger(name: str) -> logging.Logger:
    """Get a logger for an API router with standardized naming.
    
    Args:
        name: Router name (e.g., "threads", "messages")
        
    Returns:
        Logger instance with name formatted as "api.{name}"
    """
    return logging.getLogger(f"api.{name}")


def get_tool_logger(name: str) -> logging.Logger:
    """Get a logger for a tool with standardized naming.
    
    Args:
        name: Tool name
        
    Returns:
        Logger instance with name formatted as "tools.{name}"
    """
    return logging.getLogger(f"tools.{name}")
