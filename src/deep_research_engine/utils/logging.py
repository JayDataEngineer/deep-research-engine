"""Logging utilities."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "info",
    format: Optional[str] = None,
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Log level (debug, info, warning, error)
        format: Optional custom format
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=format,
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
