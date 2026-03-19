"""Logging module - Loguru + Rich."""

from .rich_logging import (
    get_logger,
    log_stage,
    log_critic_review,
    log_error,
    setup_logging,
)

__all__ = [
    "get_logger",
    "log_stage",
    "log_critic_review",
    "log_error",
    "setup_logging",
]
