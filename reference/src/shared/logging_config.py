"""Unified logging configuration for the application.

This module provides a SINGLE source of truth for all logging configuration.
It properly integrates structlog with standard library logging to avoid conflicts.

Key principles:
- ONE place to configure logging globally
- Proper structlog + stdlib integration
- Consistent format: 'YYYY-MM-DD HH:MM:SS [level] message'
- No duplicate log levels or timestamps
"""
import logging
import sys

# Track if we've already configured to prevent duplicate configuration
_configured = False


def configure_logging(level: int = logging.INFO, use_structlog: bool = True) -> None:
    """Configure logging with consistent format for all modules.

    This is the SINGLE point of logging configuration for the entire application.
    Call this once at startup in main.py before importing other modules.

    Args:
        level: Logging level (default: INFO)
        use_structlog: Whether to configure structlog integration (default: True)

    The format is: 'YYYY-MM-DD HH:MM:SS [level] message'
    Example: '2026-02-13 09:05:00 [INFO] Application started'
    """
    global _configured

    # Only configure once
    if _configured:
        return
    _configured = True

    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Define consistent format
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    if use_structlog:
        # Configure with structlog if available
        try:
            import structlog
            import structlog.stdlib

            # Configure structlog to work with stdlib logging
            # This prevents double-formatting issues
            structlog.configure(
                processors=[
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.add_logger_name,
                    structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                    # Don't add extra formatting - let stdlib handler do it
                    structlog.stdlib.render_to_log_kwargs,
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
                # IMPORTANT: Don't let structlog add its own handler
                # Use the stdlib handler we configure below
            )

            # Create stdlib handler with consistent format
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(log_format, date_format))
            root_logger.addHandler(handler)
            root_logger.setLevel(level)

        except ImportError:
            # structlog not available, fall back to stdlib only
            logging.basicConfig(
                level=level,
                format=log_format,
                datefmt=date_format,
                force=True
            )
        except Exception:
            # Any structlog configuration error, fall back to stdlib only
            logging.basicConfig(
                level=level,
                format=log_format,
                datefmt=date_format,
                force=True
            )
    else:
        # Standard logging configuration
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            force=True
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger that uses the shared logging configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        A logger that respects the shared logging configuration
    """
    return logging.getLogger(name)


def reset_configuration() -> None:
    """Reset logging configuration (for testing only).

    This clears all handlers and resets the configured flag.
    """
    global _configured
    _configured = False

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
