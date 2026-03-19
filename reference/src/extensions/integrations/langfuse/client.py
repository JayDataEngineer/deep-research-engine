"""Simplified LangFuse client management.

With LiteLLM integration, LangFuse tracing is automatic. This module
only provides client access for manual score additions.

LiteLLM handles:
- Automatic Langfuse tracing (no callbacks needed)
- Session tracking via metadata

This module provides:
- Client access for adding evaluation scores
- Health check utilities
"""

import os
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Cache the client to avoid repeated initialization
_cached_client: Optional["Langfuse"] = None
_client_error: Optional[Exception] = None


def get_langfuse_client() -> Optional["Langfuse"]:
    """Get or create LangFuse client singleton.

    Returns the cached client if available, otherwise creates a new one.
    Returns None if LangFuse is not configured or disabled.

    Returns:
        Langfuse client instance or None if not configured
    """
    global _cached_client, _client_error

    # Return cached client if available
    if _cached_client is not None:
        return _cached_client

    # Return None if we already tried and failed
    if _client_error is not None:
        return None

    # Check if explicitly disabled
    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false":
        logger.debug("langfuse_disabled_by_config")
        return None

    # Get credentials from environment
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Check if credentials are configured
    if not secret_key or not public_key:
        logger.debug(
            "langfuse_not_configured",
            has_secret_key=bool(secret_key),
            has_public_key=bool(public_key),
        )
        return None

    try:
        from langfuse import Langfuse

        _cached_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )

        logger.info(
            "langfuse_client_initialized",
            host=host,
        )

        return _cached_client

    except Exception as e:
        _client_error = e
        logger.warning(
            "langfuse_client_init_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


def is_langfuse_enabled() -> bool:
    """Check if LangFuse is configured and enabled.

    Returns:
        True if LangFuse client is available, False otherwise
    """
    return get_langfuse_client() is not None


def flush_langfuse_client() -> bool:
    """Flush pending LangFuse events.

    Call this before program exit to ensure all events are sent.

    Returns:
        True if flush succeeded, False otherwise
    """
    client = get_langfuse_client()
    if client is None:
        return False

    try:
        client.flush()
        logger.debug("langfuse_flushed")
        return True
    except Exception as e:
        logger.warning(
            "langfuse_flush_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


def shutdown_langfuse_client() -> bool:
    """Shutdown the LangFuse client and flush all events.

    Call this when the application is exiting to ensure all
    events are sent to LangFuse.

    Returns:
        True if shutdown succeeded, False otherwise
    """
    global _cached_client, _client_error

    client = _cached_client
    if client is None:
        return False

    try:
        client.shutdown()
        logger.debug("langfuse_shutdown_complete")
        return True
    except Exception as e:
        logger.warning(
            "langfuse_shutdown_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return False
    finally:
        # Clear the cache after shutdown
        _cached_client = None
        _client_error = None


def reset_langfuse_client() -> None:
    """Reset the cached client.

    Useful for testing or when credentials change.
    """
    global _cached_client, _client_error
    _cached_client = None
    _client_error = None
    logger.debug("langfuse_client_reset")
