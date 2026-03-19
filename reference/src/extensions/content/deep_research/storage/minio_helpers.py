"""MinIO backend helpers for deep_research.

This module provides centralized access to MinioBackend instances,
replacing direct S3Backend usage with the cleaner MinioBackend API
that includes grep/glob/ls/edit functionality.

Usage:
    from ..storage.minio_helpers import (
        get_minio_backend,
        get_web_session_backend,
        get_dumps_backend,
        get_artifacts_backend,
    )

    # Get backend for a specific purpose
    backend = get_web_session_backend("session_123")
    backend.write("/page.html", html_content)
    matches = backend.grep_raw("error", glob="*.html")
"""

import os
from typing import Optional

from src.extensions.features.tools.context_engine.storage.minio_backend import (
    MinioBackend,
    get_minio_backend_for_session,
)

# MinIO configuration defaults
DEFAULT_MINIO_BUCKET = os.getenv("MINIO_BUCKET", "context-engine")
DEFAULT_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", os.getenv("MINIO_ENDPOINT_URL", "http://localhost:8899/minio"))
DEFAULT_MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin"))
DEFAULT_MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"))


def get_minio_backend(
    prefix: str = "",
    session_id: Optional[str] = None,
    bucket: Optional[str] = None,
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> MinioBackend:
    """Get a MinioBackend instance with configuration.

    Args:
        prefix: Path prefix (e.g., "web/session", "dumps/session", "raw")
        session_id: Session ID for isolation
        bucket: MinIO bucket (default: from env)
        endpoint: MinIO endpoint (default: from env)
        access_key: MinIO access key (default: from env)
        secret_key: MinIO secret key (default: from env)

    Returns:
        MinioBackend instance
    """
    return MinioBackend(
        prefix=prefix,
        session_id=session_id,
        bucket=bucket or DEFAULT_MINIO_BUCKET,
        endpoint=endpoint or DEFAULT_MINIO_ENDPOINT,
        access_key=access_key or DEFAULT_MINIO_ACCESS_KEY,
        secret_key=secret_key or DEFAULT_MINIO_SECRET_KEY,
    )


def get_web_session_backend(session_id: str) -> MinioBackend:
    """Get MinioBackend for web session data.

    Path structure: /web/session/{session_id}/

    Use for:
    - Scraped HTML pages
    - API responses (JSON)
    - Screenshots and media

    Args:
        session_id: Research session ID

    Returns:
        MinioBackend instance with web/session prefix
    """
    return get_minio_backend(prefix="web/session", session_id=session_id)


def get_dumps_backend(session_id: str) -> MinioBackend:
    """Get MinioBackend for context dumps.

    Path structure: /dumps/session/{session_id}/

    Use for:
    - Agent state snapshots
    - Checkpoint files
    - Conversation summaries
    - Memory dumps

    Args:
        session_id: Research session ID

    Returns:
        MinioBackend instance with dumps/session prefix
    """
    return get_minio_backend(prefix="dumps/session", session_id=session_id)


def get_artifacts_backend(session_id: str) -> MinioBackend:
    """Get MinioBackend for research artifacts.

    Path structure: /raw/{session_id}/

    Use for:
    - Worker findings (large files)
    - Architect findings
    - Boss decisions
    - Critic evaluations
    - Synthesis drafts

    Args:
        session_id: Research session ID

    Returns:
        MinioBackend instance with raw prefix
    """
    return get_minio_backend(prefix="raw", session_id=session_id)


def get_outputs_backend(session_id: str) -> MinioBackend:
    """Get MinioBackend for research outputs.

    Path structure: /outputs/{session_id}/

    Use for:
    - worker_outputs.json
    - raw_sources.json
    - output.md

    Args:
        session_id: Research session ID

    Returns:
        MinioBackend instance with outputs prefix
    """
    return get_minio_backend(prefix="outputs", session_id=session_id)


def get_logs_backend(session_id: str) -> MinioBackend:
    """Get MinioBackend for logs.

    Path structure: /logs/{session_id}/

    Use for:
    - app.log
    - llm_calls.log
    - metrics.json
    - diagnostics.log

    Args:
        session_id: Research session ID

    Returns:
        MinioBackend instance with logs prefix
    """
    return get_minio_backend(prefix="logs", session_id=session_id)


# Legacy compatibility - raw MinioBackend without session isolation
def get_raw_minio_backend() -> MinioBackend:
    """Get MinioBackend for raw file access (no session isolation).

    Use this for admin/maintenance operations that need to access
    files across all sessions.

    Returns:
        MinioBackend instance without prefix
    """
    return get_minio_backend(prefix="")


__all__ = [
    "get_minio_backend",
    "get_web_session_backend",
    "get_dumps_backend",
    "get_artifacts_backend",
    "get_outputs_backend",
    "get_logs_backend",
    "get_raw_minio_backend",
    # Configuration defaults
    "DEFAULT_MINIO_BUCKET",
    "DEFAULT_MINIO_ENDPOINT",
    "DEFAULT_MINIO_ACCESS_KEY",
    "DEFAULT_MINIO_SECRET_KEY",
]
