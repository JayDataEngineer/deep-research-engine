"""
Observability tools for Context Engine.

Provides @tool decorated functions for logging, metrics, and run history.
These tools enable deep_research to use context_engine for all observability
instead of direct file I/O.

Tools:
    - log_observation: Write log entries to context_engine storage
    - save_metrics: Save run metrics as artifacts
    - get_run_history: Get historical runs for a session
    - get_logs: Retrieve logs for debugging

Usage from deep_research:
    from src.shared.tools.registry import get_tool

    log_observation = get_tool("log_observation")
    await log_observation.ainvoke({
        "session_id": session_id,
        "log_type": "app",
        "content": "[SCOUT] Starting research..."
    })
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING TOOLS
# =============================================================================

@tool
def log_observation(
    session_id: str,
    log_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    runtime: ToolRuntime = None,
) -> str:
    """Log an observation to context engine storage.

    Use this for all logging from deep_research instead of direct file I/O.
    Logs are stored in MinIO/S3Backend with session isolation.

    Args:
        session_id: Research session ID for isolation
        log_type: Type of log ("app", "llm_calls", "metrics", "diagnostics")
        content: Log entry content
        metadata: Optional metadata dict
        runtime: ToolRuntime for access to context engine

    Returns:
        Path where log was stored

    Example:
        >>> log_observation(session_id, "app", "[SCOUT] Starting research...")
        "/logs/session_123/app.log"
    """
    # Get context engine from state or create direct connection
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    # Fallback: direct S3Backend connection
    if not context_engine or not context_engine.s3_backend:
        from ..storage.s3_backend import S3Backend

        s3_backend = S3Backend(
            bucket=os.getenv("MINIO_BUCKET", "context-engine"),
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:8899/minio"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin")),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")),
        )
    else:
        s3_backend = context_engine.s3_backend

    try:
        # Build log path with session isolation
        path = f"/logs/{session_id}/{log_type}.log"

        # Add timestamp and format entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {content}\n"

        # Append to log file
        result = s3_backend.append(path, entry)

        if "error" in result:
            logger.warning(f"Log write failed: {result['error']}")
            return f"Error: {result['error']}"

        return path

    except Exception as e:
        logger.error(f"Failed to log observation: {e}")
        return f"Error logging: {str(e)}"


@tool
def save_metrics(
    session_id: str,
    metrics: Dict[str, Any],
    runtime: ToolRuntime = None,
) -> str:
    """Save run metrics as a Context Engine artifact.

    Use this to persist research run metrics (tokens, timing, quality scores)
    to context_engine for later analysis and comparison.

    Args:
        session_id: Research session ID
        metrics: Dictionary of metrics to save
        runtime: ToolRuntime for access to context engine

    Returns:
        Artifact ID or path where metrics were saved

    Example:
        >>> save_metrics(session_id, {
        ...     "total_tokens": 5000,
        ...     "total_time": 45.2,
        ...     "quality_score": 0.85
        ... })
        "metrics_session_123"
    """
    # Get context engine from state or create direct connection
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    try:
        # Add timestamp
        metrics["saved_at"] = datetime.now().isoformat()

        # Path for metrics JSON
        path = f"/logs/{session_id}/metrics.json"
        content = json.dumps(metrics, indent=2, default=str)

        if context_engine and context_engine.s3_backend:
            result = context_engine.s3_backend.write(path, content)
        else:
            from ..storage.s3_backend import S3Backend
            s3_backend = S3Backend(
                bucket=os.getenv("MINIO_BUCKET", "context-engine"),
                endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:8899/minio"),
                aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin")),
                aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")),
            )
            result = s3_backend.write(path, content)

        if "error" in result:
            logger.warning(f"Metrics save failed: {result['error']}")
            return f"Error: {result['error']}"

        return path

    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        return f"Error saving metrics: {str(e)}"


@tool
def get_logs(
    session_id: str,
    log_type: str = "app",
    limit: int = 100,
    runtime: ToolRuntime = None,
) -> str:
    """Retrieve logs from context engine storage.

    Use this to read back logs for debugging or analysis.

    Args:
        session_id: Research session ID
        log_type: Type of log ("app", "llm_calls", "metrics", "diagnostics")
        limit: Maximum number of lines to return
        runtime: ToolRuntime for access to context engine

    Returns:
        Log content as string

    Example:
        >>> get_logs(session_id, "app", limit=50)
        "[2024-01-15 10:30:00] [SCOUT] Starting...\\n..."
    """
    # Get context engine from state or create direct connection
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    if not context_engine or not context_engine.s3_backend:
        from ..storage.s3_backend import S3Backend
        s3_backend = S3Backend(
            bucket=os.getenv("MINIO_BUCKET", "context-engine"),
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:8899/minio"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin")),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")),
        )
    else:
        s3_backend = context_engine.s3_backend

    try:
        path = f"/logs/{session_id}/{log_type}.log"
        content = s3_backend.read(path, limit=limit)
        return content

    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        return f"Error reading logs: {str(e)}"


@tool
async def get_run_history(
    session_id: str,
    runtime: ToolRuntime = None,
) -> str:
    """Get historical run data from context engine.

    Use this to retrieve metrics and summaries from previous runs
    for comparison or learning.

    Args:
        session_id: Research session ID
        runtime: ToolRuntime for access to context engine

    Returns:
        JSON string with run history data

    Example:
        >>> get_run_history(session_id)
        '{"metrics": {...}, "artifacts": [...], "logs": [...]}'
    """
    # Get context engine from state or create direct connection
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    if not context_engine:
        return "Error: Context Engine not available"

    try:
        history = {
            "session_id": session_id,
            "artifacts": [],
            "metrics": None,
            "logs": []
        }

        # Get artifacts from Postgres
        if context_engine.postgres_store:
            artifacts = await context_engine.postgres_store.list_by_session(session_id)
            history["artifacts"] = [
                {
                    "id": a.id,
                    "type": a.artifact_type.value if hasattr(a.artifact_type, 'value') else str(a.artifact_type),
                    "created_at": str(a.created_at) if a.created_at else None,
                }
                for a in artifacts[:20]  # Limit to 20
            ]

        # Get metrics from S3
        if context_engine.s3_backend:
            try:
                metrics_content = context_engine.s3_backend.read(f"/logs/{session_id}/metrics.json")
                history["metrics"] = json.loads(metrics_content)
            except:
                pass  # Metrics may not exist

            # List available logs
            try:
                log_files = context_engine.s3_backend.ls(f"/logs/{session_id}/")
                history["logs"] = [f["name"] for f in log_files if f.get("name", "").endswith(".log")]
            except:
                pass

        return json.dumps(history, indent=2, default=str)

    except Exception as e:
        logger.error(f"Failed to get run history: {e}")
        return f"Error getting history: {str(e)}"


@tool
def save_output(
    session_id: str,
    output_type: str,
    content: str,
    filename: Optional[str] = None,
    runtime: ToolRuntime = None,
) -> str:
    """Save research output to context engine storage.

    Use this to save final outputs like output.md, worker findings, etc.

    Args:
        session_id: Research session ID
        output_type: Type of output ("final", "draft", "worker_findings")
        content: Content to save
        filename: Optional custom filename
        runtime: ToolRuntime for access to context engine

    Returns:
        Path where output was saved

    Example:
        >>> save_output(session_id, "final", markdown_content, "output.md")
        "/outputs/session_123/final/output.md"
    """
    # Get context engine from state or create direct connection
    context_engine = None
    if runtime:
        context_engine = runtime.state.get("_context_engine")

    if not context_engine or not context_engine.s3_backend:
        from ..storage.s3_backend import S3Backend
        s3_backend = S3Backend(
            bucket=os.getenv("MINIO_BUCKET", "context-engine"),
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:8899/minio"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin")),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")),
        )
    else:
        s3_backend = context_engine.s3_backend

    try:
        # Build path
        if filename:
            safe_filename = filename.replace("/", "_").replace("..", "_")
        else:
            safe_filename = f"{output_type}.md"

        path = f"/outputs/{session_id}/{output_type}/{safe_filename}"

        # Write content
        result = s3_backend.write(path, content)

        if "error" in result:
            logger.warning(f"Output save failed: {result['error']}")
            return f"Error: {result['error']}"

        return path

    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        return f"Error saving output: {str(e)}"


# =============================================================================
# TOOL LISTS FOR REGISTRY
# =============================================================================

CONTEXT_ENGINE_OBSERVABILITY_TOOLS = [
    log_observation,
    save_metrics,
    get_logs,
    get_run_history,
    save_output,
]


__all__ = [
    "log_observation",
    "save_metrics",
    "get_logs",
    "get_run_history",
    "save_output",
    "CONTEXT_ENGINE_OBSERVABILITY_TOOLS",
]
