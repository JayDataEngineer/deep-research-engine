"""Artifact path helpers for deep research.

This module provides path generation functions that work with context_engine's S3Backend.
These functions replace the path generation methods from WorkspaceBackend.

All paths use session_id for isolation:
    /raw/{session_id}/workers/iter_{iteration:03d}/{worker_id}.txt
    /raw/{session_id}/worker/iter_{iteration:03d}/brief.md
    /raw/{session_id}/boss/iter_{iteration:03d}/decision.md
    etc.

New paths for deep_agents:
    /web/session/{session_id}/{data_type}/{filename}
    /dumps/session/{session_id}/{dump_type}/{filename}

Usage with context_engine:
    from ..helpers.artifact_paths import get_worker_artifact_path

    path = get_worker_artifact_path(session_id, "researcher_0", iteration=0)
    context_engine.s3_backend.write(path, content)
"""

from typing import Optional


def get_worker_artifact_path(
    session_id: str,
    worker_id: str,
    iteration: int,
    artifact_type: str = "findings"
) -> str:
    """Get the standard path for a worker artifact.

    Args:
        session_id: Research session ID
        worker_id: Worker identifier
        iteration: Iteration number
        artifact_type: Type of artifact (findings, brief, etc.)

    Returns:
        S3Backend-compatible path
    """
    safe_worker_id = worker_id.replace("/", "_").replace("..", "_")
    return f"/raw/{session_id}/workers/iter_{iteration:03d}/{safe_worker_id}.txt"


def get_worker_brief_path(
    session_id: str,
    iteration: int
) -> str:
    """Get the standard path for worker brief.

    Args:
        session_id: Research session ID
        iteration: Iteration number

    Returns:
        S3Backend-compatible path
    """
    return f"/raw/{session_id}/worker/iter_{iteration:03d}/brief.md"


def get_boss_decision_path(
    session_id: str,
    iteration: int
) -> str:
    """Get the standard path for boss decision.

    Args:
        session_id: Research session ID
        iteration: Iteration number

    Returns:
        S3Backend-compatible path
    """
    return f"/raw/{session_id}/boss/iter_{iteration:03d}/decision.md"


def get_critic_evaluation_path(
    session_id: str,
    iteration: int
) -> str:
    """Get the standard path for critic evaluation.

    Args:
        session_id: Research session ID
        iteration: Iteration number

    Returns:
        S3Backend-compatible path
    """
    return f"/raw/{session_id}/critic/iter_{iteration:03d}/evaluation.md"


def get_synthesis_draft_path(
    session_id: str,
    iteration: int
) -> str:
    """Get the standard path for synthesis draft.

    Args:
        session_id: Research session ID
        iteration: Iteration number

    Returns:
        S3Backend-compatible path
    """
    return f"/raw/{session_id}/synthesis/iter_{iteration:03d}/draft.md"


def get_synthesis_final_path(
    session_id: str,
    iteration: int
) -> str:
    """Get the standard path for final synthesis output.

    Args:
        session_id: Research session ID
        iteration: Iteration number

    Returns:
        S3Backend-compatible path
    """
    return f"/raw/{session_id}/synthesis/iter_{iteration:03d}/final.md"


def get_output_path(
    session_id: str,
    name: str = "output.md"
) -> str:
    """Get the standard path for final output.

    Args:
        session_id: Research session ID
        name: Output filename

    Returns:
        S3Backend-compatible path
    """
    safe_name = name.replace("/", "_").replace("..", "_")
    return f"/raw/{session_id}/output/{safe_name}"


def get_synthesis_state_path(
    session_id: str,
    stage: str
) -> str:
    """Get the standard path for synthesis state artifact.

    Args:
        session_id: Research session ID
        stage: Stage identifier (e.g., "before_lazy_loading", "final")

    Returns:
        S3Backend-compatible path
    """
    safe_stage = stage.replace("/", "_").replace("..", "_")
    return f"/raw/{session_id}/synthesis/state_{safe_stage}.md"


def get_critic_state_path(
    session_id: str,
    stage: str
) -> str:
    """Get the standard path for critic state artifact.

    Args:
        session_id: Research session ID
        stage: Stage identifier (e.g., "before_evaluation", "final")

    Returns:
        S3Backend-compatible path
    """
    safe_stage = stage.replace("/", "_").replace("..", "_")
    return f"/raw/{session_id}/critic/state_{safe_stage}.md"


def get_worker_summary_path(
    session_id: str,
    worker_id: str,
    iteration: int
) -> str:
    """Get the standard path for worker summary artifact (Map-Reduce).

    Args:
        session_id: Research session ID
        worker_id: Worker identifier
        iteration: Iteration number

    Returns:
        S3Backend-compatible path
    """
    safe_worker_id = worker_id.replace("/", "_").replace("..", "_")
    return f"/raw/{session_id}/summaries/iter_{iteration:03d}/{safe_worker_id}_summary.md"


# =============================================================================
# WEB SESSION PATHS (for deep_agents)
# =============================================================================

def get_web_session_path(
    session_id: str,
    data_type: str,
    filename: str
) -> str:
    """Get the standard path for web session data (deep_agents).

    This path structure is used by deep_agents for storing web-related
    session data such as:
    - HTML pages scraped during research
    - JSON data from API calls
    - Screenshots or other media
    - Intermediate state from web tools

    Args:
        session_id: Research session ID
        data_type: Type of web data (html, json, screenshots, etc.)
        filename: Filename for the data

    Returns:
        S3Backend-compatible path
    """
    safe_data_type = data_type.replace("/", "_").replace("..", "_")
    safe_filename = filename.replace("/", "_").replace("..", "_")
    return f"/web/session/{session_id}/{safe_data_type}/{safe_filename}"


def get_web_session_html_path(
    session_id: str,
    url_hash: str
) -> str:
    """Get the standard path for web session HTML data.

    Args:
        session_id: Research session ID
        url_hash: Hash of the URL (for deduplication)

    Returns:
        S3Backend-compatible path
    """
    safe_hash = url_hash.replace("/", "_").replace("..", "_")
    return f"/web/session/{session_id}/html/{safe_hash}.html"


def get_web_session_json_path(
    session_id: str,
    source: str,
    data_id: str
) -> str:
    """Get the standard path for web session JSON data.

    Args:
        session_id: Research session ID
        source: Source identifier (api, scrape, etc.)
        data_id: Unique identifier for the data

    Returns:
        S3Backend-compatible path
    """
    safe_source = source.replace("/", "_").replace("..", "_")
    safe_id = data_id.replace("/", "_").replace("..", "_")
    return f"/web/session/{session_id}/json/{safe_source}_{safe_id}.json"


# =============================================================================
# CONTEXT DUMP PATHS (for deep_agents state management)
# =============================================================================

def get_context_dump_path(
    session_id: str,
    dump_type: str,
    filename: str
) -> str:
    """Get the standard path for context dumps (deep_agents state management).

    This path structure is used by deep_agents for storing context dumps
    for extra state management such as:
    - Agent state snapshots
    - Artifact checkpoints
    - Conversation summaries
    - Tool call histories
    - Memory dumps

    Args:
        session_id: Research session ID
        dump_type: Type of dump (state, artifact, checkpoint, summary, etc.)
        filename: Filename for the dump

    Returns:
        S3Backend-compatible path
    """
    safe_dump_type = dump_type.replace("/", "_").replace("..", "_")
    safe_filename = filename.replace("/", "_").replace("..", "_")
    return f"/dumps/session/{session_id}/{safe_dump_type}/{safe_filename}"


def get_state_dump_path(
    session_id: str,
    step: int,
    agent_name: str = "main"
) -> str:
    """Get the standard path for agent state dumps.

    Args:
        session_id: Research session ID
        step: Step number in the agent's execution
        agent_name: Name of the agent (for multi-agent systems)

    Returns:
        S3Backend-compatible path
    """
    safe_agent = agent_name.replace("/", "_").replace("..", "_")
    return f"/dumps/session/{session_id}/state/{safe_agent}_step_{step:04d}.json"


def get_checkpoint_dump_path(
    session_id: str,
    checkpoint_id: str
) -> str:
    """Get the standard path for checkpoint dumps.

    Args:
        session_id: Research session ID
        checkpoint_id: Unique checkpoint identifier

    Returns:
        S3Backend-compatible path
    """
    safe_checkpoint = checkpoint_id.replace("/", "_").replace("..", "_")
    return f"/dumps/session/{session_id}/checkpoint/{safe_checkpoint}.json"


def get_summary_dump_path(
    session_id: str,
    iteration: int
) -> str:
    """Get the standard path for conversation summary dumps.

    Args:
        session_id: Research session ID
        iteration: Iteration number

    Returns:
        S3Backend-compatible path
    """
    return f"/dumps/session/{session_id}/summary/iter_{iteration:04d}.json"


def get_memory_dump_path(
    session_id: str,
    memory_type: str
) -> str:
    """Get the standard path for memory dumps.

    Args:
        session_id: Research session ID
        memory_type: Type of memory (short_term, long_term, working, etc.)

    Returns:
        S3Backend-compatible path
    """
    safe_type = memory_type.replace("/", "_").replace("..", "_")
    return f"/dumps/session/{session_id}/memory/{safe_type}.json"


__all__ = [
    "get_worker_artifact_path",
    "get_worker_brief_path",
    "get_boss_decision_path",
    "get_critic_evaluation_path",
    "get_synthesis_draft_path",
    "get_synthesis_final_path",
    "get_output_path",
    "get_synthesis_state_path",
    "get_critic_state_path",
    "get_worker_summary_path",
    # Web session paths (for deep_agents)
    "get_web_session_path",
    "get_web_session_html_path",
    "get_web_session_json_path",
    # Context dump paths (for deep_agents state management)
    "get_context_dump_path",
    "get_state_dump_path",
    "get_checkpoint_dump_path",
    "get_summary_dump_path",
    "get_memory_dump_path",
]
