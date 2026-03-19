"""Formatter node - Post-synthesis content formatting.

This node runs AFTER synthesis and critic quality gate.
It transforms raw research findings into the target format
(tweets, podcast script, etc.) using the content_formatter extension.

CRASH-ONLY PHILOSOPHY: This node has NO fallbacks. Any error raises immediately.
If formatting fails, the raw synthesis is still available in final_response.

IMPORTANT: This is the ONLY place where final synthesis output is saved to disk.
Intermediate syntheses (during critic iterations) are NOT saved.
"""

import time
from typing import Dict, Any, Literal, Optional
from pathlib import Path

# Import shared types and helpers from common module
from .common import (
    GlobalState,
    ResearchRuntimeConfig,
    get_run_logger_from_config,
    get_yaml_config_from_context,
    require_run_logger,
)

# Import logging decorators for standardized phase logging
from ..decorators import research_node, Phase

from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.graph import END


# =============================================================================
# SYNTHESIS OUTPUT SAVING
# =============================================================================

# Default output directory for synthesis results
DEFAULT_OUTPUT_DIR = Path("data/outputs/deep_research")


def save_synthesis_output(
    session_id: str,
    query: str,
    response: str,
    elapsed: float,
    output_dir: Path = None,
    metadata: Dict[str, Any] = None,
) -> Path:
    """Save synthesis output to files.

    Creates a session directory with:
    - synthesis.md: The final synthesis content
    - synthesis.json: Metadata (query, elapsed, timestamp, etc.)

    Args:
        session_id: Session identifier
        query: Original research query
        response: Synthesis content
        elapsed: Time elapsed in seconds
        output_dir: Base output directory (default: data/outputs/deep_research)
        metadata: Additional metadata to include

    Returns:
        Path to the session directory
    """
    import json
    from datetime import datetime

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Create session directory
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save synthesis content as markdown
    synthesis_file = session_dir / "synthesis.md"
    with open(synthesis_file, "w", encoding="utf-8") as f:
        f.write(f"# Research Synthesis\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(f"**Session:** {session_id}\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Elapsed:** {elapsed:.2f}s\n\n")
        f.write("---\n\n")
        f.write(response)

    # Save metadata as JSON
    meta = {
        "session_id": session_id,
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "content_length": len(response),
        **(metadata or {}),
    }
    meta_file = session_dir / "synthesis.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return session_dir


@research_node(Phase.SYNTHESIS)  # Re-use SYNTHESIS phase since it's the final output phase
async def formatter_node(
    state: GlobalState,
    *,
    runtime: Optional[Runtime[ResearchRuntimeConfig]] = None,
    store: Optional[Any] = None,
    config: RunnableConfig | None = None,
) -> Command[Literal[END]]:
    """Format final research output using content_formatter extension.

    This node:
    1. Takes final_response (raw synthesis) from state
    2. Gets output format from config (tweets, podcast, etc.)
    3. Calls content_formatter to transform the content
    4. Stores result in formatted_response

    If formatting fails or is not needed, formatted_response remains None
    and the raw synthesis (final_response) is used.

    Args:
        state: Current global state
        runtime: Optional Runtime context
        store: Optional BaseStore
        config: LangGraph RunnableConfig (contains run_logger)

    Returns:
        Command with state update and goto=END

    Raises:
        RuntimeError: On critical errors (but NOT on formatting failures - those are graceful)
    """
    start_time = time.time()

    yaml_config = get_yaml_config_from_context()
    run_logger = get_run_logger_from_config(config)
    require_run_logger(run_logger, "formatter node")

    run_logger._write_to_file("[FORMATTER] Starting formatter node")

    # Extract key state variables (needed for saving synthesis)
    session_id = state.get("session_id")
    query = state.get("query", "")
    final_response = state.get("final_response")

    # Get output mode from config
    output_config = yaml_config.get("output", {})
    output_mode = output_config.get("mode", "text")

    # Check if formatting is needed
    # "text" mode means no formatting - just use raw synthesis
    if output_mode == "text" or not output_mode:
        run_logger._write_to_file("[FORMATTER] Output mode is 'text' - no formatting needed")

        # Save final synthesis (raw synthesis since no formatting)
        if final_response and session_id:
            try:
                elapsed = time.time() - start_time
                output_path = save_synthesis_output(
                    session_id=session_id,
                    query=query,
                    response=final_response,
                    elapsed=elapsed,
                    metadata={"output_mode": output_mode, "formatted": False}
                )
                run_logger._write_to_file(f"[FORMATTER] Final synthesis saved to: {output_path}")
            except Exception as e:
                run_logger._write_to_file(f"[FORMATTER] WARNING: Failed to save synthesis: {e}")

        return Command(
            update={
                "formatted_response": None,
                "current_phase": "complete",
            },
            goto=END,
        )

    # Get the raw synthesis to format
    # NOTE: final_response was extracted above, but check again here
    if not final_response:
        run_logger._write_to_file("[FORMATTER] No final_response to format - skipping")
        # Nothing to save since there's no content
        return Command(
            update={
                "formatted_response": None,
                "current_phase": "complete",
            },
            goto=END,
        )

    run_logger._write_to_file(f"[FORMATTER] Formatting {len(final_response)} chars as '{output_mode}'")

    # Map output_mode to content_formatter format names
    # The variant config uses 'twitter' but formatter uses 'tweets'
    format_mapping = {
        "twitter": "tweets",
        "tweets": "tweets",
        "podcast": "podcast",
        "substack": "substack",
        "article": "article",
        "markdown": "markdown",
    }

    format_name = format_mapping.get(output_mode, output_mode)

    try:
        # Import the content_formatter extension
        from src.extensions.content.content_formatter.extension import deep_formatter

        # Call the formatter
        formatted_result = await deep_formatter(
            config_selection=format_name,
            content=final_response,
            session_id=session_id,
            retries=3,  # Allow retries for formatting
        )

        # Extract formatted content from Pydantic model
        if hasattr(formatted_result, 'model_dump'):
            formatted_content = formatted_result.model_dump()
        elif hasattr(formatted_result, 'dict'):
            formatted_content = formatted_result.dict()
        else:
            formatted_content = str(formatted_result)

        elapsed = time.time() - start_time
        run_logger._write_to_file(f"[FORMATTER] Formatting complete in {elapsed:.2f}s")
        run_logger._write_to_file(f"[FORMATTER] Formatted output type: {type(formatted_content).__name__}")

        # Save final synthesis (formatted content)
        if session_id:
            try:
                # Extract string content from formatted_content for saving
                if isinstance(formatted_content, dict):
                    content_to_save = formatted_content.get("content") or formatted_content.get("text") or str(formatted_content)
                else:
                    content_to_save = str(formatted_content)

                output_path = save_synthesis_output(
                    session_id=session_id,
                    query=query,
                    response=content_to_save,
                    elapsed=elapsed,
                    metadata={"output_mode": output_mode, "formatted": True}
                )
                run_logger._write_to_file(f"[FORMATTER] Final synthesis saved to: {output_path}")
            except Exception as e:
                run_logger._write_to_file(f"[FORMATTER] WARNING: Failed to save synthesis: {e}")

        return Command(
            update={
                "formatted_response": formatted_content,
                "current_phase": "complete",
            },
            goto=END,
        )

    except Exception as e:
        # GRACEFUL DEGRADATION: If formatting fails, we still have raw synthesis
        # Log the error but don't crash - the raw final_response is still available
        elapsed = time.time() - start_time
        run_logger._write_to_file(f"[FORMATTER] Formatting failed in {elapsed:.2f}s: {type(e).__name__}: {e}")
        run_logger._write_to_file("[FORMATTER] Falling back to raw synthesis (final_response)")

        # Save final synthesis (raw synthesis as fallback since formatting failed)
        if final_response and session_id:
            try:
                output_path = save_synthesis_output(
                    session_id=session_id,
                    query=query,
                    response=final_response,
                    elapsed=elapsed,
                    metadata={"output_mode": output_mode, "formatted": False, "formatting_error": str(e)}
                )
                run_logger._write_to_file(f"[FORMATTER] Final synthesis saved to: {output_path}")
            except Exception as save_e:
                run_logger._write_to_file(f"[FORMATTER] WARNING: Failed to save synthesis: {save_e}")

        # Return None for formatted_response - consumers should use final_response
        return Command(
            update={
                "formatted_response": None,
                "current_phase": "complete",
                "formatting_error": str(e),
            },
            goto=END,
        )
