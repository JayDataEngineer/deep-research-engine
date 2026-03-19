"""
Editor node for the Audio QA LangGraph workflow.

Applies fixes to detected errors using FFmpeg tools or flags for manual review.
"""

import logging
import os

from src.extensions.libraries.audio_analysis.state import AudiobookState
from src.extensions.libraries.audio_analysis.tools.ffmpeg import (
    extract_segment,
    normalize_audio,
    remove_segment,
    trim_silence,
)

logger = logging.getLogger(__name__)


def editor_node(state: AudiobookState, config: dict) -> AudiobookState:
    """
    Process the latest detected error and apply fixes if possible.

    Examines the most recent error and determines:
    - Can it be fixed automatically (trim, remove segment)?
    - Does it need manual review?
    - Does it need regeneration?

    Args:
        state: Current audiobook state
        config: Configuration dict

    Returns:
        Updated state with fixes applied or error flagged
    """
    if not state["identified_errors"]:
        logger.info("No errors to edit.")
        return state

    # Get the most recent error
    error = state["identified_errors"][-1]
    chunk_id = error["chunk_id"]

    # Check if error matches current chunk
    if chunk_id != state["current_chunk_id"]:
        logger.warning(f"Latest error chunk {chunk_id} doesn't match current {state['current_chunk_id']}")
        return state

    logger.info(f"Editing error in chunk {chunk_id}: {error['error_type']}")

    try:
        # Determine action based on error type
        action = _determine_action(error, config)

        # Apply action
        result = _apply_action(error, action, config)

        # Update state - only log if actually did something (not Skip)
        new_state = state.copy()
        if not result.startswith("Skip:"):
            new_state["messages"].append({
                "role": "editor",
                "chunk_id": chunk_id,
                "content": {
                    "error_type": error["error_type"],
                    "action": action,
                    "result": result
                }
            })

        logger.info(f"Editor action for chunk {chunk_id}: {action} -> {result}")

        return new_state

    except Exception as e:
        logger.error(f"Error in editor_node: {e}")
        state["messages"].append({
            "role": "error",
            "chunk_id": chunk_id,
            "content": f"Editor failed: {str(e)}"
        })
        return state


def _determine_action(error: dict, config: dict) -> str:
    """Determine the appropriate action for an error."""
    error_type = error["error_type"]
    proposed_fix = error.get("proposed_fix", "manual_review")

    # Respect the proposed fix from judge
    if proposed_fix != "manual_review":
        return proposed_fix

    # Editor persona from config
    persona = config.get("prompts", {}).get("editor_persona", "conservative")

    if persona == "aggressive":
        # More aggressive editing
        if error_type == "phantom_music":
            return "delete"
        elif error_type == "monotone":
            return "manual_review"  # Still can't auto-fix monotone
    else:
        # Conservative - prefer manual review
        if error_type == "phantom_music":
            # Check if at edges (trimmable)
            duration = error["timestamp_end"] - error["timestamp_start"]
            if duration < 2.0:  # Short segment at edge
                return "trim_silence"
        elif error_type == "monotone":
            return "manual_review"
        elif error_type == "voice_drift":
            return "regenerate"

    return "manual_review"


def _apply_action(error: dict, action: str, config: dict) -> str:
    """Apply the specified action to fix the error."""
    from src.constants import DATA_OUTPUTS_DIR
    chunk_id = error["chunk_id"]
    audio_path = error.get("audio_path") or _find_chunk_audio(config, chunk_id)

    if not audio_path or not os.path.exists(audio_path):
        return f"Skip: Audio file not found for chunk {chunk_id}"

    output_dir = config.get("output_dir", str(DATA_OUTPUTS_DIR / "processed"))

    if action == "trim_silence":
        output_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}_trimmed.wav")
        result_path = trim_silence(audio_path, output_path=output_path)
        return f"Trimmed silence: {result_path}"

    elif action == "delete":
        # Remove the error segment
        start = error["timestamp_start"]
        end = error["timestamp_end"]
        output_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}_cleaned.wav")
        result_path = remove_segment(audio_path, start, end, output_path)
        return f"Removed segment {start}-{end}: {result_path}"

    elif action == "apply_filter":
        # Apply normalization filter
        output_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}_filtered.wav")
        result_path = normalize_audio(audio_path, output_path=output_path)
        return f"Applied filter: {result_path}"

    elif action == "extract":
        # Extract for manual review
        output_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}_error_excerpt.wav")
        result_path = extract_segment(
            audio_path,
            error["timestamp_start"],
            error["timestamp_end"],
            output_path
        )
        return f"Extracted for review: {result_path}"

    elif action == "regenerate":
        return f"Flagged for regeneration: chunk {chunk_id}"

    else:  # manual_review
        # Create excerpt for manual review
        output_path = os.path.join(output_dir, "review", f"chunk_{chunk_id:04d}_review.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        extract_segment(
            audio_path,
            error["timestamp_start"],
            error["timestamp_end"],
            output_path
        )
        return f"Flagged for manual review: {output_path}"


def _find_chunk_audio(config: dict, chunk_id: int) -> str | None:
    """Find the audio file path for a chunk."""
    from src.constants import DATA_OUTPUTS_DIR
    chunks_dir = config.get("chunks_dir", str(DATA_OUTPUTS_DIR / "chunks"))
    # Try common patterns
    for pattern in [f"chunk_{chunk_id:04d}.wav", f"chunk_{chunk_id}.wav", f"chunk_{chunk_id:04d}.mp3"]:
        path = os.path.join(chunks_dir, pattern)
        if os.path.exists(path):
            return path
    return None


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_edl(state: AudiobookState, output_path: str) -> None:
    """
    Generate a human-readable Edit Decision List (EDL) from all identified errors.

    Args:
        state: Current audiobook state
        output_path: Path to write EDL file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("  AUDIO QA REPORT - Issues Found\n")
        f.write("=" * 70 + "\n\n")

        if not state["identified_errors"]:
            f.write("✓ No issues detected! Audio passed all quality checks.\n")
            f.write("=" * 70 + "\n")
            return

        # Summary
        f.write(f"Total Issues: {len(state['identified_errors'])}\n")
        f.write(f"Critical Issues: {state['critical_errors_found']}\n\n")

        # Error type descriptions for humans
        type_descriptions = {
            "hallucination": "🗣️ Wrong words / extra content not in script",
            "phantom_music": "🎵 Background music or noise detected",
            "monotone": "🤖 Voice sounds robotic/monotone",
            "voice_drift": "🎭 Speaker voice changed",
            "dropout": "🔇 Silence / audio drop detected",
            "volume_inconsistency": "🔊 Sudden volume change (loud/quiet)",
            "clipping": "📶 Audio distortion from clipping",
            "clicks_pops": "⚡ Clicks, pops, or crackles",
            "background_noise": "🌫️ Elevated noise floor (hiss, hum)",
            "extended_silence": "⏱️ Extended silence or dropout",
            "awkward_pause": "⏸️ Awkward timing/pause",
            "spectral_change": "🎚️ Recording environment change",
        }

        # Issues list
        f.write("-" * 70 + "\n")
        f.write("ISSUES LIST\n")
        f.write("-" * 70 + "\n\n")

        for i, error in enumerate(state["identified_errors"], 1):
            start_ts = _format_timestamp(error['timestamp_start'])
            end_ts = _format_timestamp(error['timestamp_end'])

            f.write(f"[{i}] {start_ts} → {end_ts}\n")
            f.write(f"    Type: {type_descriptions.get(error['error_type'], error['error_type'])}\n")
            f.write(f"    Severity: {'⚠️ ' if error['severity'] >= 7 else '⚡ '} {error['severity']}/10\n")
            f.write(f"    Reason: {error['description']}\n")
            f.write(f"    Suggested Fix: {error['proposed_fix'].replace('_', ' ').title()}\n")
            f.write("\n")

        # Footer
        f.write("=" * 70 + "\n")
        f.write("Generated by Audio QA Automation\n")

    logger.info(f"EDL written to {output_path}")


def generate_summary_report(state: AudiobookState, output_path: str) -> dict:
    """
    Generate a summary report of the QA analysis.

    Args:
        state: Current audiobook state
        output_path: Path to write report

    Returns:
        Summary dictionary
    """
    summary = {
        "total_chunks": len(state["manifest"]),
        "chunks_processed": state["chunks_processed_count"],
        "total_errors": len(state["identified_errors"]),
        "critical_errors": state["critical_errors_found"],
        "job_status": state["job_status"],
    }

    # Error breakdown
    error_types = {}
    for error in state["identified_errors"]:
        etype = error["error_type"]
        error_types[etype] = error_types.get(etype, 0) + 1

    summary["error_types"] = error_types

    # Write report
    import json
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary report written to {output_path}")

    return summary
