"""
Extended Analyst node for the Audio QA LangGraph workflow.

Runs signal-based audio analysis to detect technical issues like
volume inconsistencies, clipping, background noise, timing issues.
"""

import logging

from src.extensions.libraries.audio_analysis.state import AudiobookState, AudioError

logger = logging.getLogger(__name__)


def extended_analyst_node(state: AudiobookState, config: dict) -> AudiobookState:
    """
    Run extended signal-based analysis on the current audio chunk.

    Detects technical audio issues that complement the semantic analysis
    from Qwen-Omni:
    - Volume inconsistencies (sudden loud/quiet)
    - Audio artifacts (clipping, clicks/pops)
    - Background noise (hiss, hum)
    - Timing issues (awkward pauses, extended silence)
    - Spectral changes (recording environment changes)

    Args:
        state: Current audiobook state
        config: Configuration dict

    Returns:
        Updated state with additional errors if any were found
    """
    if not state["current_chunk_audio"]:
        logger.warning("No current chunk audio, skipping extended analysis.")
        return state

    chunk_id = state["current_chunk_id"]
    audio_path = state["current_chunk_audio"]

    # Check if extended analysis is enabled
    if not config.get("extended_analysis", {}).get("enabled", True):
        return state

    logger.info(f"Running extended analysis on chunk {chunk_id}: {audio_path}")

    try:
        from src.extensions.libraries.audio_analysis.tools.extended_analysis import (
            detect_audio_artifacts,
            detect_background_noise,
            detect_spectral_inconsistencies,
            detect_timing_issues,
            detect_volume_inconsistencies,
        )

        # Get analysis thresholds from config
        ext_config = config.get("extended_analysis", {})
        volume_threshold = ext_config.get("volume_threshold_db", 6.0)
        clip_threshold = ext_config.get("clip_threshold", 0.95)
        noise_floor = ext_config.get("noise_floor_db", -50)
        min_pause = ext_config.get("min_silence_duration", 2.0)
        max_silence = ext_config.get("max_silence_duration", 10.0)

        # Run all detection methods
        import librosa
        audio, sr = librosa.load(audio_path, sr=None)

        new_errors = []

        # Volume inconsistencies
        volume_issues = detect_volume_inconsistencies(
            audio, sr,
            threshold_db=volume_threshold
        )
        for issue in volume_issues:
            new_errors.append(_issue_to_error(
                issue, chunk_id, audio_path, state, config
            ))

        # Audio artifacts
        artifact_issues = detect_audio_artifacts(
            audio, sr,
            clip_threshold=clip_threshold
        )
        for issue in artifact_issues:
            new_errors.append(_issue_to_error(
                issue, chunk_id, audio_path, state, config
            ))

        # Background noise
        noise_issues = detect_background_noise(
            audio, sr,
            noise_floor_db=noise_floor
        )
        for issue in noise_issues:
            new_errors.append(_issue_to_error(
                issue, chunk_id, audio_path, state, config
            ))

        # Timing issues
        timing_issues = detect_timing_issues(
            audio, sr,
            min_silence_duration=min_pause,
            max_silence_duration=max_silence
        )
        for issue in timing_issues:
            new_errors.append(_issue_to_error(
                issue, chunk_id, audio_path, state, config
            ))

        # Spectral inconsistencies
        spectral_issues = detect_spectral_inconsistencies(audio, sr)
        for issue in spectral_issues:
            new_errors.append(_issue_to_error(
                issue, chunk_id, audio_path, state, config
            ))

        # Update state
        new_state = state.copy()
        if new_errors:
            new_state["identified_errors"] = state["identified_errors"] + new_errors

            # Log messages for extended analysis errors
            for err in new_errors:
                new_state["messages"].append({
                    "role": "extended_analyst",
                    "chunk_id": chunk_id,
                    "content": {
                        "error_type": err["error_type"],
                        "description": err["description"],
                        "severity": err["severity"]
                    }
                })

            logger.info(f"Extended analysis found {len(new_errors)} issues in chunk {chunk_id}")

        # Update critical error count
        critical_count = sum(1 for e in new_errors if e["severity"] >= 7)
        new_state["critical_errors_found"] = state["critical_errors_found"] + critical_count

        return new_state

    except Exception as e:
        logger.error(f"Error in extended_analyst_node: {e}")
        state["messages"].append({
            "role": "error",
            "chunk_id": chunk_id,
            "content": f"Extended analysis failed: {str(e)}"
        })
        return state


def _issue_to_error(
    issue: dict,
    chunk_id: int,
    audio_path: str,
    state: AudiobookState,
    config: dict
) -> AudioError:
    """
    Convert an extended analysis issue to an AudioError.

    Adjusts timestamps to be relative to the full audio file,
    determines the proposed fix based on error type.

    Args:
        issue: Issue dict from extended analysis
        chunk_id: Current chunk ID
        audio_path: Path to chunk audio file
        state: Current audiobook state
        config: Configuration dict

    Returns:
        AudioError object
    """
    # Get chunk timing to offset timestamps
    chunk_info = _get_chunk_info(state, chunk_id, config)
    chunk_start = chunk_info.get("start", 0.0) if chunk_info else 0.0
    chunk_info.get("end", 0.0) if chunk_info else 0.0

    # Adjust timestamps to be relative to full audio
    timestamp_start = chunk_start + issue["timestamp_start"]
    timestamp_end = chunk_start + issue["timestamp_end"]

    # Determine proposed fix based on error type
    error_type = issue["type"]
    proposed_fix = _determine_extended_fix(error_type, issue, config)

    return AudioError(
        chunk_id=chunk_id,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        error_type=error_type,
        description=issue["description"],
        severity=issue["severity"],
        proposed_fix=proposed_fix
    )


def _determine_extended_fix(error_type: str, issue: dict, config: dict) -> str:
    """
    Determine the proposed fix for an extended analysis error.

    Args:
        error_type: Type of error
        issue: Issue dict
        config: Configuration dict

    Returns:
        Proposed fix action
    """
    fix_map = {
        "volume_inconsistency": "normalize",
        "clipping": "regenerate",  # Clipping means re-record
        "clicks_pops": "apply_filter",  # Can filter out clicks
        "background_noise": "apply_filter",  # Noise reduction
        "extended_silence": "trim_silence",
        "awkward_pause": "trim_silence",
        "spectral_change": "manual_review",  # Environment change - need human check
    }

    return fix_map.get(error_type, "manual_review")


def _get_chunk_info(state: AudiobookState, chunk_id: int, config: dict = None) -> dict:
    """Get chunk info from manifest or chunk_map."""
    # Try chunk_map first (more efficient)
    if config and "_chunk_map" in config:
        chunk_map = config["_chunk_map"]
        if chunk_id in chunk_map:
            start, end, file_path = chunk_map[chunk_id]
            return {"start": start, "end": end, "file": file_path, "chunk_id": chunk_id}

    # Fallback to manifest lookup
    for chunk in state["manifest"]:
        if chunk["chunk_id"] == chunk_id:
            return chunk
    return None
