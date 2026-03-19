"""
Judge node for the Audio QA LangGraph workflow.

Uses Qwen-Omni-3 to analyze audio quality and detect errors.
"""

import logging

from src.extensions.libraries.audio_analysis.state import AudiobookState, AudioError
from src.extensions.libraries.audio_analysis.validators import (
    JudgeOutput,
    VoiceQuality,
    parse_judge_output,
)

logger = logging.getLogger(__name__)


def judge_node(state: AudiobookState, model, config: dict) -> AudiobookState:
    """
    Analyze the current audio chunk for quality issues.

    Calls the Qwen-Omni model to analyze the audio and detect:
    - Hallucinations (transcript mismatches)
    - Phantom events (music, noise)
    - Voice quality issues (monotone, distorted)

    Args:
        state: Current audiobook state
        model: Qwen-Omni model instance (or mock)
        config: Configuration dict

    Returns:
        Updated state with errors added if any were found
    """
    if not state["current_chunk_audio"]:
        logger.warning("No current chunk audio, skipping judge.")
        return state

    chunk_id = state["current_chunk_id"]
    audio_path = state["current_chunk_audio"]
    reference_text = state.get("current_transcript") or config.get("reference_text", "")

    logger.info(f"Judging chunk {chunk_id}: {audio_path}")

    try:
        # Call model to analyze audio
        if hasattr(model, 'analyze_audio_quality'):
            raw_output = model.analyze_audio_quality(
                audio_path=audio_path,
                reference_text=reference_text,
            )
        elif hasattr(model, 'analyze_audio'):
            # Mock model interface - check if it supports chunk_id
            import inspect
            sig = inspect.signature(model.analyze_audio)
            if 'chunk_id' in sig.parameters:
                result = model.analyze_audio(
                    audio_path=audio_path,
                    reference_text=reference_text,
                    chunk_id=chunk_id
                )
            else:
                result = model.analyze_audio(
                    audio_path=audio_path,
                    reference_text=reference_text
                )
            # If mock returns JudgeOutput directly
            if isinstance(result, JudgeOutput):
                judge_result = result
            else:
                judge_result = result
        else:
            raise ValueError(f"Model {type(model)} has no analyze method")

        # Parse the output
        if isinstance(judge_result, JudgeOutput):
            result = judge_result
        else:
            result = parse_judge_output(str(raw_output), strict=False)

        # Check if errors detected
        new_errors = []

        # Check for critical issues
        if result.critical_fail:
            # Determine error type
            error_type = _determine_error_type(result)

            # Calculate severity based on accuracy score
            severity = _calculate_severity(result)

            # Determine proposed fix
            proposed_fix = _determine_fix(error_type, result, config)

            # Get timing from manifest or chunk_map
            chunk_info = _get_chunk_info(state, chunk_id, config)
            timestamp_start = chunk_info.get("start", 0.0) if chunk_info else 0.0
            timestamp_end = chunk_info.get("end", 0.0) if chunk_info else 0.0

            error = AudioError(
                chunk_id=chunk_id,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                error_type=error_type,
                description=result.reasoning,
                severity=severity,
                proposed_fix=proposed_fix
            )
            new_errors.append(error)

            logger.info(f"Error detected in chunk {chunk_id}: {error_type} (severity: {severity})")

        # Update state
        new_state = state.copy()
        new_state["identified_errors"] = state["identified_errors"] + new_errors

        # Only log messages for errors, not every pass (memory optimization)
        if new_errors:
            new_state["messages"].append({
                "role": "judge",
                "chunk_id": chunk_id,
                "content": {
                    "transcript": result.transcript,
                    "accuracy_score": result.transcript_accuracy_score,
                    "artifacts": result.detected_artifacts,
                    "voice_quality": result.voice_quality_flag.value,
                    "critical_fail": result.critical_fail,
                    "reasoning": result.reasoning
                }
            })

        # Update critical error count
        critical_count = sum(1 for e in new_errors if e["severity"] >= 7)
        new_state["critical_errors_found"] = state["critical_errors_found"] + critical_count

        return new_state

    except Exception as e:
        logger.error(f"Error in judge_node: {e}")
        state["messages"].append({
            "role": "error",
            "chunk_id": chunk_id,
            "content": f"Judge analysis failed: {str(e)}"
        })
        return state


def _determine_error_type(result: JudgeOutput) -> str:
    """Determine error type from judge output."""
    # Check for phantom artifacts first
    if result.detected_artifacts:
        artifact = result.detected_artifacts[0].lower()
        if artifact in ("music", "singing", "instrumental"):
            return "phantom_music"
        elif artifact in ("static", "noise", "hiss", "buzz", "beep"):
            return "phantom_music"  # Also treat as phantom event

    # Check voice quality
    if result.voice_quality_flag == VoiceQuality.MONOTONE:
        return "monotone"
    elif result.voice_quality_flag == VoiceQuality.DISTORTED:
        return "phantom_music"  # Could be voice drift or noise

    # Check transcript accuracy
    if result.transcript_accuracy_score < 90:
        return "hallucination"

    return "hallucination"  # Default


def _calculate_severity(result: JudgeOutput) -> int:
    """Calculate severity (1-10) from judge output."""
    # Critical issues start at severity 7
    if not result.critical_fail:
        return 1

    base = 7

    # Adjust based on accuracy score
    if result.transcript_accuracy_score < 50:
        return min(10, base + 3)
    elif result.transcript_accuracy_score < 70:
        return min(10, base + 2)
    elif result.transcript_accuracy_score < 85:
        return min(10, base + 1)

    # Voice quality issues
    if result.voice_quality_flag == VoiceQuality.DISTORTED:
        return min(10, base + 2)
    elif result.voice_quality_flag == VoiceQuality.MONOTONE:
        return base

    # Artifacts
    if "music" in result.detected_artifacts or "singing" in result.detected_artifacts:
        return min(10, base + 2)

    return base


def _determine_fix(error_type: str, result: JudgeOutput, config: dict) -> str:
    """Determine the proposed fix for an error."""
    config.get("thresholds", {})

    # Hallucinations need manual review or regeneration
    if error_type == "hallucination":
        if result.transcript_accuracy_score < 50:
            return "regenerate"
        return "manual_review"

    # Phantom music might be trimmable if at edges
    if error_type == "phantom_music":
        # Check if artifacts list suggests trimmable
        if "static" in result.detected_artifacts or "hiss" in result.detected_artifacts:
            return "apply_filter"
        return "delete"

    # Monotone voice needs regeneration
    if error_type == "monotone":
        return "regenerate"

    # Voice drift needs regeneration
    if error_type == "voice_drift":
        return "regenerate"

    return "manual_review"


def _get_chunk_info(state: AudiobookState, chunk_id: int, config: dict = None) -> dict | None:
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
