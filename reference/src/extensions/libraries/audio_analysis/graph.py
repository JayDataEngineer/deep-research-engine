"""
LangGraph workflow for Audio QA automation.

This module defines the state machine and node connections for
the audiobook quality assurance pipeline.
"""

from src.constants import DATA_OUTPUTS_DIR
from src.extensions.libraries.audio_analysis.logging_config import get_logger
from src.extensions.libraries.audio_analysis.nodes.editor import editor_node
from src.extensions.libraries.audio_analysis.nodes.extended_analyst import extended_analyst_node
from src.extensions.libraries.audio_analysis.nodes.judge import judge_node
from src.extensions.libraries.audio_analysis.nodes.loader import loader_node
from src.extensions.libraries.audio_analysis.nodes.reporter import print_summary, reporter_node
from src.extensions.libraries.audio_analysis.nodes.scheduler import (
    advance_queue,
    check_for_errors,
    check_queue,
    scheduler_node,
)
from src.extensions.libraries.audio_analysis.state import AudiobookState, create_initial_state
from langgraph.graph import END, StateGraph

logger = get_logger(__name__)


def create_workflow(model=None, config: dict | None = None):
    """
    Create the LangGraph workflow for audio QA.

    Args:
        model: Qwen-Omni model instance (or mock)
        config: Configuration dictionary

    Returns:
        Compiled StateGraph
    """
    if config is None:
        config = {}

    # Check if extended analysis is enabled
    extended_enabled = config.get("extended_analysis", {}).get("enabled", True)

    # Create the state graph
    workflow = StateGraph(AudiobookState)

    # Add nodes
    workflow.add_node("loader", lambda s: loader_node(s, config))
    workflow.add_node("scheduler", lambda s: scheduler_node(s, config))
    workflow.add_node("advance", advance_queue)  # New: advance queue after processing
    workflow.add_node("judge", lambda s: judge_node(s, model, config))

    # Only add extended analyst if enabled
    if extended_enabled:
        workflow.add_node("extended_analyst", lambda s: extended_analyst_node(s, config))

    workflow.add_node("editor", lambda s: editor_node(s, config))
    workflow.add_node("reporter", lambda s: reporter_node(s, config))

    # Set entry point
    workflow.set_entry_point("loader")

    # Add edges
    workflow.add_edge("loader", "scheduler")

    # Conditional: scheduler -> judge or END
    workflow.add_conditional_edges(
        "scheduler",
        check_queue,
        {
            "judge": "judge",
            "__end__": "reporter"
        }
    )

    if extended_enabled:
        # Judge -> extended_analyst -> check for errors -> editor/advance
        workflow.add_edge("judge", "extended_analyst")

        # Conditional: extended_analyst -> editor or advance
        workflow.add_conditional_edges(
            "extended_analyst",
            check_for_errors,
            {
                "editor": "editor",
                "advance": "advance"  # No error -> advance to next chunk
            }
        )
    else:
        # Conditional: judge -> editor or advance
        workflow.add_conditional_edges(
            "judge",
            check_for_errors,
            {
                "editor": "editor",
                "advance": "advance"  # No error -> advance to next chunk
            }
        )

    # Editor -> advance (after handling error)
    workflow.add_edge("editor", "advance")

    # Advance -> scheduler (pick next chunk)
    workflow.add_edge("advance", "scheduler")

    # Reporter -> END
    workflow.add_edge("reporter", END)

    # Compile the graph
    return workflow.compile()


def run_analysis(
    input_path: str,
    model,
    config: dict,
    reference_text: str | None = None
) -> AudiobookState:
    """
    Run the full audio QA analysis.

    Args:
        input_path: Path to input audio file
        model: Qwen-Omni model instance
        config: Configuration dictionary
        reference_text: Optional reference text for comparison

    Returns:
        Final state after analysis
    """
    # Write to debug file (only if DEBUG_AUDIO_QA is enabled)
    if os.getenv("DEBUG_AUDIO_QA") == "true":
        from src.constants import ROOT_DATA_DIR
        debug_dir = ROOT_DATA_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / "audio_qa_debug.txt"

        with open(debug_file, "a") as f:
            f.write(f"[{input_path}] run_analysis called\n")

    import sys
    if os.getenv("DEBUG_AUDIO_QA") == "true":
        print(f"[DEBUG] run_analysis called with input_path={input_path}", file=sys.stderr)

    # Update config with input path
    config = config.copy()
    config["input_path"] = input_path
    config["reference_text"] = reference_text

    # Create initial state
    initial_state = create_initial_state()
    initial_state["job_status"] = "loading"

    print("[DEBUG] Initial state created, creating workflow...", file=sys.stderr)

    # Create and run workflow
    logger.info("Starting Audio QA analysis workflow")

    workflow = create_workflow(model, config)

    try:
        # Invoke the workflow (set high recursion limit for large files)
        logger.info(f"About to call workflow.invoke, initial_state manifest={len(initial_state.get('manifest', []))}")

        for handler in logger.handlers:
            handler.flush()

        final_state = workflow.invoke(initial_state, {"recursion_limit": 10000})

        logger.info(f"workflow.invoke returned, job_status={final_state.get('job_status')}")
        logger.info(f"chunks_processed_count={final_state.get('chunks_processed_count')}")
        logger.info(f"identified_errors={len(final_state.get('identified_errors', []))}")

        for handler in logger.handlers:
            handler.flush()

        # Print summary
        print_summary(final_state)

        return final_state

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        initial_state["job_status"] = "failed"
        initial_state["messages"].append({
            "role": "error",
            "content": f"Workflow failed: {str(e)}"
        })
        return initial_state


def run_analysis_batched(
    input_path: str,
    model,
    config: dict,
    reference_text: str | None = None,
    batch_size: int = 100
) -> AudiobookState:
    """
    Run audio QA analysis in batches to handle large files.

    Processes chunks in batches to avoid memory issues with LangGraph state
    accumulation on very large files.

    Args:
        input_path: Path to input audio file
        model: Qwen-Omni model instance
        config: Configuration dictionary
        reference_text: Optional reference text for comparison
        batch_size: Number of chunks to process per batch

    Returns:
        Final state after analysis
    """
    # Update config with input path
    config = config.copy()
    config["input_path"] = input_path
    config["reference_text"] = reference_text

    # First, load and split audio to get manifest

    from src.extensions.libraries.audio_analysis.tools.audio_utils import split_audio_by_silence

    output_dir = config.get("output_dir", str(DATA_OUTPUTS_DIR / "chunks"))

    logger.info("Loading audio and creating manifest...")

    chunk_length = config.get("audio", {}).get("chunk_length_seconds", 60)
    min_chunk_duration = chunk_length / 2  # Skip chunks shorter than half the target length

    manifest = split_audio_by_silence(
        audio_path=input_path,
        output_dir=output_dir,
        chunk_duration=chunk_length,
        min_chunk_duration=min_chunk_duration,
        prefix="chunk"
    )

    logger.info(f"Created manifest with {len(manifest)} chunks")

    # Create chunk_map for config
    chunk_map = {chunk["chunk_id"]: (chunk["start"], chunk["end"], chunk["file"]) for chunk in manifest}
    config["_manifest"] = manifest
    config["_chunk_map"] = chunk_map

    # Accumulated results
    all_errors = []
    all_messages = []
    chunks_processed = 0

    # Process in batches
    total_chunks = len(manifest)
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunk_ids = [manifest[i]["chunk_id"] for i in range(batch_start, batch_end)]

        logger.info(f"Processing batch {batch_start // batch_size + 1}: chunks {batch_start+1}-{batch_end}")

        # Create initial state for this batch
        batch_state = create_initial_state()
        batch_state["manifest"] = [{"chunk_id": c} for c in batch_chunk_ids]
        batch_state["processing_queue"] = batch_chunk_ids
        batch_state["job_status"] = "scanning"
        batch_state["chunks_processed_count"] = chunks_processed
        batch_state["critical_errors_found"] = sum(1 for e in all_errors if e["severity"] >= 7)
        batch_state["identified_errors"] = list(all_errors)
        batch_state["messages"] = list(all_messages)

        # Create workflow for this batch
        workflow = create_workflow(model, config)

        try:
            batch_result = workflow.invoke(batch_state, {"recursion_limit": 10000})

            # Accumulate results
            all_errors.extend(batch_result["identified_errors"][len(all_errors):])
            all_messages.extend(batch_result["messages"][len(all_messages):])
            chunks_processed = batch_result["chunks_processed_count"]

            # Log progress
            logger.info(f"Batch complete: {chunks_processed}/{total_chunks} chunks processed, {len(all_errors)} errors found")

        except Exception as e:
            logger.error(f"Batch {batch_start // batch_size + 1} failed: {e}")
            all_messages.append({
                "role": "error",
                "content": f"Batch {batch_start // batch_size + 1} failed: {str(e)}"
            })

    # Create final state
    final_state = create_initial_state()
    final_state["manifest"] = [{"chunk_id": c["chunk_id"]} for c in manifest]
    final_state["processing_queue"] = []
    final_state["identified_errors"] = all_errors
    final_state["messages"] = all_messages
    final_state["chunks_processed_count"] = chunks_processed
    final_state["critical_errors_found"] = sum(1 for e in all_errors if e["severity"] >= 7)
    final_state["job_status"] = "complete"

    # Print summary
    print_summary(final_state)

    return final_state


def run_analysis_simple(
    input_path: str,
    model,
    config: dict,
    reference_text: str | None = None
) -> AudiobookState:
    """
    Run audio QA analysis using simple iteration (bypasses LangGraph state overhead).

    This is optimized for large files with mock models where LangGraph's state
    management becomes a bottleneck. Direct iteration avoids state copying.

    Args:
        input_path: Path to input audio file
        model: Qwen-Omni model instance
        config: Configuration dictionary
        reference_text: Optional reference text for comparison

    Returns:
        Final state after analysis
    """
    import os

    from src.extensions.libraries.audio_analysis.nodes.editor import generate_edl
    from src.extensions.libraries.audio_analysis.nodes.judge import (
        _calculate_severity,
        _determine_error_type,
        _determine_fix,
    )
    from src.extensions.libraries.audio_analysis.state import AudioError
    from src.extensions.libraries.audio_analysis.tools.audio_utils import split_audio_by_silence

    output_dir = config.get("output_dir", str(DATA_OUTPUTS_DIR / "chunks"))
    logger.info("Loading audio and creating manifest (simple mode)...")

    chunk_length = config.get("audio", {}).get("chunk_length_seconds", 60)
    min_chunk_duration = chunk_length / 2  # Skip chunks shorter than half the target length

    manifest = split_audio_by_silence(
        audio_path=input_path,
        output_dir=output_dir,
        chunk_duration=chunk_length,
        min_chunk_duration=min_chunk_duration,
        prefix="chunk"
    )

    logger.info(f"Created manifest with {len(manifest)} chunks")

    # Check if extended analysis is enabled
    extended_enabled = config.get("extended_analysis", {}).get("enabled", True)
    if extended_enabled:
        from src.extensions.libraries.audio_analysis.tools.extended_analysis import run_extended_analysis

    # Process chunks directly
    all_errors = []
    all_messages = []

    for i, chunk in enumerate(manifest):
        chunk_id = chunk["chunk_id"]
        audio_path = chunk["file"]

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{len(manifest)} chunks processed, {len(all_errors)} errors found")

        # Call model
        try:
            if hasattr(model, 'analyze_audio'):
                import inspect
                sig = inspect.signature(model.analyze_audio)
                if 'chunk_id' in sig.parameters:
                    result = model.analyze_audio(audio_path=audio_path, reference_text=reference_text, chunk_id=chunk_id)
                else:
                    result = model.analyze_audio(audio_path=audio_path, reference_text=reference_text)
            else:
                logger.warning(f"Chunk {chunk_id}: Model has no analyze_audio method")
                continue

            # Check for errors from model
            if result.critical_fail:
                error_type = _determine_error_type(result)
                severity = _calculate_severity(result)
                proposed_fix = _determine_fix(error_type, result, config)

                error = AudioError(
                    chunk_id=chunk_id,
                    timestamp_start=chunk["start"],
                    timestamp_end=chunk["end"],
                    error_type=error_type,
                    description=result.reasoning,
                    severity=severity,
                    proposed_fix=proposed_fix
                )
                all_errors.append(error)

                # Only log message for error (memory optimization)
                all_messages.append({
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

            # Run extended analysis if enabled
            if extended_enabled:
                try:
                    extended_issues = run_extended_analysis(audio_path)
                    for issue in extended_issues:
                        # Adjust timestamps to be relative to full audio
                        timestamp_start = chunk["start"] + issue["timestamp_start"]
                        timestamp_end = chunk["start"] + issue["timestamp_end"]

                        # Determine proposed fix
                        error_type = issue["type"]
                        fix_map = {
                            "volume_inconsistency": "normalize",
                            "clipping": "regenerate",
                            "clicks_pops": "apply_filter",
                            "background_noise": "apply_filter",
                            "extended_silence": "trim_silence",
                            "awkward_pause": "trim_silence",
                            "spectral_change": "manual_review",
                        }
                        proposed_fix = fix_map.get(error_type, "manual_review")

                        error = AudioError(
                            chunk_id=chunk_id,
                            timestamp_start=timestamp_start,
                            timestamp_end=timestamp_end,
                            error_type=error_type,
                            description=issue["description"],
                            severity=issue["severity"],
                            proposed_fix=proposed_fix
                        )
                        all_errors.append(error)

                        # Log message for extended analysis error
                        all_messages.append({
                            "role": "extended_analyst",
                            "chunk_id": chunk_id,
                            "content": {
                                "error_type": error_type,
                                "description": issue["description"],
                                "severity": issue["severity"]
                            }
                        })

                except Exception as e:
                    logger.warning(f"Extended analysis failed for chunk {chunk_id}: {e}")

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")

    # Create final state
    final_state = create_initial_state()
    final_state["manifest"] = [{"chunk_id": c["chunk_id"]} for c in manifest]
    final_state["processing_queue"] = []
    final_state["identified_errors"] = all_errors
    final_state["messages"] = all_messages
    final_state["chunks_processed_count"] = len(manifest)
    final_state["critical_errors_found"] = sum(1 for e in all_errors if e["severity"] >= 7)
    final_state["job_status"] = "complete"

    # Generate human-readable EDL
    edl_path = os.path.join(output_dir, "qa_report.txt")
    generate_edl(final_state, edl_path)
    logger.info(f"Human-readable report written to {edl_path}")

    # Print summary
    print_summary(final_state)

    return final_state


def run_analysis_extended_only(
    input_path: str,
    config: dict,
    reference_text: str | None = None
) -> AudiobookState:
    """
    Run audio QA analysis using ONLY extended DSP analysis (fast, no model).

    This bypasses the slow Qwen-Omni model and uses only DSP-based analysis:
    - Volume inconsistency detection
    - Clipping detection
    - Click/pop detection
    - Background noise floor detection
    - Extended silence detection
    - Spectral change detection

    This is much faster (~1 second per chunk vs ~80 seconds with model).

    Args:
        input_path: Path to input audio file
        config: Configuration dictionary
        reference_text: Optional reference text (not used in extended-only mode)

    Returns:
        Final state after analysis
    """
    import os

    from src.extensions.libraries.audio_analysis.nodes.editor import generate_edl
    from src.extensions.libraries.audio_analysis.state import AudioError
    from src.extensions.libraries.audio_analysis.tools.audio_utils import split_audio_by_silence
    from src.extensions.libraries.audio_analysis.tools.extended_analysis import run_extended_analysis

    output_dir = config.get("output_dir", str(DATA_OUTPUTS_DIR / "chunks"))
    logger.info("Loading audio and creating manifest (extended-only mode - FAST!)...")

    chunk_length = config.get("audio", {}).get("chunk_length_seconds", 60)
    min_chunk_duration = chunk_length / 2

    manifest = split_audio_by_silence(
        audio_path=input_path,
        output_dir=output_dir,
        chunk_duration=chunk_length,
        min_chunk_duration=min_chunk_duration,
        prefix="chunk"
    )

    logger.info(f"Created manifest with {len(manifest)} chunks")
    logger.info(f"Running extended-only analysis (no model - ~{len(manifest)} seconds expected)...")

    # Process chunks directly with extended analysis only
    all_errors = []
    all_messages = []

    for i, chunk in enumerate(manifest):
        chunk_id = chunk["chunk_id"]
        audio_path = chunk["file"]

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{len(manifest)} chunks processed, {len(all_errors)} errors found")

        # Run extended analysis ONLY (skip the slow model)
        try:
            extended_issues = run_extended_analysis(audio_path)
            for issue in extended_issues:
                # Adjust timestamps to be relative to full audio
                timestamp_start = chunk["start"] + issue["timestamp_start"]
                timestamp_end = chunk["start"] + issue["timestamp_end"]

                # Determine proposed fix
                error_type = issue["type"]
                fix_map = {
                    "volume_inconsistency": "normalize",
                    "clipping": "regenerate",
                    "clicks_pops": "apply_filter",
                    "background_noise": "apply_filter",
                    "extended_silence": "trim_silence",
                    "awkward_pause": "trim_silence",
                    "spectral_change": "manual_review",
                }
                proposed_fix = fix_map.get(error_type, "manual_review")

                error = AudioError(
                    chunk_id=chunk_id,
                    timestamp_start=timestamp_start,
                    timestamp_end=timestamp_end,
                    error_type=error_type,
                    description=issue["description"],
                    severity=issue["severity"],
                    proposed_fix=proposed_fix
                )
                all_errors.append(error)

                # Log message for extended analysis error
                all_messages.append({
                    "role": "extended_analyst",
                    "chunk_id": chunk_id,
                    "content": {
                        "error_type": error_type,
                        "description": issue["description"],
                        "severity": issue["severity"]
                    }
                })

        except Exception as e:
            logger.warning(f"Extended analysis failed for chunk {chunk_id}: {e}")

    # Create final state
    final_state = create_initial_state()
    final_state["manifest"] = [{"chunk_id": c["chunk_id"]} for c in manifest]
    final_state["processing_queue"] = []
    final_state["identified_errors"] = all_errors
    final_state["messages"] = all_messages
    final_state["chunks_processed_count"] = len(manifest)
    final_state["critical_errors_found"] = sum(1 for e in all_errors if e["severity"] >= 7)
    final_state["job_status"] = "complete"

    # Generate human-readable EDL
    edl_path = os.path.join(output_dir, "qa_report.txt")
    generate_edl(final_state, edl_path)
    logger.info(f"Human-readable report written to {edl_path}")

    # Print summary
    print_summary(final_state)

    return final_state


def run_analysis_with_mock(
    input_path: str,
    config: dict,
    scenario: str = "mixed",
    reference_text: str | None = None,
    batch_size: int = 25
) -> AudiobookState:
    """
    Run analysis with mock model for testing.

    Args:
        input_path: Path to input audio file
        config: Configuration dictionary
        scenario: Test scenario ("all_pass", "mixed", "all_errors")
        reference_text: Optional reference text
        batch_size: Number of chunks to process per batch (for large files)

    Returns:
        Final state after analysis
    """
    from src.extensions.libraries.audio_analysis.models.mock_judge import (
        TEST_SCENARIOS,
        DeterministicMockQwenOmni,
    )

    # Create mock model with scenario
    scenario_map = TEST_SCENARIOS.get(scenario, TEST_SCENARIOS["mixed"])
    model = DeterministicMockQwenOmni(scenario_map)

    # Use simple iteration for mock model (bypasses LangGraph state overhead)
    return run_analysis_simple(input_path, model, config, reference_text)


def run_analysis_modular(
    input_path: str,
    config: dict,
    reference_text: str | None = None
) -> AudiobookState:
    """
    Run audio QA using the modular "surgical strike" pipeline.

    This uses specialized tools for fast feature extraction:
    - CLAP: Music/hallucination detection
    - Whisper: Transcription and WER calculation
    - Librosa: DSP features (pitch, volume, spectral)

    Architecture: Audio → [Specialized Extractors] → Feature Dict → [LLM Judge] → Pass/Fail

    This is the "middle ground" between extended-only (no model) and full model analysis:
    - Extended-only: ~2 minutes, DSP only, no transcription
    - Modular: ~5-10 minutes, DSP + transcription + music detection
    - Full model: ~12.6 hours, slow AI analysis

    Args:
        input_path: Path to input audio file
        config: Configuration dictionary
        reference_text: Optional reference text for WER calculation

    Returns:
        Final state after analysis
    """
    import os

    from src.extensions.libraries.audio_analysis.nodes.editor import generate_edl
    from src.extensions.libraries.audio_analysis.state import AudioError
    from src.extensions.libraries.audio_analysis.tools import HAS_MODULAR_EXTRACTOR
    from src.extensions.libraries.audio_analysis.tools.audio_utils import split_audio_by_silence

    if not HAS_MODULAR_EXTRACTOR:
        logger.warning("Modular extractor not available - falling back to extended-only mode")
        logger.info("Install with: uv sync --extra audio-qa")
        return run_analysis_extended_only(input_path, config, reference_text)

    from src.extensions.libraries.audio_analysis.tools.modular_extractor import ModularFeatureExtractor

    output_dir = config.get("output_dir", str(DATA_OUTPUTS_DIR / "chunks"))
    logger.info("Loading audio and creating manifest (modular pipeline - CLAP + Whisper + Librosa)...")

    chunk_length = config.get("audio", {}).get("chunk_length_seconds", 60)
    min_chunk_duration = chunk_length / 2

    manifest = split_audio_by_silence(
        audio_path=input_path,
        output_dir=output_dir,
        chunk_duration=chunk_length,
        min_chunk_duration=min_chunk_duration,
        prefix="chunk"
    )

    logger.info(f"Created manifest with {len(manifest)} chunks")
    logger.info("Running modular analysis (fast feature extraction + LLM judgment)...")

    # Initialize extractor (models will be lazy-loaded)
    extractor = ModularFeatureExtractor(
        clap_model_path=config.get("modular", {}).get("clap_model", "laion/larger_clap_general"),
        whisper_model_path=config.get("modular", {}).get("whisper_model", "Systran/faster-whisper-large-v3"),
        device="cuda" if config.get("use_cuda", True) else "cpu",
        reference_text=reference_text,
    )

    # Process chunks
    all_errors = []
    all_messages = []
    all_features = []

    for i, chunk in enumerate(manifest):
        chunk_id = chunk["chunk_id"]
        audio_path = chunk["file"]

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(manifest)} chunks processed, {len(all_errors)} errors found")

        try:
            # Extract all features for this chunk
            features = extractor.extract_all_features(
                audio_path=audio_path,
                chunk_id=chunk_id,
                start_time=chunk["start"],
                end_time=chunk["end"],
                reference_text=reference_text,
            )
            all_features.append(features)

            # Check for issues based on extracted features
            for issue in features.get("issues_detected", []):
                # Map issue names to error types
                issue_type_map = {
                    "volume_inconsistent": "volume_inconsistency",
                    "too_quiet": "volume_inconsistency",
                    "monotone_robotic": "voice_quality",
                    "music_detected": "hallucination",
                    "hallucination": "hallucination",
                    "high_wer": "transcription_error",
                }

                error_type = issue_type_map.get(issue, "quality_issue")

                # Determine severity based on overall quality score
                quality_score = features.get("overall_quality_score", 1.0)
                if quality_score < 0.3:
                    severity = 9
                elif quality_score < 0.5:
                    severity = 7
                elif quality_score < 0.7:
                    severity = 5
                else:
                    severity = 3

                # Create description from features
                description_parts = []
                if features.get("music_probability", 0) > 0.5:
                    description_parts.append(f"Music detected (prob: {features['music_probability']:.2f})")
                if features.get("wer_score") and features["wer_score"] > 0.2:
                    description_parts.append(f"High WER: {features['wer_score']:.2f}")
                if issue == "monotone_robotic":
                    description_parts.append(f"Low pitch variance: {features.get('pitch_variance', 0):.1f}")

                description = f"{issue}: " + (", ".join(description_parts) if description_parts else "Quality issue detected")

                # Determine proposed fix
                fix_map = {
                    "volume_inconsistency": "normalize",
                    "voice_quality": "regenerate",
                    "hallucination": "regenerate",
                    "transcription_error": "manual_review",
                    "quality_issue": "manual_review",
                }
                proposed_fix = fix_map.get(error_type, "manual_review")

                error = AudioError(
                    chunk_id=chunk_id,
                    timestamp_start=chunk["start"],
                    timestamp_end=chunk["end"],
                    error_type=error_type,
                    description=description,
                    severity=severity,
                    proposed_fix=proposed_fix
                )
                all_errors.append(error)

                # Log message
                all_messages.append({
                    "role": "modular_analyst",
                    "chunk_id": chunk_id,
                    "content": {
                        "error_type": error_type,
                        "description": description,
                        "severity": severity,
                        "features": {
                            "quality_score": quality_score,
                            "music_probability": features.get("music_probability"),
                            "wer_score": features.get("wer_score"),
                            "pitch_variance": features.get("pitch_variance"),
                            "volume_variance": features.get("volume_variance"),
                        }
                    }
                })

        except Exception as e:
            logger.error(f"Modular extraction failed for chunk {chunk_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Create final state
    final_state = create_initial_state()
    final_state["manifest"] = [{"chunk_id": c["chunk_id"]} for c in manifest]
    final_state["processing_queue"] = []
    final_state["identified_errors"] = all_errors
    final_state["messages"] = all_messages
    final_state["chunks_processed_count"] = len(manifest)
    final_state["critical_errors_found"] = sum(1 for e in all_errors if e["severity"] >= 7)
    final_state["job_status"] = "complete"

    # Store extracted features for analysis
    final_state["_extracted_features"] = all_features

    # Generate human-readable EDL
    edl_path = os.path.join(output_dir, "qa_report.txt")
    generate_edl(final_state, edl_path)
    logger.info(f"Human-readable report written to {edl_path}")

    # Print summary
    print_summary(final_state)

    return final_state


# Visual representation for documentation
WORKFLOW_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO QA WORKFLOW                             │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
         ▼
    ┌──────────┐     ┌─────────────┐
    │  Loader  │────►│   Manifest  │
    │  Split   │     │   Created   │
    └────┬─────┘     └─────────────┘
         │
         ▼
    ┌──────────┐     ┌─────────────┐
    │Scheduler │────►│   Queue     │
    │  Next    │     │   Empty?    │
    └────┬─────┘     └──────┬──────┘
         │                   │
         │         ┌─────────┴─────────┐
         │         │                   │
         │    Yes ▼                   │ No
         │  ┌─────────┐               │
         └──│  END    │               │
            └─────────┘               │
                                      ▼
                              ┌──────────────┐
                              │    Judge     │
                              │ (Qwen-Omni)  │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │   Extended   │
                              │   Analyst    │
                              │ (Signal)     │
                              └──────┬───────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                      Error?                Pass
                          │                     │
                       Yes ▼                   │
                    ┌─────────┐                │
                    │ Editor  │                │
                    │ (FFmpeg)│                │
                    └────┬────┘                │
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                            ┌──────────────┐
                            │  Scheduler   │◄────┐
                            │  (Advance)   │     │
                            └──────────────┘     │
                                    │             │
                                    └─────────────┘
"""
