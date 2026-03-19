"""
Loader node for the Audio QA LangGraph workflow.

Handles audio file loading and splitting into chunks.
"""

import logging
import os

from src.extensions.libraries.audio_analysis.state import AudiobookState
from src.extensions.libraries.audio_analysis.tools.audio_utils import split_audio_by_silence

logger = logging.getLogger(__name__)


def loader_node(state: AudiobookState, config: dict) -> AudiobookState:
    """
    Load audio file and split into chunks.

    If manifest is empty, split the input audio into chunks based on silence.

    Args:
        state: Current audiobook state
        config: Configuration dict with input_path and output_dir

    Returns:
        Updated state with manifest and processing queue populated
    """
    from src.constants import DATA_OUTPUTS_DIR
    logger.info(f"[LOADER] loader_node called, manifest={len(state.get('manifest', []))}")
    for handler in logger.handlers:
        handler.flush()

    # If we already have a manifest, skip
    if state["manifest"]:
        logger.info("Manifest already exists, skipping loader.")
        return state

    input_path = config.get("input_path")
    output_dir = config.get("output_dir", str(DATA_OUTPUTS_DIR / "chunks"))

    if not input_path:
        logger.error("No input_path provided in config.")
        state["job_status"] = "failed"
        return state

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        state["job_status"] = "failed"
        return state

    logger.info(f"Loading audio from: {input_path}")
    state["job_status"] = "loading"

    try:
        # Split audio into chunks using ffmpeg streaming
        chunk_length = config.get("audio", {}).get("chunk_length_seconds", 60)
        min_chunk_duration = chunk_length / 2  # Skip chunks shorter than half the target length

        manifest = split_audio_by_silence(
            audio_path=input_path,
            output_dir=output_dir,
            chunk_duration=chunk_length,
            min_chunk_duration=min_chunk_duration,
            prefix="chunk"
        )

        # Create processing queue (all chunk IDs)
        processing_queue = [chunk["chunk_id"] for chunk in manifest]

        # Create lightweight chunk_map: chunk_id -> (start, end, file_path)
        # This is much more memory-efficient than full manifest
        chunk_map = {chunk["chunk_id"]: (chunk["start"], chunk["end"], chunk["file"]) for chunk in manifest}

        # Store manifest in config instead of state (memory optimization)
        config["_manifest"] = manifest
        config["_chunk_map"] = chunk_map

        # Update state - only store lightweight manifest in state
        new_state = state.copy()
        new_state["manifest"] = [{"chunk_id": c["chunk_id"]} for c in manifest]  # Just IDs
        new_state["processing_queue"] = processing_queue
        new_state["job_status"] = "scanning"
        new_state["messages"].append({
            "role": "system",
            "content": f"Loaded {len(manifest)} chunks from {input_path}"
        })

        logger.info(f"Created manifest with {len(manifest)} chunks")
        return new_state

    except Exception as e:
        logger.error(f"Error in loader_node: {e}")
        state["job_status"] = "failed"
        state["messages"].append({
            "role": "error",
            "content": f"Loader failed: {str(e)}"
        })
        return state


def load_reference_text(reference_path: str) -> str | None:
    """
    Load reference text from file.

    Args:
        reference_path: Path to text file

    Returns:
        Text content or None if error
    """
    try:
        with open(reference_path, encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading reference text: {e}")
        return None


def create_manifest_from_reference(
    reference_text: str,
    words_per_chunk: int = 50
) -> list:
    """
    Create a manifest by splitting reference text into chunks.

    Useful when audio is already chunked and we need to align text.

    Args:
        reference_text: Full reference text
        words_per_chunk: Approximate words per chunk

    Returns:
        List of text chunks
    """
    words = reference_text.split()
    chunks = []

    for i in range(0, len(words), words_per_chunk):
        chunk_text = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk_text)

    return chunks
