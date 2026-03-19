"""
Scheduler node for the Audio QA LangGraph workflow.

Manages the processing queue and selects chunks for analysis.
"""

import logging

from src.extensions.libraries.audio_analysis.state import AudiobookState

logger = logging.getLogger(__name__)


def scheduler_node(state: AudiobookState, config: dict = None) -> AudiobookState:
    """
    Select the next chunk from the processing queue.

    Pops the next chunk_id from the queue and loads its metadata
    into the current_chunk fields.

    Args:
        state: Current audiobook state
        config: Configuration dict containing chunk_map

    Returns:
        Updated state with current_chunk_* fields populated
    """
    # Check if there are chunks to process
    if not state["processing_queue"]:
        logger.info("Processing queue is empty.")
        state["job_status"] = "complete"
        return state

    # Get next chunk ID
    chunk_id = state["processing_queue"][0]

    # Get chunk info from chunk_map (stored in config by loader)
    chunk_map = config.get("_chunk_map", {}) if config else {}

    if chunk_id not in chunk_map:
        # Fallback to manifest lookup
        chunk = None
        for c in state["manifest"]:
            if c["chunk_id"] == chunk_id:
                chunk = c
                break

        if chunk is None:
            logger.error(f"Chunk {chunk_id} not found in manifest.")
            # Remove from queue and continue
            new_state = state.copy()
            new_state["processing_queue"] = state["processing_queue"][1:]
            return new_state

        start, end, file_path = chunk.get("start", 0), chunk.get("end", 0), chunk.get("file", "")
        reference_text = chunk.get("reference_text")
    else:
        start, end, file_path = chunk_map[chunk_id]
        reference_text = None

    # Update state with current chunk info
    new_state = state.copy()
    new_state["current_chunk_id"] = chunk_id
    new_state["current_chunk_audio"] = file_path
    new_state["current_transcript"] = reference_text

    logger.info(f"Scheduled chunk {chunk_id}: {file_path}")

    return new_state


def advance_queue(state: AudiobookState) -> AudiobookState:
    """
    Remove the current chunk from the processing queue.

    Called after a chunk has been processed.

    Args:
        state: Current audiobook state

    Returns:
        Updated state with current chunk removed from queue
    """
    if not state["processing_queue"]:
        return state

    new_state = state.copy()
    new_state["processing_queue"] = state["processing_queue"][1:]
    new_state["current_chunk_id"] = None
    new_state["current_chunk_audio"] = None
    new_state["current_transcript"] = None
    new_state["chunks_processed_count"] = state["chunks_processed_count"] + 1

    logger.info(f"Advanced queue. Remaining: {len(new_state['processing_queue'])} chunks")

    return new_state


def check_queue(state: AudiobookState) -> str:
    """
    Conditional edge function: check if processing queue is empty.

    Args:
        state: Current audiobook state

    Returns:
        "judge" if chunks remain, "__end__" if complete
    """
    if len(state["processing_queue"]) > 0:
        return "judge"
    return "__end__"


def check_for_errors(state: AudiobookState) -> str:
    """
    Conditional edge function: check if current chunk has errors.

    Routes to editor if errors were found on the current chunk,
    otherwise advance to next chunk.

    Args:
        state: Current audiobook state

    Returns:
        "editor" if errors found, "advance" otherwise
    """
    if not state["identified_errors"]:
        return "advance"

    # Check if the latest error is for the current chunk
    last_error = state["identified_errors"][-1]
    if last_error["chunk_id"] == state["current_chunk_id"]:
        return "editor"

    return "advance"
