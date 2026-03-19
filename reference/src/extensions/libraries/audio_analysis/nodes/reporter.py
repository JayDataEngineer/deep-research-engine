"""
Reporter node for the Audio QA LangGraph workflow.

Generates final reports and summaries.
"""

import json
import logging
import os
from datetime import datetime

from src.extensions.libraries.audio_analysis.nodes.editor import generate_edl, generate_summary_report
from src.extensions.libraries.audio_analysis.state import AudiobookState

logger = logging.getLogger(__name__)


def reporter_node(state: AudiobookState, config: dict) -> AudiobookState:
    """
    Generate final reports after all chunks are processed.

    Creates:
    - Summary JSON with statistics
    - Edit Decision List (EDL) with all errors
    - Detailed error log

    Args:
        state: Current audiobook state
        config: Configuration dict

    Returns:
        Updated state with report paths recorded
    """
    from src.constants import DATA_OUTPUTS_DIR
    output_dir = config.get("output_dir", str(DATA_OUTPUTS_DIR))
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate summary report
    summary_path = os.path.join(reports_dir, f"summary_{timestamp}.json")
    summary = generate_summary_report(state, summary_path)

    # Generate EDL if errors found
    edl_path = None
    if state["identified_errors"]:
        edl_path = os.path.join(reports_dir, f"edl_{timestamp}.txt")
        generate_edl(state, edl_path)

    # Generate detailed JSON report
    detail_path = os.path.join(reports_dir, f"detail_{timestamp}.json")
    _generate_detailed_report(state, detail_path)

    # Update state
    new_state = state.copy()
    new_state["messages"].append({
        "role": "reporter",
        "content": {
            "summary_path": summary_path,
            "edl_path": edl_path,
            "detail_path": detail_path,
            "summary": summary
        }
    })

    logger.info(f"Reports generated in {reports_dir}")

    return new_state


def _generate_detailed_report(state: AudiobookState, output_path: str) -> None:
    """Generate detailed JSON report with all errors and metadata."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "job_status": state["job_status"],
        "statistics": {
            "total_chunks": len(state["manifest"]),
            "chunks_processed": state["chunks_processed_count"],
            "total_errors": len(state["identified_errors"]),
            "critical_errors": state["critical_errors_found"],
        },
        "manifest": state["manifest"],
        "errors": state["identified_errors"],
        "message_log": state["messages"][-50:],  # Last 50 messages
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Detailed report written to {output_path}")


def print_summary(state: AudiobookState) -> None:
    """Print a summary to console."""
    print("\n" + "=" * 60)
    print("AUDIO QA ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total Chunks:        {len(state['manifest'])}")
    print(f"Chunks Processed:    {state['chunks_processed_count']}")
    print(f"Total Errors:        {len(state['identified_errors'])}")
    print(f"Critical Errors:     {state['critical_errors_found']}")
    print(f"Status:              {state['job_status']}")

    if state["identified_errors"]:
        print("\nError Breakdown:")
        error_counts = {}
        for error in state["identified_errors"]:
            etype = error["error_type"]
            error_counts[etype] = error_counts.get(etype, 0) + 1

        for etype, count in sorted(error_counts.items()):
            print(f"  - {etype}: {count}")

    print("=" * 60 + "\n")
