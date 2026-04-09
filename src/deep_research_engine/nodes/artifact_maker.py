from typing import Any

from ..config import get_phase_config
from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..workers import WorkerConfig, WorkerRegistry

logger = get_loguru_logger("nodes.artifact_maker")


@node_context("artifact_maker")
async def artifact_maker_node(state: EngineState) -> dict[str, Any]:
    """Artifact Maker node - Formats the final research output into an artifact."""
    logger.info("Starting artifact maker node")

    # Get config and check if enabled
    config_data = state.get("context", {}).get("config", {})
    # Fallback to 'synthesizer' for backwards compatibility
    phase_config = get_phase_config(config_data, "artifact_maker")
    if not phase_config:
        phase_config = get_phase_config(config_data, "synthesizer")

    artifact_maker_enabled = phase_config.get("enabled", True)

    if not artifact_maker_enabled:
        logger.info("Artifact maker node disabled. Skipping.")
        return {"__goto__": "complete"}

    # 1. Try to use ArtifactMakerWorker if available
    worker_class = WorkerRegistry.get("artifact_maker")
    # Fallback for backwards compatibility
    if not worker_class:
        worker_class = WorkerRegistry.get("synthesis")

    if worker_class:
        try:
            worker_config = WorkerConfig(
                worker_id="artifact_maker_worker",
                worker_type="artifact_maker",
                task="Produce research artifact",
                custom=phase_config
            )
            worker = worker_class(worker_config)
            worker_result = await worker.execute(state, worker_config)

            if worker_result.get("success"):
                logger.info("Artifact maker worker completed successfully.")

                # Check for either key to be safe
                output = state.get("artifact_output") or state.get("synthesis_output")
                metadata = state.get("artifact_metadata") or state.get("synthesis_metadata", {})

                return {
                    "artifact_output": output,
                    "artifact_metadata": {
                        **metadata,
                        "worker_count": len(state.get("worker_results", []))
                    },
                    "__goto__": "complete"
                }
            else:
                logger.warning(f"Artifact maker worker failed: {worker_result.get('error', 'unknown')}. Falling back to basic aggregation.")
        except Exception as e:
            logger.error(f"Error executing artifact maker worker: {e}. Falling back.")

    # 2. Detective Report (ingestion workflows with structured data)
    if state.get("context", {}).get("ingestion_structured"):
        try:
            from pathlib import Path

            from ..artifacts.detective_report import build_detective_report
            from ..artifacts.file_organizer import organize_files

            structured = state["context"]["ingestion_structured"]
            media_items = structured.get("media_items", [])

            # 2a. Organize media files into browsable directories
            big_dump = state.get("context", {}).get("big_text_dump")
            output_dir = Path(big_dump).parent if big_dump else Path("output/artifacts")
            input_dir = Path(state.get("input_data", {}).get("path", "."))

            org_stats = organize_files(output_dir, media_items, input_dir)
            logger.info(f"File organization: {org_stats}")

            # 2b. Build detective report
            report = await build_detective_report(state)
            _write_artifact_to_disk(state, report)
            _write_report_sections(state, report)

            logger.info(f"Detective report generated ({len(report)} chars).")
            return {
                "artifact_output": report,
                "artifact_metadata": {
                    "format": "detective_report",
                    "organized_files": org_stats,
                    "worker_count": len(state.get("worker_results", [])),
                },
                "__goto__": "complete",
            }
        except Exception as e:
            logger.error(f"Detective report failed: {e}", exc_info=True)
            # Fall through to basic aggregation

    # 3. Basic fallback aggregation
    worker_results = state.get("worker_results", [])
    successful_results = [r for r in worker_results if r.get("success")]

    if not successful_results:
        logger.warning("No successful worker results to turn into an artifact.")
        return {
            "artifact_output": "No research results were generated.",
            "__goto__": "complete"
        }

    # Aggregate results
    content_parts = []
    for res in successful_results:
        output = res.get("output")
        if isinstance(output, str):
            content_parts.append(output)
        else:
            content_parts.append(str(output))

    final_output = "# Final Research Artifact\n\n" + "\n\n".join(content_parts)

    # Write final artifact to disk
    _write_artifact_to_disk(state, final_output)

    logger.info(f"Basic aggregation complete. Output length: {len(final_output)} chars.")
    return {
        "artifact_output": final_output,
        "artifact_metadata": {
            "format": "basic_aggregation",
            "worker_count": len(successful_results)
        },
        "__goto__": "complete"
    }


def _write_artifact_to_disk(state: EngineState, artifact: str) -> None:
    """Write the final artifact to the output directory."""
    from pathlib import Path

    # Find output directory from ingestion output or create one
    big_dump = state.get("context", {}).get("big_text_dump")
    if big_dump:
        output_dir = Path(big_dump).parent
    else:
        output_dir = Path("output/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the main artifact
    artifact_path = output_dir / "final_report.md"
    artifact_path.write_text(artifact, encoding="utf-8")
    logger.info(f"Wrote final artifact to {artifact_path} ({len(artifact)} chars)")


def _write_report_sections(state: EngineState, report: str) -> None:
    """Split report into section files for RAG retrievability."""
    import re
    from pathlib import Path

    big_dump = state.get("context", {}).get("big_text_dump")
    if big_dump:
        output_dir = Path(big_dump).parent
    else:
        output_dir = Path("output/artifacts")

    sections_dir = output_dir / "report_sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    # Split on ## headers (but not # main title)
    parts = re.split(r"(?=^## )", report, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract section name from ## header
        match = re.match(r"^## (.+)$", part, re.MULTILINE)
        if not match:
            continue
        section_name = match.group(1).lower().replace(" ", "_")[:40]
        section_path = sections_dir / f"{section_name}.md"
        section_path.write_text(part, encoding="utf-8")

    logger.info(f"Wrote {len(list(sections_dir.glob('*.md')))} report sections to {sections_dir}")
