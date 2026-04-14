"""Output Validator — post-pipeline quality checks.

Scans all output .md files and pipeline checkpoints for common quality issues
and returns structured warnings for logging and a validation report file.

Catches:
- Unresolved speaker/cluster IDs in output text
- ASR/VLM error strings that leaked into output
- Empty files and missing reports
- Pipeline failures: Neo4j down, empty identity maps, VLM circuit breaker trips
- Stale organized directories
- Checkpoint corruption
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 — used at runtime in function bodies


@dataclass
class ValidationWarning:
    """A single quality issue found in the output directory."""

    check: str  # e.g. "unresolved_speakers"
    severity: str  # "warning" | "error"
    message: str  # Human-readable description
    files: list[str] = field(default_factory=list)


# Patterns for raw IDs that should never appear in final output
_RAW_ID_PATTERN = re.compile(
    r"\b(SPEAKER_\d+|Global_Speaker_\d+|Speaker_\d+|Identity_Cluster_\d+)\b"
)

# Error strings from failed ASR/VLM/face recognition that leak into output
_ERROR_STRING_PATTERN = re.compile(
    r"\[(?:ASR failed|Image analysis failed|Video analysis failed|No frames)"
)

# File types that should be excluded from organized output
_PIPELINE_DIRS = frozenset({"pipeline"})


def _read_all_md(output_dir: Path) -> dict[str, str]:
    """Read all non-pipeline .md files into a dict keyed by relative path."""
    result: dict[str, str] = {}
    for md_file in output_dir.glob("**/*.md"):
        if "pipeline" in md_file.parts:
            continue
        if md_file.name == "validation_warnings.md":
            continue
        rel = str(md_file.relative_to(output_dir))
        result[rel] = md_file.read_text(encoding="utf-8", errors="ignore")
    return result


def validate_output(output_dir: Path) -> list[ValidationWarning]:
    """Scan output directory for quality issues."""
    warnings: list[ValidationWarning] = []

    md_files = _read_all_md(output_dir)

    # 1. Unresolved speaker/cluster IDs
    for rel_path, text in md_files.items():
        matches = _RAW_ID_PATTERN.findall(text)
        if matches:
            unique = sorted(set(matches))
            warnings.append(
                ValidationWarning(
                    check="unresolved_ids",
                    severity="error",
                    message=f"Found {len(unique)} unresolved IDs: {', '.join(unique)}",
                    files=[rel_path],
                )
            )

    # 2. Error strings from failed ASR/VLM
    for rel_path, text in md_files.items():
        matches = _ERROR_STRING_PATTERN.findall(text)
        if matches:
            warnings.append(
                ValidationWarning(
                    check="failed_service_calls",
                    severity="error",
                    message=(
                        f"Found {len(matches)} error strings from failed "
                        f"ASR/VLM/face-recognition calls in output. "
                        f"Examples: {sorted(set(matches))[:5]}"
                    ),
                    files=[rel_path],
                )
            )

    # 3. Empty files
    for md_file in output_dir.glob("*.md"):
        if md_file.name == "validation_warnings.md":
            continue
        if md_file.stat().st_size == 0:
            warnings.append(
                ValidationWarning(
                    check="empty_file",
                    severity="warning",
                    message=f"Empty file: {md_file.name}",
                    files=[md_file.name],
                )
            )

    # 4. Raw dict artifacts
    report = output_dir / "final_report.md"
    if report.exists():
        first_100 = report.read_text(encoding="utf-8", errors="ignore")[:100]
        if first_100.lstrip().startswith("{") or "{'findings':" in first_100:
            warnings.append(
                ValidationWarning(
                    check="raw_dict_artifact",
                    severity="error",
                    message="final_report.md appears to be raw dict output (basic aggregation fallback)",
                    files=["final_report.md"],
                )
            )

    # 5. Missing expected outputs
    if not report.exists():
        warnings.append(
            ValidationWarning(
                check="missing_report",
                severity="error",
                message="No final_report.md found",
                files=[],
            )
        )

    # 6. Pipeline checkpoint health checks
    pipeline_dir = output_dir / "pipeline"
    if pipeline_dir.exists():
        _check_pipeline_checkpoints(pipeline_dir, output_dir, warnings)

    # 7. Stale organized directory
    _check_organized_dir(output_dir, warnings)

    return warnings


def _check_pipeline_checkpoints(
    pipeline_dir: Path, output_dir: Path, warnings: list[ValidationWarning]
) -> None:
    """Check pipeline checkpoints for common failure patterns."""
    session_id = output_dir.name

    # --- Check for corrupted checkpoints ---
    for ckpt_file in pipeline_dir.glob("*.json"):
        try:
            with ckpt_file.open("r", encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(
                ValidationWarning(
                    check="corrupted_checkpoint",
                    severity="error",
                    message=f"Corrupted checkpoint {ckpt_file.name}: {e}",
                    files=[f"pipeline/{ckpt_file.name}"],
                )
            )

    # --- Phase 2: Empty identity_map when photos exist ---
    p2 = _load_checkpoint_safe(pipeline_dir, "phase2_embeddings")
    if p2 is not None:
        identity_map = p2.get("identity_map", {})
        # identity_map values are lists of cluster IDs
        total_face_entries = sum(len(v) if isinstance(v, list) else 1 for v in identity_map.values())
        if total_face_entries == 0:
            # Check if there were photo items
            p7 = _load_checkpoint_safe(pipeline_dir, "phase7_vlm_enriched")
            if p7:
                photo_count = sum(
                    1 for i in p7.get("items", []) if i.get("type") == "photo"
                )
                if photo_count > 10:
                    warnings.append(
                        ValidationWarning(
                            check="no_face_clusters",
                            severity="warning",
                            message=(
                                f"Phase 2 produced 0 face clusters from {photo_count} photos. "
                                f"CompreFace may be down or face detection is not finding faces."
                            ),
                            files=[],
                        )
                    )

    # --- Phase 6: Empty resolved identity map ---
    p6 = _load_checkpoint_safe(pipeline_dir, "phase6_identity_map")
    if p6 is not None:
        resolved_map = p6 if isinstance(p6, dict) else {}
        if not resolved_map:
            warnings.append(
                ValidationWarning(
                    check="empty_identity_map",
                    severity="error",
                    message="Phase 6 identity resolution produced an empty mapping. All speakers/faces remain unresolved.",
                    files=[],
                )
            )
        else:
            # Check for unresolved identity clusters
            total_clusters = 0
            resolved_clusters = 0
            if p2:
                for cluster_list in p2.get("identity_map", {}).values():
                    if isinstance(cluster_list, list):
                        for cid in cluster_list:
                            if isinstance(cid, str) and cid.startswith("Identity_Cluster_"):
                                total_clusters += 1
                                if cid in resolved_map:
                                    resolved_clusters += 1
            if total_clusters > 0 and resolved_clusters < total_clusters:
                unresolved = total_clusters - resolved_clusters
                warnings.append(
                    ValidationWarning(
                        check="partial_identity_resolution",
                        severity="warning",
                        message=(
                            f"Phase 6 resolved {resolved_clusters}/{total_clusters} face clusters. "
                            f"{unresolved} clusters remain as raw Identity_Cluster_N IDs."
                        ),
                        files=[],
                    )
                )

    # --- Phase 7: VLM failures ---
    p7 = _load_checkpoint_safe(pipeline_dir, "phase7_vlm_enriched")
    if p7:
        items = p7.get("items", [])
        reported_fails = p7.get("vlm_fail_count", 0)

        # Count actual error strings in descriptions
        actual_fails = 0
        for item in items:
            desc = item.get("description", "")
            if desc and _ERROR_STRING_PATTERN.search(desc):
                actual_fails += 1

        if actual_fails > 0:
            warnings.append(
                ValidationWarning(
                    check="vlm_failures_in_output",
                    severity="error" if actual_fails > len(items) * 0.3 else "warning",
                    message=(
                        f"{actual_fails}/{len(items)} items have failed VLM/ASR descriptions "
                        f"in their output (checkpoint reported {reported_fails} failures). "
                        f"VLM or ASR service may have been unavailable."
                    ),
                    files=[],
                )
            )

        # Check for circuit breaker trips
        processed = p7.get("processed_count", len(items))
        total = p7.get("total_count", 0)
        if total > 0 and processed < total:
            gap = total - processed
            warnings.append(
                ValidationWarning(
                    check="vlm_circuit_breaker",
                    severity="warning",
                    message=(
                        f"VLM processed {processed}/{total} items. "
                        f"{gap} items may have been skipped by circuit breaker."
                    ),
                    files=[],
                )
            )

    # --- Phase 5: Speaker unification check ---
    p5 = _load_checkpoint_safe(pipeline_dir, "phase5_speaker_map")
    p4 = _load_checkpoint_safe(pipeline_dir, "phase4_transcripts")
    if p4 and p5 is not None:
        # p5 is None if no checkpoint exists (single file or skipped)
        # If p5 exists but is empty dict, unification may have failed
        if isinstance(p5, dict) and not p5:
            audio_count = len(p4) if isinstance(p4, dict) else 0
            if audio_count > 1:
                warnings.append(
                    ValidationWarning(
                        check="speaker_unification_failed",
                        severity="warning",
                        message=(
                            f"Phase 5 speaker unification produced empty map "
                            f"with {audio_count} audio files. Speakers remain per-file."
                        ),
                        files=[],
                    )
                )


def _check_organized_dir(output_dir: Path, warnings: list[ValidationWarning]) -> None:
    """Check organized directory for stale data and broken symlinks."""
    organized = output_dir / "organized"
    if not organized.exists():
        return

    # Check for broken symlinks
    broken = []
    for link in organized.rglob("*"):
        if link.is_symlink() and not link.exists():
            broken.append(str(link.relative_to(organized)))
    if broken:
        warnings.append(
            ValidationWarning(
                check="broken_symlinks",
                severity="error",
                message=f"Found {len(broken)} broken symlinks in organized/. First 5: {broken[:5]}",
                files=broken[:5],
            )
        )

    # Check if by_person is all "Unknown"
    by_person = organized / "by_person"
    if by_person.exists():
        person_dirs = [d for d in by_person.iterdir() if d.is_dir()]
        if person_dirs:
            unknown_count = 0
            known_count = 0
            for pd in person_dirs:
                file_count = sum(1 for _ in pd.rglob("*") if _.is_symlink() or _.is_file())
                if pd.name.lower() == "unknown":
                    unknown_count += file_count
                else:
                    known_count += file_count
            total = unknown_count + known_count
            if total > 0 and unknown_count / total > 0.9:
                warnings.append(
                    ValidationWarning(
                        check="mostly_unknown",
                        severity="warning",
                        message=(
                            f"{unknown_count}/{total} organized files ({unknown_count * 100 // total}%) "
                            f"are in 'Unknown/'. Person attribution is very weak — "
                            f"face recognition or identity resolution may have failed."
                        ),
                        files=[],
                    )
                )


def _load_checkpoint_safe(pipeline_dir: Path, name: str) -> dict | list | None:
    """Load a checkpoint JSON safely, returning None on any error."""
    path = pipeline_dir / f"{name}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_validation_report(output_dir: Path, warnings: list[ValidationWarning]) -> Path:
    """Write a validation_warnings.md summarizing all issues found."""
    lines = ["# Output Validation Report\n"]
    errors = [w for w in warnings if w.severity == "error"]
    warns = [w for w in warnings if w.severity == "warning"]

    lines.append(f"**{len(errors)} errors, {len(warns)} warnings**\n")

    for w in warnings:
        icon = "ERROR" if w.severity == "error" else "WARN"
        lines.append(f"### [{icon}] {w.check}\n")
        lines.append(f"{w.message}\n")
        if w.files:
            lines.append("Files: " + ", ".join(f"`{f}`" for f in w.files))
        lines.append("")

    report_path = output_dir / "validation_warnings.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
