"""Output Validator — post-pipeline quality checks.

Scans all output .md files for common quality issues and returns
structured warnings for logging and a validation report file.
"""

from __future__ import annotations

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


def validate_output(output_dir: Path) -> list[ValidationWarning]:
    """Scan output directory for quality issues."""
    warnings: list[ValidationWarning] = []

    # 1. Unresolved speaker IDs
    speaker_pattern = re.compile(r"\b(SPEAKER_\d+|Global_Speaker_\d+|Speaker_\d+)\b")
    for md_file in output_dir.glob("**/*.md"):
        if "pipeline" in md_file.parts:
            continue
        # Skip the validation report itself (self-referential)
        if md_file.name == "validation_warnings.md":
            continue
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        matches = speaker_pattern.findall(text)
        if matches:
            unique = sorted(set(matches))
            warnings.append(
                ValidationWarning(
                    check="unresolved_speakers",
                    severity="error",
                    message=f"Found {len(unique)} unresolved speaker IDs: {', '.join(unique)}",
                    files=[str(md_file.relative_to(output_dir))],
                )
            )

    # 2. Empty files
    warnings.extend(
        ValidationWarning(
            check="empty_file",
            severity="warning",
            message=f"Empty file: {md_file.name}",
            files=[md_file.name],
        )
        for md_file in output_dir.glob("*.md")
        if md_file.stat().st_size == 0
    )

    # 3. Raw dict artifacts
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

    # 4. Missing expected outputs
    if not report.exists():
        warnings.append(
            ValidationWarning(
                check="missing_report",
                severity="error",
                message="No final_report.md found",
                files=[],
            )
        )

    return warnings


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
