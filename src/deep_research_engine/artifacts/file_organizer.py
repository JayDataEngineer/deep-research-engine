"""File Organizer — arranges media files into browsable directory trees.

Creates symlink-based directory structures from the structured media items
produced by the ingestion pipeline:

  organized/
  ├── by_person/{name}/photos/|videos/|voice/
  ├── by_type/all_photos/|all_videos/|all_voice/
  ├── by_date/{YYYY-MM-DD}/photos/|videos/|voice/
  └── catalog.md

No LLM calls needed — pure file system operations.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..utils.logging import get_loguru_logger

logger = get_loguru_logger("artifacts.file_organizer")

# Characters unsafe in directory names
_UNSAFE = set('/\\:*?"<>|\n\r\t')

# Raw speaker IDs that should never appear as person names
_RAW_SPEAKER_ID = re.compile(r"^(SPEAKER_\d+|Global_Speaker_\d+|Speaker_\d+)$")


def _safe_name(name: str) -> str:
    """Make a name safe for use as a directory name."""
    safe = "".join(c if c not in _UNSAFE else "_" for c in name)
    safe = safe.strip(" .")
    return safe[:80] if safe else "unknown"


def _relative_symlink(src: Path, dst: Path) -> None:
    """Create a relative symlink from dst -> src."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(str(src), str(dst.parent))
    dst.symlink_to(rel)


def organize_files(
    output_dir: Path,
    media_items: list[dict[str, Any]],
    input_dir: Path | None = None,
) -> dict[str, Any]:
    """Organize media files into browsable directory structures.

    Args:
        output_dir: Base output directory (e.g. output/ingested/{session}/).
        media_items: Structured media items from ingestion pipeline.
        input_dir: Original data directory for resolving relative paths.

    Returns:
        Stats dict with counts of organized files.
    """
    organized_dir = output_dir / "organized"
    organized_dir.mkdir(parents=True, exist_ok=True)
    # Only skip if actual organized subdirectories exist (not just catalog.md)
    if (organized_dir / "by_person").exists() or (organized_dir / "by_type").exists():
        logger.info(f"Organized directory already exists at {organized_dir}")
        return {"status": "already_exists", "path": str(organized_dir)}

    input_dir = input_dir or Path()

    # Collect per-person, per-type, per-date items
    by_person: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_date: dict[str, list[dict]] = defaultdict(list)

    for item in media_items:
        path_rel = item.get("path")
        if not path_rel:
            continue
        item_type = item.get("type", "unknown")
        if item_type == "message":
            continue  # text messages have no files

        # Resolve absolute source path
        src = Path(path_rel)
        if not src.is_absolute():
            src = input_dir / path_rel
        if not src.exists():
            continue

        # Per-type
        type_key = {"voice": "voice", "photo": "photos", "video": "videos"}.get(item_type, "other")
        by_type[type_key].append({**item, "_src": src})

        # Per-person: collect from sender, speakers, faces
        people: set[str] = set()
        sender = item.get("sender", "")
        if sender and sender != "Unknown" and not _RAW_SPEAKER_ID.match(sender):
            people.add(sender)
        for s in item.get("speakers", []):
            if s and s != "Unknown" and not _RAW_SPEAKER_ID.match(s):
                people.add(s)
        for f in item.get("faces", []):
            if f and f != "Unknown" and not _RAW_SPEAKER_ID.match(f):
                people.add(f)
        if people:
            for person in people:
                by_person[person].append({**item, "_src": src})
        else:
            by_person["Unknown"].append({**item, "_src": src})

        # Per-date
        ts = item.get("timestamp", "")
        date = ts[:10] if ts and ts[:4].isdigit() else "unknown"
        by_date[date].append({**item, "_src": src})

    # Create directory trees
    link_count = 0

    # by_type/
    for type_key, items in by_type.items():
        for item in items:
            src: Path = item["_src"]
            dst = organized_dir / "by_type" / f"all_{type_key}" / src.name
            _relative_symlink(src, dst)
            link_count += 1

    # by_person/{name}/{type}/
    for person, items in by_person.items():
        safe = _safe_name(person)
        for item in items:
            src = item["_src"]
            item_type = item.get("type", "unknown")
            type_dir = {
                "voice": "voice",
                "photo": "photos",
                "video": "videos",
            }.get(item_type, "other")
            dst = organized_dir / "by_person" / safe / type_dir / src.name
            _relative_symlink(src, dst)
            link_count += 1

    # by_date/{date}/{type}/
    for date, items in by_date.items():
        for item in items:
            src = item["_src"]
            item_type = item.get("type", "unknown")
            type_dir = {
                "voice": "voice",
                "photo": "photos",
                "video": "videos",
            }.get(item_type, "other")
            dst = organized_dir / "by_date" / date / type_dir / src.name
            _relative_symlink(src, dst)
            link_count += 1

    # Write catalog.md
    _write_catalog(organized_dir, by_person, by_type, by_date)

    stats = {
        "link_count": link_count,
        "person_count": len(by_person),
        "date_count": len(by_date),
        "type_counts": {k: len(v) for k, v in by_type.items()},
    }
    logger.info(
        f"Organized {link_count} links for {len(by_person)} people, "
        f"{len(by_date)} dates into {organized_dir}"
    )
    return stats


def _write_catalog(
    organized_dir: Path,
    by_person: dict[str, list[dict]],
    by_type: dict[str, list[dict]],
    by_date: dict[str, list[dict]],
) -> None:
    """Write a master catalog.md listing all organized files."""
    lines = ["# Organized Media Catalog\n"]

    # By person
    lines.append("## By Person\n")
    for person in sorted(by_person.keys()):
        items = by_person[person]
        safe = _safe_name(person)
        type_counts: dict[str, int] = defaultdict(int)
        for item in items:
            t = item.get("type", "other")
            type_counts[t] += 1
        parts = [f"{v} {k}" + ("s" if v > 1 else "") for k, v in sorted(type_counts.items())]
        lines.append(f"- **{person}** ({', '.join(parts)}) → `by_person/{safe}/`")

    lines.append("")

    # By type
    lines.append("## By Type\n")
    for type_key in sorted(by_type.keys()):
        items = by_type[type_key]
        lines.append(f"- **{type_key}**: {len(items)} files → `by_type/all_{type_key}/`")

    lines.append("")

    # By date
    lines.append("## By Date\n")
    for date in sorted(by_date.keys()):
        items = by_date[date]
        type_counts = defaultdict(int)
        for item in items:
            type_counts[item.get("type", "other")] += 1
        parts = [f"{v} {k}" + ("s" if v > 1 else "") for k, v in sorted(type_counts.items())]
        lines.append(f"- **{date}**: {', '.join(parts)} → `by_date/{date}/`")

    lines.append("")

    catalog_path = organized_dir / "catalog.md"
    catalog_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote catalog to {catalog_path}")
