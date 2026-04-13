"""Detective Report — rich structured report from pipeline state.

Reads all structured data produced by ingestion, content clustering,
and Neo4j builder phases, then formats it into a clean markdown report
usable by humans and RAG systems.

No LLM calls needed — pure data formatting.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_loguru_logger

logger = get_loguru_logger("artifacts.detective_report")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Neo4jReportData:
    """Data fetched from Neo4j (may be None if unreachable)."""

    person_connections: dict[str, int] = field(default_factory=dict)
    person_rels: list[dict[str, Any]] = field(default_factory=list)
    topic_entities: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ReportData:
    """Normalized data extracted from pipeline state."""

    identity_map: dict[str, str] = field(default_factory=dict)
    all_people: list[str] = field(default_factory=list)
    all_locations: list[str] = field(default_factory=list)
    media_items: list[dict[str, Any]] = field(default_factory=list)
    content_clusters: list[dict[str, Any]] = field(default_factory=list)
    pre_named_entities: list[str] = field(default_factory=list)
    date_range: tuple[str, str] | None = None
    neo4j_data: Neo4jReportData | None = None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _merge_people_names(names: set[str], identity_map: dict[str, str]) -> list[str]:
    """Merge similar person names using identity map + first-name overlap.

    Rules:
    1. If identity_map maps alias X -> name Y, merge X into Y.
    2. If name A's first word matches name B's first word AND one is a
       prefix/substring of the other (e.g. "Scott" in "Scott Mjodr Gregorisson"),
       merge the shorter into the longer.
    3. NEVER merge names that are clearly different people even if they share
       a first name (e.g. "Tyler Oliveira" stays separate from "Tyler Dipeppe").
    """
    # Build canonical groups from identity_map
    canonical_for: dict[str, str] = {}
    for raw_id, resolved in identity_map.items():
        # Skip raw speaker IDs
        if re.match(r"^(SPEAKER_\d+|Global_Speaker_\d+|Speaker_\d+|file\d+:)", raw_id):
            continue
        canonical_for[raw_id] = resolved

    # Group: map canonical name -> set of aliases (including itself)
    groups: dict[str, set[str]] = defaultdict(set)
    for name in names:
        if name in canonical_for:
            groups[canonical_for[name]].add(name)
        else:
            groups[name].add(name)

    # Merge groups where one name's first word matches and it's a substring
    group_names = list(groups.keys())
    for i, name_a in enumerate(group_names):
        for name_b in group_names[i + 1:]:
            if _names_likely_same(name_a, name_b):
                # Merge shorter into longer
                canon = name_a if len(name_a) >= len(name_b) else name_b
                alias = name_b if canon == name_a else name_a
                groups[canon] |= groups.pop(alias, set())
                break

    return sorted(groups.keys())


def _names_likely_same(a: str, b: str) -> bool:
    """Check if two names likely refer to the same person.

    Conservative: only matches when one name is clearly a shortened form
    of the other (e.g. "Scott" -> "Scott Mjodr Gregorisson",
    "semok" -> "Christopher Anthony semok").
    Does NOT match "Tyler Oliveira" with "Tyler Dipeppe" or "Chris" with "Christopher".
    """
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if a_lower == b_lower:
        return True
    # Only merge if the shorter name is a complete word prefix OR suffix of the longer
    # i.e., "Scott Mjodr Gregorisson" starts with "Scott " (whole word)
    # OR "Christopher Anthony semok" ends with " semok" (last name match)
    short, long = (a_lower, b_lower) if len(a_lower) <= len(b_lower) else (b_lower, a_lower)
    return long.startswith(short + " ") or long.endswith(" " + short)


def _collect_people(
    identity_map: dict[str, str],
    entity_extractions: list[dict[str, Any]],
    pre_named: list[str],
    media_items: list[dict[str, Any]],
) -> list[str]:
    """Deduplicate people from all sources, merging name variations."""
    people = set(identity_map.values())
    people.update(pre_named)
    for batch in entity_extractions:
        for p in batch.get("people", []):
            if isinstance(p, str):
                people.add(p)
            elif isinstance(p, dict):
                people.add(p.get("name", ""))
    for item in media_items:
        for s in item.get("speakers", []):
            if s:
                people.add(s)
        for f in item.get("faces", []):
            if f:
                people.add(f)
        sender = item.get("sender", "")
        if sender and sender not in ("Unknown", ""):
            people.add(sender)
    _skip = re.compile(
        r"^(SPEAKER_\d+|Global_Speaker_\d+|Speaker_\d+|Unknown|Unknown Speaker \d+)$"
    )
    people = {p for p in people if p and not _skip.match(p)}
    return _merge_people_names(people, identity_map)


def _collect_locations(
    entity_extractions: list[dict[str, Any]],
) -> list[str]:
    """Deduplicate locations."""
    locs: set[str] = set()
    for batch in entity_extractions:
        for loc in batch.get("locations", []):
            if isinstance(loc, str):
                locs.add(loc)
            elif isinstance(loc, dict):
                locs.add(loc.get("name", ""))
    return sorted(loc for loc in locs if loc)


def _compute_date_range(
    media_items: list[dict[str, Any]],
) -> tuple[str, str] | None:
    """Find earliest and latest timestamps."""
    timestamps = []
    for item in media_items:
        ts = item.get("timestamp", "")
        if ts:
            timestamps.append(ts)
    if not timestamps:
        return None
    timestamps.sort()
    return timestamps[0], timestamps[-1]


def extract_report_data(state: dict[str, Any]) -> ReportData:
    """Pull and normalize all data from pipeline state."""
    ctx = state.get("context", {})
    structured = ctx.get("ingestion_structured", {})
    clusters = ctx.get("content_clusters", [])

    identity_map = structured.get("resolved_identity_map", {})
    entity_extractions = structured.get("entity_extractions", [])
    media_items = structured.get("media_items", [])
    pre_named = structured.get("pre_named_entities", [])

    all_people = _collect_people(identity_map, entity_extractions, pre_named, media_items)
    all_locations = _collect_locations(entity_extractions)
    date_range = _compute_date_range(media_items)

    return ReportData(
        identity_map=identity_map,
        all_people=all_people,
        all_locations=all_locations,
        media_items=media_items,
        content_clusters=clusters,
        pre_named_entities=pre_named,
        date_range=date_range,
    )


# ---------------------------------------------------------------------------
# Neo4j enrichment (optional)
# ---------------------------------------------------------------------------


async def fetch_neo4j_data(config_data: dict[str, Any]) -> Neo4jReportData | None:
    """Fetch enrichment data from Neo4j. Returns None if unreachable."""
    from ..tools.storage import get_neo4j_from_config

    namespace = config_data.get("storage", {}).get("neo4j", {}).get("namespace", "default")

    storage = await get_neo4j_from_config(config_data)
    if storage is None:
        return None

    try:
        neo4j = Neo4jReportData()

        # Person connection counts — only Person nodes
        _skip_name = re.compile(r"^(Unknown|Unknown Speaker \d+)$")
        rows = await storage.execute_cypher(
            "MATCH (p:Person) WHERE p.namespace = $namespace "
            "RETURN p.name as name, count {(p)--()} as connections "
            "ORDER BY connections DESC LIMIT 50",
            {"namespace": namespace},
        )
        for r in rows:
            if "error" not in r and r.get("name") and not _skip_name.match(r["name"]):
                neo4j.person_connections[r["name"]] = r.get("connections", 0)

        # Person-to-person via shared media — only Person nodes on both sides
        rows = await storage.execute_cypher(
            "MATCH (p1:Person)-[r1]-(m)-[r2]-(p2:Person) "
            "WHERE p1.name < p2.name AND p1.namespace = $namespace "
            "RETURN p1.name as person_a, p2.name as person_b, "
            "count(DISTINCT m) as shared_media, "
            "collect(DISTINCT type(r1)) as rel_types "
            "ORDER BY shared_media DESC LIMIT 100",
            {"namespace": namespace},
        )
        for r in rows:
            if "error" not in r:
                a = r.get("person_a", "")
                b = r.get("person_b", "")
                # Skip Unknown Speaker IDs
                if _skip_name.match(a) or _skip_name.match(b):
                    continue
                neo4j.person_rels.append(r)

        # Topic entities
        rows = await storage.execute_cypher(
            "MATCH (p)-[:BELONGS_TO]->(t) "
            "WHERE t.namespace = $namespace "
            "RETURN t.name as topic, collect(p.name) as entities",
            {"namespace": namespace},
        )
        for r in rows:
            if "error" not in r and r.get("topic"):
                neo4j.topic_entities[r["topic"]] = r.get("entities", [])

        return neo4j
    except Exception as e:
        logger.warning(f"Neo4j enrichment failed (non-fatal): {e}")
        return None
    finally:
        with __import__("contextlib").suppress(Exception):
            await storage.cleanup()


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def build_executive_summary(data: ReportData) -> str:
    """Build executive summary section."""
    lines = ["## Executive Summary\n"]

    # Counts
    n_media = len(data.media_items)
    n_voice = sum(1 for m in data.media_items if m.get("type") == "voice")
    n_photo = sum(1 for m in data.media_items if m.get("type") == "photo")
    n_video = sum(1 for m in data.media_items if m.get("type") == "video")
    n_text = sum(1 for m in data.media_items if m.get("type") in ("text", "message"))
    n_people = len(data.all_people)
    n_locations = len(data.all_locations)
    n_clusters = len(data.content_clusters)

    lines.append(f"- **{n_media}** items processed")
    if n_text:
        lines.append(f"  - {n_text} text messages")
    if n_voice:
        lines.append(f"  - {n_voice} voice messages")
    if n_photo:
        lines.append(f"  - {n_photo} photos")
    if n_video:
        lines.append(f"  - {n_video} videos")
    lines.append(f"- **{n_people}** unique people identified")
    if n_locations:
        lines.append(f"- **{n_locations}** unique locations mentioned")
    if n_clusters:
        lines.append(f"- **{n_clusters}** content topics discovered")

    if data.date_range:
        lines.append(f"- Date range: **{data.date_range[0]}** to **{data.date_range[1]}**")

    if data.neo4j_data and data.neo4j_data.person_connections:
        lines.append(f"- Neo4j graph: **{len(data.neo4j_data.person_connections)}** connected entities")

    return "\n".join(lines)


def build_people_directory(data: ReportData) -> str:
    """Build per-person directory section."""
    if not data.all_people:
        return "## People Directory\n\nNo people identified."

    lines = ["## People Directory\n"]

    # Compute per-person stats
    person_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for item in data.media_items:
        for person in item.get("speakers", []):
            if person and person != "Unknown":
                person_stats[person]["voice"] += 1
        for person in item.get("faces", []):
            if person and person != "Unknown":
                itype = item.get("type", "photo")
                person_stats[person][itype] += 1
        sender = item.get("sender", "")
        if sender and sender != "Unknown":
            itype = item.get("type", "text")
            person_stats[sender][itype] += 1

    # Build aliases map (reverse identity_map)
    # Skip raw speaker IDs and metadata keys that would pollute the report
    _skip_alias = re.compile(
        r"^(SPEAKER_\d+|Global_Speaker_\d+|Speaker_\d+)$|:name_reference"
    )
    aliases: dict[str, list[str]] = defaultdict(list)
    for cluster_id, name in data.identity_map.items():
        if _skip_alias.search(cluster_id):
            continue
        aliases[name].append(cluster_id)

    # Sort by total appearances
    sorted_people = sorted(
        data.all_people,
        key=lambda p: sum(person_stats.get(p, {}).values()),
        reverse=True,
    )

    for person in sorted_people:
        stats = person_stats.get(person, {})
        total = sum(stats.values())
        lines.append(f"### {person}\n")

        # Aliases
        person_aliases = aliases.get(person, [])
        if person_aliases:
            lines.append(f"- Aliases: {', '.join(person_aliases)}")

        # Appearances
        parts = []
        if stats.get("voice"):
            parts.append(f"{stats['voice']} voice")
        if stats.get("photo"):
            parts.append(f"{stats['photo']} photos")
        if stats.get("video"):
            parts.append(f"{stats['video']} videos")
        if stats.get("text"):
            parts.append(f"{stats['text']} messages")
        if stats.get("message"):
            parts.append(f"{stats['message']} messages")
        if parts:
            lines.append(f"- Appearances: {', '.join(parts)} ({total} total)")

        # Neo4j connections
        if data.neo4j_data and person in data.neo4j_data.person_connections:
            conn = data.neo4j_data.person_connections[person]
            lines.append(f"- Graph connections: {conn}")

        lines.append("")

    return "\n".join(lines)


def build_topic_breakdown(data: ReportData) -> str:
    """Build content clusters section."""
    if not data.content_clusters:
        return "## Topic Breakdown\n\nNo topics discovered."

    lines = ["## Topic Breakdown\n"]

    for i, cluster in enumerate(data.content_clusters, 1):
        name = cluster.get("name", f"Topic {i}")
        summary = cluster.get("summary", "")
        labels = cluster.get("suggested_labels", [])
        entities = cluster.get("key_entities", [])

        lines.append(f"### {i}. {name}\n")
        if summary:
            lines.append(f"{summary}\n")
        if labels:
            lines.append(f"- Labels: {', '.join(labels)}")
        if entities:
            lines.append(f"- Key entities: {', '.join(str(e) for e in entities)}")

        # Neo4j enrichment
        if data.neo4j_data:
            graph_entities = data.neo4j_data.topic_entities.get(name, [])
            if graph_entities:
                lines.append(f"- Graph entities: {', '.join(graph_entities)}")

        lines.append("")

    return "\n".join(lines)


def build_media_inventory(data: ReportData) -> str:
    """Build media tables section."""
    lines = ["## Media Inventory\n"]

    voice_items = [m for m in data.media_items if m.get("type") == "voice"]
    photo_items = [m for m in data.media_items if m.get("type") == "photo"]
    video_items = [m for m in data.media_items if m.get("type") == "video"]
    text_items = [m for m in data.media_items if m.get("type") in ("text", "message")]

    # Text messages
    if text_items:
        lines.append(f"### Text Messages ({len(text_items)})\n")
        lines.append("| # | Timestamp | Sender | Text |")
        lines.append("|---|-----------|--------|------|")
        for i, item in enumerate(text_items, 1):
            ts = item.get("timestamp", "")[:19]
            sender = item.get("sender", "Unknown")
            text = (item.get("text", "") or "")[:150].replace("|", "/").replace("\n", " ")
            if len(item.get("text", "") or "") > 150:
                text += "..."
            lines.append(f"| {i} | {ts} | {sender} | {text} |")
        lines.append("")

    # Voice messages
    lines.append(f"### Voice Messages ({len(voice_items)})\n")
    if voice_items:
        lines.append("| # | Timestamp | Sender | Speakers | Transcript |")
        lines.append("|---|-----------|--------|----------|------------|")
        for i, item in enumerate(voice_items, 1):
            ts = item.get("timestamp", "")[:19]
            sender = item.get("sender", "Unknown")
            speakers = ", ".join(item.get("speakers", []))
            transcript = (item.get("transcript", "") or "")[:120].replace("|", "/")
            if len(item.get("transcript", "") or "") > 120:
                transcript += "..."
            lines.append(f"| {i} | {ts} | {sender} | {speakers} | {transcript} |")
    else:
        lines.append("No voice messages.")
    lines.append("")

    # Photos
    lines.append(f"### Photos ({len(photo_items)})\n")
    if photo_items:
        lines.append("| # | Timestamp | Sender | Faces | Description |")
        lines.append("|---|-----------|--------|-------|-------------|")
        for i, item in enumerate(photo_items, 1):
            ts = item.get("timestamp", "")[:19]
            sender = item.get("sender", "Unknown")
            faces = ", ".join(item.get("faces", []))
            desc = (item.get("description", "") or "")[:120].replace("|", "/")
            if len(item.get("description", "") or "") > 120:
                desc += "..."
            lines.append(f"| {i} | {ts} | {sender} | {faces} | {desc} |")
    else:
        lines.append("No photos.")
    lines.append("")

    # Videos
    lines.append(f"### Videos ({len(video_items)})\n")
    if video_items:
        lines.append("| # | Timestamp | Sender | Faces | Description |")
        lines.append("|---|-----------|--------|-------|-------------|")
        for i, item in enumerate(video_items, 1):
            ts = item.get("timestamp", "")[:19]
            sender = item.get("sender", "Unknown")
            faces = ", ".join(item.get("faces", []))
            desc = (item.get("description", "") or "")[:120].replace("|", "/")
            if len(item.get("description", "") or "") > 120:
                desc += "..."
            lines.append(f"| {i} | {ts} | {sender} | {faces} | {desc} |")
    else:
        lines.append("No videos.")

    return "\n".join(lines)


def build_timeline(data: ReportData) -> str:
    """Build chronological timeline section."""
    lines = ["## Timeline\n"]

    if not data.media_items:
        lines.append("No timestamped items.")
        return "\n".join(lines)

    # Group items by date
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data.media_items:
        ts = item.get("timestamp", "")
        if ts:
            date = ts[:10]
            by_date[date].append(item)

    if not by_date:
        lines.append("No timestamped items.")
        return "\n".join(lines)

    for date in sorted(by_date.keys()):
        items = by_date[date]
        counts: dict[str, int] = Counter(m.get("type", "text") for m in items)
        parts = [f"{v} {k}" + ("s" if v > 1 else "") for k, v in sorted(counts.items())]
        lines.append(f"### {date} ({len(items)} items)\n")
        lines.append(", ".join(parts))

        # Show notable media (not text)
        notable = [m for m in items if m.get("type") not in ("text", "message") and (m.get("transcript") or m.get("description"))]
        if notable:
            lines.append("")
            for m in notable[:5]:
                itype = m.get("type", "media")
                sender = m.get("sender", "Unknown")
                if itype == "voice":
                    snippet = (m.get("transcript", "") or "")[:100]
                else:
                    snippet = (m.get("description", "") or "")[:100]
                if snippet:
                    lines.append(f"- [{itype}] {sender}: {snippet}...")
        lines.append("")

    return "\n".join(lines)


def build_relationship_map(data: ReportData) -> str:
    """Build person-to-person relationship map."""
    lines = ["## Relationship Map\n"]

    # Compute co-occurrence from state data
    pair_counts: Counter[str] = Counter()

    for item in data.media_items:
        people_in_item: set[str] = set()
        for s in item.get("speakers", []):
            if s and s != "Unknown":
                people_in_item.add(s)
        for f in item.get("faces", []):
            if f and f != "Unknown":
                people_in_item.add(f)
        sender = item.get("sender", "")
        if sender and sender != "Unknown":
            people_in_item.add(sender)

        people_list = sorted(people_in_item)
        for i in range(len(people_list)):
            for j in range(i + 1, len(people_list)):
                pair = f"{people_list[i]} <-> {people_list[j]}"
                pair_counts[pair] += 1

    # Merge with Neo4j data — only include pairs where both names are known people
    people_set = set(data.all_people) if data.all_people else set()
    neo4j_pairs: dict[str, dict[str, Any]] = {}
    if data.neo4j_data:
        for rel in data.neo4j_data.person_rels:
            a = rel.get("person_a", "")
            b = rel.get("person_b", "")
            # Skip non-person entities (organizations, locations, dates, etc.)
            if people_set and (a not in people_set or b not in people_set):
                continue
            if a and b:
                key = f"{a} <-> {b}" if a < b else f"{b} <-> {a}"
                neo4j_pairs[key] = rel

    # Combine
    all_pairs: set[str] = set(pair_counts.keys()) | set(neo4j_pairs.keys())

    if not all_pairs:
        lines.append("No relationships discovered.")
        return "\n".join(lines)

    sorted_pairs = sorted(all_pairs, key=lambda p: pair_counts.get(p, 0), reverse=True)

    for pair in sorted_pairs:
        count = pair_counts.get(pair, 0)
        neo4j_rel = neo4j_pairs.get(pair)
        parts = []
        if count:
            parts.append(f"{count} shared media")
        if neo4j_rel:
            rel_types = neo4j_rel.get("rel_types", [])
            if rel_types:
                parts.append(f"graph: {', '.join(rel_types)}")
        detail = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"- {pair}{detail}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def build_detective_report(state: dict[str, Any]) -> str:
    """Build the full detective report from pipeline state."""
    data = extract_report_data(state)

    # Try Neo4j enrichment
    config_data = state.get("context", {}).get("config", {})
    data.neo4j_data = await fetch_neo4j_data(config_data)

    sections = [
        "# Detective Report: Social Archive Analysis\n",
        build_executive_summary(data),
        build_people_directory(data),
        build_topic_breakdown(data),
        build_media_inventory(data),
        build_timeline(data),
        build_relationship_map(data),
    ]

    return "\n\n".join(sections)
