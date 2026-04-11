"""Neo4j Builder Node - Extracts triplets and writes structured data to Neo4j.

Uses Neo4jStorage (async driver) exclusively. No raw GraphDatabase.
- Structured ingestion data (speakers, entities, media) written directly without LLM.
- Semantic relationships extracted via LLM with dynamic schema (no hardcoded ontology).
"""

import json
from typing import Any

from ..config import get_phase_config
from ..shared import get_langchain_llm
from ..state import EngineState
from ..storage.backends.neo4j import Neo4jStorage
from ..tools.storage import initialize_storage_tools
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context

logger = get_loguru_logger("nodes.neo4j_builder")


@node_context("neo4j_builder")
async def neo4j_builder_node(state: EngineState) -> dict[str, Any]:
    """Neo4j Builder node - writes structured data + LLM-extracted triplets."""
    logger.info("Starting Neo4j Builder node")

    # 1. Get configuration
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "neo4j_builder")

    neo4j_enabled = phase_config.get("enabled", False)
    if not neo4j_enabled:
        neo4j_enabled = config_data.get("storage", {}).get("neo4j", {}).get("enabled", False)

    if not neo4j_enabled:
        logger.info("Skipping Neo4j Builder (not enabled).")
        return {"__goto__": "artifact_maker"}

    # 2. Initialize Neo4jStorage from config
    storage = await _get_neo4j_storage(config_data)
    if storage is None:
        logger.warning("Neo4j storage not available. Skipping builder.")
        return {"__goto__": "artifact_maker"}

    namespace = config_data.get("storage", {}).get("neo4j", {}).get("namespace", "default")
    stats = {"nodes": 0, "rels": 0}

    try:
        # Initialize storage tools for vector_search_datastore
        initialize_storage_tools(config_data)

        # 3. Write structured ingestion data directly (no LLM needed)
        structured = state.get("context", {}).get("ingestion_structured", {})
        if structured:
            logger.info("Writing structured ingestion data to Neo4j...")
            s_result = await _write_structured_data(storage, structured, namespace)
            stats["nodes"] += s_result.get("nodes", 0)
            stats["rels"] += s_result.get("rels", 0)

        # 3.5. Write Topic nodes from content clusters (if available)
        content_clusters = state.get("context", {}).get("content_clusters", [])
        if content_clusters:
            logger.info(f"Writing {len(content_clusters)} content clusters as Topic nodes...")
            c_result = await _write_content_clusters(storage, content_clusters, namespace)
            stats["nodes"] += c_result.get("nodes", 0)
            stats["rels"] += c_result.get("rels", 0)

        # 4. Extract semantic triplets via LLM (dynamic schema)
        artifact_output = ""
        for result in state.get("worker_results", []):
            if result.get("output"):
                artifact_output += result["output"] + "\n"

        if not artifact_output:
            # Fall back to big_text_dump
            dump_path = state.get("context", {}).get("big_text_dump")
            if dump_path:
                from pathlib import Path

                dump_file = Path(dump_path)
                if dump_file.exists():
                    artifact_output = dump_file.read_text()[:20000]

        if artifact_output:
            logger.info("Extracting semantic triplets via LLM...")
            existing_labels = await storage.get_existing_labels()
            triplets = await _extract_triplets_dynamic(
                artifact_output, config_data, existing_labels
            )

            if triplets:
                t_result = await _write_triplets(storage, triplets, namespace)
                stats["nodes"] += t_result.get("nodes", 0)
                stats["rels"] += t_result.get("rels", 0)

        logger.info(
            f"Neo4j Builder complete: {stats['nodes']} nodes, {stats['rels']} relationships"
        )
        return {
            "neo4j_result": f"Wrote {stats['nodes']} nodes and {stats['rels']} relationships to Neo4j.",
            "__goto__": "artifact_maker",
        }

    except Exception as e:
        logger.error(f"Neo4j Builder error: {e}", exc_info=True)
        return {"neo4j_error": str(e), "__goto__": "artifact_maker"}
    finally:
        await storage.cleanup()


async def _get_neo4j_storage(config_data: dict[str, Any]) -> Neo4jStorage | None:
    """Create and initialize Neo4jStorage from workflow config."""
    from ..tools.storage import get_neo4j_from_config

    return await get_neo4j_from_config(config_data)


async def _write_structured_data(
    storage: Neo4jStorage,
    structured: dict[str, Any],
    namespace: str,
) -> dict[str, int]:
    """Write structured ingestion data directly to Neo4j (no LLM).

    Returns dict with nodes/rels counts.
    """
    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    # Resolved identities → Person nodes
    for cluster_id, name in structured.get("resolved_identity_map", {}).items():
        nodes.append(
            {
                "labels": ["Person"],
                "properties": {"name": name, "cluster_id": str(cluster_id)},
                "merge_on": ["name"],
            }
        )

    # NER entities → typed nodes
    for extraction in structured.get("entity_extractions", []):
        nodes.extend(
            {
                "labels": ["Person"],
                "properties": {"name": person},
                "merge_on": ["name"],
            }
            for person in extraction.get("people", [])
        )
        nodes.extend(
            {
                "labels": ["Location"],
                "properties": {"name": location},
                "merge_on": ["name"],
            }
            for location in extraction.get("locations", [])
        )

    # Media items → Media nodes + relationships
    for item in structured.get("media_items", []):
        if not item.get("path"):
            continue
        item_type = item.get("type", "Media")
        media_props: dict[str, Any] = {
            "path": item["path"],
            "media_type": item_type,
        }
        if item.get("timestamp"):
            media_props["timestamp"] = item["timestamp"]
        if item.get("sender"):
            media_props["sender"] = item["sender"]
        if item.get("transcript"):
            media_props["transcript"] = item["transcript"]
        if item.get("description"):
            media_props["description"] = item["description"]

        nodes.append(
            {
                "labels": ["Media"],
                "properties": media_props,
                "merge_on": ["path"],
            }
        )

        # Sender → Media (SENT)
        if item.get("sender") and item["sender"] != "Unknown":
            relationships.append(
                {
                    "source_labels": ["Person"],
                    "source_match": {"name": item["sender"]},
                    "target_labels": ["Media"],
                    "target_match": {"path": item["path"]},
                    "rel_type": "SENT",
                    "properties": {"timestamp": item.get("timestamp", "")},
                }
            )
            # Ensure sender Person node exists
            nodes.append(
                {
                    "labels": ["Person"],
                    "properties": {"name": item["sender"]},
                    "merge_on": ["name"],
                }
            )

        # Speakers → Media (SPOKE_IN)
        relationships.extend(
            {
                "source_labels": ["Person"],
                "source_match": {"name": speaker},
                "target_labels": ["Media"],
                "target_match": {"path": item["path"]},
                "rel_type": "SPOKE_IN",
            }
            for speaker in item.get("speakers", [])
        )

        # Faces/identities → Media (APPEARS_IN)
        relationships.extend(
            {
                "source_labels": ["Person"],
                "source_match": {"name": face_cluster},
                "target_labels": ["Media"],
                "target_match": {"path": item["path"]},
                "rel_type": "APPEARS_IN",
            }
            for face_cluster in item.get("faces", [])
        )

    if not nodes and not relationships:
        return {"nodes": 0, "rels": 0}

    result = await storage.bulk_write(nodes, relationships, namespace)
    meta = result.metadata or {}
    return {"nodes": meta.get("nodes_written", 0), "rels": meta.get("rels_written", 0)}


async def _extract_triplets_dynamic(
    text: str,
    config: dict[str, Any],
    existing_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract triplets using LLMGraphTransformer with dynamic schema.

    Uses ``langchain-experimental.LLMGraphTransformer`` for structured
    entity/relationship extraction. Falls back to the legacy DeepAgents
    approach if the optional dependency is not installed.
    """
    llm = get_langchain_llm(enable_thinking=False, temperature=0.3, config_data=config)

    # Try LLMGraphTransformer first (cleaner, structured output)
    try:
        from langchain_core.documents import Document
        from langchain_experimental.graph_transformers import LLMGraphTransformer

        transformer = LLMGraphTransformer(llm=llm)
        doc = Document(page_content=text[:8000])
        graph_docs = await transformer.aconvert_to_graph_documents([doc])

        if not graph_docs:
            return []

        triplets: list[dict[str, Any]] = []
        for rel in graph_docs[0].relationships:
            subj_name = rel.source.id if hasattr(rel.source, "id") else str(rel.source)
            subj_label = rel.source.type if hasattr(rel.source, "type") else "Entity"
            obj_name = rel.target.id if hasattr(rel.target, "id") else str(rel.target)
            obj_label = rel.target.type if hasattr(rel.target, "type") else "Entity"
            predicate = rel.type

            triplets.append(
                {
                    "subject": {"name": subj_name, "label": subj_label},
                    "predicate": predicate,
                    "object": {"name": obj_name, "label": obj_label},
                }
            )

        logger.info(f"LLMGraphTransformer extracted {len(triplets)} triplets")
        return triplets

    except ImportError:
        logger.info("langchain-experimental not installed, using fallback extraction")
    except Exception as e:
        logger.warning(f"LLMGraphTransformer failed ({e}), using fallback extraction")

    # Fallback: legacy DeepAgents approach
    return await _extract_triplets_legacy(text, config, existing_labels)


async def _extract_triplets_legacy(
    text: str,
    config: dict[str, Any],
    existing_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Triplet extraction via direct InfraClient.chat() — avoids DeepAgents retry loops."""
    import asyncio
    import re

    from ..shared.config import INGESTION_MODEL
    from ..shared.infra_client import InfraClient

    labels_hint = ""
    if existing_labels:
        labels_hint = (
            f"\nExisting labels in the graph (REUSE these when applicable): {existing_labels}"
        )

    system_prompt = (
        "Extract (Subject, Predicate, Object) triplets from the text.\n\n"
        "DYNAMIC SCHEMA:\n"
        "- Invent appropriate node labels based on the content domain "
        "(e.g. Person, Organization, Technology, Location, Event, Concept).\n"
        f"{labels_hint}\n"
        "- Relationship types should be descriptive verbs in UPPERCASE_SNAKE_CASE "
        "(e.g. WORKS_AT, MENTIONED_IN, DEVELOPED_BY).\n\n"
        "RULES:\n"
        "- Deduplicate entities (e.g. 'US' vs 'USA' → use one canonical form).\n"
        "- Only extract relationships explicitly stated in the text.\n"
        "- Predicates should be descriptive verbs.\n\n"
        'Respond with JSON: {"triplets": [{"subject": {"name": "...", "label": "..."},'
        ' "predicate": "...", "object": {"name": "...", "label": "..."}}]}'
    )

    # Chunk text for extraction (8K chars per chunk to fit context)
    chunk_size = 8000
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        if end < len(text):
            para_break = text.rfind("\n\n", pos, end)
            if para_break > pos:
                end = para_break + 2
        chunks.append(text[pos:end])
        pos = end

    all_triplets: list[dict[str, Any]] = []

    infra = InfraClient()
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        logger.info(f"Triplet extraction chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        try:
            response = await asyncio.to_thread(
                infra.chat,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Extract knowledge graph triplets from the following text "
                            f"(part {i + 1} of {len(chunks)}):\n\n{chunk}"
                        ),
                    },
                ],
                model=INGESTION_MODEL,
                max_tokens=8000,
            )
            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.warning(f"Triplet extraction chunk {i + 1} failed: {e}")
            continue

        if not response_text:
            continue

        # Parse JSON from response
        try:
            # Extract JSON from markdown code block
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()

            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "triplets" in parsed:
                all_triplets.extend(parsed["triplets"])
            elif isinstance(parsed, list):
                all_triplets.extend(parsed)
        except (json.JSONDecodeError, ValueError):
            # Try to find bare JSON object
            brace_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if brace_match:
                try:
                    parsed = json.loads(brace_match.group(0))
                    if isinstance(parsed, dict) and "triplets" in parsed:
                        all_triplets.extend(parsed["triplets"])
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Could not parse triplets from chunk {i + 1}")

    logger.info(f"Triplet extraction complete: {len(all_triplets)} triplets from {len(chunks)} chunks")
    return all_triplets


async def _write_triplets(
    storage: Neo4jStorage,
    triplets: list[dict[str, Any]],
    namespace: str,
) -> dict[str, int]:
    """Write LLM-extracted triplets to Neo4j via Neo4jStorage."""
    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    for t in triplets:
        subj = t.get("subject", {})
        obj = t.get("object", {})
        predicate = t.get("predicate", "").strip().upper().replace(" ", "_")

        subj_name = subj.get("name", "").strip()
        obj_name = obj.get("name", "").strip()
        if not subj_name or not obj_name or not predicate:
            continue

        subj_label = subj.get("label", "Entity")
        obj_label = obj.get("label", "Entity")

        # Deduplicate nodes
        subj_key = f"{subj_label}:{subj_name}"
        obj_key = f"{obj_label}:{obj_name}"

        if subj_key not in seen_nodes:
            seen_nodes.add(subj_key)
            nodes.append(
                {
                    "labels": [subj_label],
                    "properties": {"name": subj_name},
                    "merge_on": ["name"],
                }
            )
        if obj_key not in seen_nodes:
            seen_nodes.add(obj_key)
            nodes.append(
                {
                    "labels": [obj_label],
                    "properties": {"name": obj_name},
                    "merge_on": ["name"],
                }
            )

        relationships.append(
            {
                "source_labels": [subj_label],
                "source_match": {"name": subj_name},
                "target_labels": [obj_label],
                "target_match": {"name": obj_name},
                "rel_type": predicate,
            }
        )

    result = await storage.bulk_write(nodes, relationships, namespace)
    meta = result.metadata or {}
    return {"nodes": meta.get("nodes_written", 0), "rels": meta.get("rels_written", 0)}


async def _write_content_clusters(
    storage: Neo4jStorage,
    clusters: list[dict[str, Any]],
    namespace: str,
) -> dict[str, int]:
    """Write Topic nodes from content clusters and link to key entities.

    For each cluster:
    - Creates a Topic node with name + summary
    - Creates BELONGS_TO relationships from key_entities (Person nodes) to Topic
    - Creates RELATED_TOPIC relationships between co-occurring topics
    """
    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    for cluster in clusters:
        name = cluster.get("name", "").strip()
        if not name:
            continue

        summary = cluster.get("summary", "")
        labels = cluster.get("suggested_labels", [])
        key_entities = cluster.get("key_entities", [])

        # Topic node
        nodes.append(
            {
                "labels": ["Topic"],
                "properties": {
                    "name": name,
                    "summary": summary,
                    "suggested_labels": ", ".join(labels) if isinstance(labels, list) else str(labels),
                },
                "merge_on": ["name"],
            }
        )

        # BELONGS_TO from key entities to Topic
        for entity_name in key_entities:
            entity_name = str(entity_name).strip()
            if not entity_name:
                continue
            # Ensure the entity Person node exists
            nodes.append(
                {
                    "labels": ["Person"],
                    "properties": {"name": entity_name},
                    "merge_on": ["name"],
                }
            )
            relationships.append(
                {
                    "source_labels": ["Person"],
                    "source_match": {"name": entity_name},
                    "target_labels": ["Topic"],
                    "target_match": {"name": name},
                    "rel_type": "BELONGS_TO",
                }
            )

    # RELATED_TOPIC between co-occurring topics (share entities)
    entity_to_topics: dict[str, list[str]] = {}
    for cluster in clusters:
        for entity in cluster.get("key_entities", []):
            entity = str(entity).strip()
            entity_to_topics.setdefault(entity, []).append(cluster.get("name", ""))

    seen_pairs: set[frozenset[str]] = set()
    for topics in entity_to_topics.values():
        for i, t1 in enumerate(topics):
            for t2 in topics[i + 1 :]:
                pair = frozenset({t1, t2})
                if pair not in seen_pairs and t1 != t2:
                    seen_pairs.add(pair)
                    relationships.append(
                        {
                            "source_labels": ["Topic"],
                            "source_match": {"name": t1},
                            "target_labels": ["Topic"],
                            "target_match": {"name": t2},
                            "rel_type": "RELATED_TOPIC",
                        }
                    )

    if not nodes and not relationships:
        return {"nodes": 0, "rels": 0}

    result = await storage.bulk_write(nodes, relationships, namespace)
    meta = result.metadata or {}
    return {"nodes": meta.get("nodes_written", 0), "rels": meta.get("rels_written", 0)}
