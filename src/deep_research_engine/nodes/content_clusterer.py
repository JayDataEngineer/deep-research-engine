"""Content Clusterer Node - Groups items into semantic clusters.

Runs between critic and neo4j_builder. Uses LLM to group worker output
into thematic clusters. Optionally queries Neo4j for existing Topic
nodes to promote reuse.

Stores results in ``state.context.content_clusters``.
"""

import asyncio
import json
from typing import Any

from ..config import get_phase_config
from ..prompts import get_prompt
from ..prompts.registry import PromptRegistry
from ..shared.config import DEFAULT_MODEL
from ..shared.multimodal import _extract_chat_text, _get_infra
from ..state import EngineState
from ..storage.backends.neo4j import Neo4jStorage
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context

logger = get_loguru_logger("nodes.content_clusterer")


@node_context("content_clusterer")
async def content_clusterer_node(state: EngineState) -> dict[str, Any]:
    """Group drafter output into semantic clusters via LLM.

    Reads ``worker_results`` and optionally ``context.big_text_dump``,
    sends the text to an LLM with a clustering prompt, and stores the
    resulting clusters in ``state.context.content_clusters``.
    """
    logger.info("Starting Content Clusterer node")

    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "content_clusterer")

    if not phase_config.get("enabled", True):
        logger.info("Content Clusterer disabled. Skipping.")
        return {"__goto__": "neo4j_builder"}

    # 1. Collect text from multiple sources (unified input)
    # Sources: Kreuzberg document chunks, worker output, big_text_dump
    text_chunks: list[str] = []

    # Source 1: Kreuzberg document chunks (from ingestion node)
    doc_chunks = state.get("context", {}).get("document_chunks", [])
    if doc_chunks:
        for chunk in doc_chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", chunk.get("content", ""))
                if text:
                    text_chunks.append(text)
        logger.info(f"Collected {len(doc_chunks)} document chunks from Kreuzberg")

    # Source 2: Worker results (audio/video transcripts, web research)
    worker_text = [
        result["output"]
        for result in state.get("worker_results", [])
        if result.get("output") and isinstance(result["output"], str)
    ]
    text_chunks.extend(worker_text)

    if not text_chunks:
        logger.info("No text to cluster. Skipping.")
        return {"__goto__": "neo4j_builder"}

    combined = "\n\n---\n\n".join(text_chunks)[:80000]

    # 2. Query existing Topic nodes from Neo4j (optional)
    existing_topics_text = ""
    namespace = config_data.get("storage", {}).get("neo4j", {}).get("namespace", "default")
    neo4j_storage = await _try_init_neo4j(config_data)

    if neo4j_storage:
        try:
            rows = await neo4j_storage.execute_cypher(
                "MATCH (t:Topic {namespace: $namespace}) "
                "RETURN t.name as name, t.summary as summary "
                "LIMIT 50",
                {"namespace": namespace},
            )
            if rows:
                lines = [f"- {r['name']}" + (f": {r['summary']}" if r.get("summary") else "")
                         for r in rows]
                existing_topics_text = (
                    "EXISTING TOPICS (reuse these when the content fits):\n"
                    + "\n".join(lines)
                )
                logger.info(f"Found {len(rows)} existing topics in Neo4j")
        except Exception as e:
            logger.warning(f"Failed to query existing topics: {e}")
        finally:
            await neo4j_storage.cleanup()

    # 3. Build prompt
    try:
        prompt_wrapper = await get_prompt(PromptRegistry.CONTENT_CLUSTER)
        prompt = prompt_wrapper.compile(
            topics_section=existing_topics_text,
            combined=combined,
        )
        model_config = prompt_wrapper.get_model_config()
        model_name = phase_config.get("model", model_config.get("model", DEFAULT_MODEL))
        temperature = phase_config.get("temperature", model_config.get("temperature", 0.3))
    except Exception as e:
        logger.error(f"Failed to load clustering prompt: {e}")
        return {"__goto__": "neo4j_builder"}

    # 4. Call LLM
    try:
        infra = _get_infra()

        response = await asyncio.to_thread(
            infra.chat,
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=temperature,
            max_tokens=16000,
        )
        raw = _extract_chat_text(response)

        # Parse JSON from response
        clusters = _parse_clusters(raw)

        if not clusters:
            logger.warning("LLM returned no clusters. Skipping.")
            return {"__goto__": "neo4j_builder"}

        logger.info(f"Content Clusterer produced {len(clusters)} clusters")

        return {
            "context": {
                **state.get("context", {}),
                "content_clusters": clusters,
            },
            "__goto__": "neo4j_builder",
        }

    except Exception as e:
        logger.error(f"Content clustering failed: {e}", exc_info=True)
        return {"__goto__": "neo4j_builder"}


def _parse_clusters(raw: str) -> list[dict[str, Any]]:
    """Extract cluster list from LLM response text."""
    # Try to extract JSON block
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "clusters" in parsed:
            return parsed["clusters"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        # Try to repair truncated JSON by closing open brackets
        for repair in [']}', '}]', '"}]}']:
            try:
                parsed = json.loads(text.rstrip(', "') + repair)
                if isinstance(parsed, dict) and "clusters" in parsed:
                    return parsed["clusters"]
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

    logger.warning(f"Could not parse clusters from LLM output: {text[:200]}")
    return []


async def _try_init_neo4j(config_data: dict[str, Any]) -> Neo4jStorage | None:
    """Try to create and initialize Neo4jStorage from config."""
    from ..tools.storage import get_neo4j_from_config

    return await get_neo4j_from_config(config_data)
