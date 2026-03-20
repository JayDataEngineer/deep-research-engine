import os
import json
from typing import Any, List, Dict
from neo4j import GraphDatabase
import httpx

from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config

logger = get_loguru_logger("nodes.neo4j_builder")


@node_context("neo4j_builder")
async def neo4j_builder_node(state: EngineState) -> dict[str, Any]:
    """Neo4j Builder node - Extracts triplets and writes to Neo4j."""
    logger.info("Starting Neo4j Builder node")

    # 1. Get configuration
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "neo4j_builder")
    
    neo4j_enabled = phase_config.get("enabled", False)
    if not neo4j_enabled:
        # Fallback to storage config
        neo4j_enabled = config_data.get("storage", {}).get("neo4j", {}).get("enabled", False)

    if not neo4j_enabled:
        logger.info("Skipping Neo4j Builder (not enabled).")
        return {"__goto__": "synthesizer"}

    synthesis_output = state.get("worker_results", [{}])[-1].get("output", "")
    if not synthesis_output:
        logger.warning("No synthesis output found to extract relationships from.")
        return {"__goto__": "synthesizer"}

    try:
        # 1. Extract Triplets using LLM
        logger.info("Extracting triplets from synthesis output...")
        triplets = await _extract_triplets(synthesis_output)
        
        # 2. Write to Neo4j
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")
        
        if triplets:
            logger.info(f"Writing {len(triplets)} triplets to Neo4j at {uri}...")
            _write_to_neo4j(uri, user, password, triplets)
            return {
                "neo4j_result": f"Successfully wrote {len(triplets)} relationships to Neo4j.",
                "__goto__": "synthesizer"
            }
        else:
            logger.info("No triplets extracted.")
            return {
                "neo4j_result": "No relationships found to write.",
                "__goto__": "synthesizer"
            }

    except Exception as e:
        logger.error(f"Neo4j Builder error: {e}", exc_info=True)
        # We don't mark fatal_error because this is an optional enrichment phase
        return {
            "neo4j_error": str(e),
            "__goto__": "synthesizer"
        }

async def _extract_triplets(text: str) -> List[Dict[str, str]]:
    """Extract (Subject, Predicate, Object) triplets from text."""
    base_url = os.environ.get("LITELLM_API_BASE", "http://localhost:4000")
    api_key = os.environ.get("LITELLM_API_KEY", "fake-key")
    
    prompt = (
        "Extract key entities and their relationships from the following text as a JSON list of triplets.\n"
        "Each triplet should have 'subject', 'predicate', and 'object'.\n\n"
        f"Text:\n{text[:4000]}\n\n"
        "Respond ONLY with a JSON list."
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0
            )
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            # Some models wrap in 'triplets' key
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "triplets" in parsed:
                return parsed["triplets"]
            return parsed if isinstance(parsed, list) else []
    except Exception as e:
        logger.error(f"Triplet extraction failed: {e}")
        return []

def _write_to_neo4j(uri, user, password, triplets):
    """Write triplets to Neo4j using Cypher MERGE."""
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            for t in triplets:
                subject = t.get("subject", "").strip()
                predicate = t.get("predicate", "").strip().upper().replace(" ", "_")
                obj = t.get("object", "").strip()
                
                if not subject or not predicate or not obj:
                    continue
                
                # Simple generic Cypher to create nodes and a relationship
                cypher = (
                    f"MERGE (s:Entity {{name: $subject}}) "
                    f"MERGE (o:Entity {{name: $object}}) "
                    f"MERGE (s)-[:{predicate}]->(o)"
                )
                session.run(cypher, subject=subject, object=obj)

