import os
import json
from typing import Any, List, Dict
from neo4j import GraphDatabase

from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config
from ..tools.storage import initialize_storage_tools, vector_search_datastore
from src.shared.chat_llm import get_chat_llm
from deepagents import create_deep_agent, SubAgent

logger = get_loguru_logger("nodes.neo4j_builder")


@node_context("neo4j_builder")
async def neo4j_builder_node(state: EngineState) -> dict[str, Any]:
    """Neo4j Builder node - Extracts triplets using a Deep Agent and writes to Neo4j."""
    logger.info("Starting Neo4j Builder node (Agentic)")

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
        # 1. Initialize storage tools with current config
        initialize_storage_tools(config_data)

        # 2. Extract Triplets using Deep Agent
        logger.info("Extracting triplets using Neo4j Deep Agent...")
        triplets = await _extract_triplets_agentic(synthesis_output, config_data)
        
        # 3. Write to Neo4j
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
        return {
            "neo4j_error": str(e),
            "__goto__": "synthesizer"
        }

async def _extract_triplets_agentic(text: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract triplets using a Deep Agent with a Reviewer subagent."""
    model_name = config.get("llm", {}).get("models", {}).get("fast", "openai:gpt-4o-mini")
    
    # 1. Define Subagents
    subagents: List[SubAgent] = [
        {
            "name": "triplet_reviewer",
            "description": "Reviews extracted knowledge graph triplets for ontology compliance and entity deduplication.",
            "system_prompt": (
                "You are a Graph Quality Critic. Review the proposed triplets for the knowledge graph.\n"
                "Constraints:\n"
                "1. Entity Resolution: Check if 'subject' or 'object' are duplicates (e.g. 'US' vs 'USA').\n"
                "2. Ontology Check: Only use these Labels: [Person, Company, Technology, Project, Event].\n"
                "3. Relationship Check: Only use these Types: [WORKS_AT, FOUNDED, DEVELOPED, PARTS_OF, ASSOCIATED_WITH, LOCATED_IN].\n"
                "Suggest corrections in a structured format."
            ),
            "model": model_name
        }
    ]

    # 2. System Prompt with strict ontology
    system_prompt = (
        "You are the Neo4j Knowledge Graph Specialist.\n"
        "Your task is to extract (Subject, Predicate, Object) triplets from research text.\n\n"
        "STRICT ONTOLOGY:\n"
        "- Node Labels: [Person, Company, Technology, Project, Event]\n"
        "- Relationship Types: [WORKS_AT, FOUNDED, DEVELOPED, PARTS_OF, ASSOCIATED_WITH, LOCATED_IN]\n\n"
        "CORE PROCESS:\n"
        "1. Identify entities and their relationships in the text.\n"
        "2. Use 'vector_search_datastore' to resolve ambiguous entities against the database.\n"
        "3. Ask the 'triplet_reviewer' to audit your proposed triplets before finishing.\n"
        "4. Respond with a final JSON object containing a 'triplets' key.\n\n"
        "SOURCE TEXT:\n"
        f"{text[:6000]}"
    )

    # 3. Create and Invoke Deep Agent
    agent = create_deep_agent(
        model=model_name,
        system_prompt=system_prompt,
        subagents=subagents,
        tools=[vector_search_datastore],
        debug=True
    )

    logger.info("Invoking Neo4j Specialist Agent...")
    result = await agent.ainvoke({"messages": [("user", "Please extract the knowledge graph triplets from the source text and review them for quality.")]})
    
    final_output = result["messages"][-1].content
    
    # 4. Parse JSON result
    try:
        # Basic cleanup for markdown code blocks
        if "```json" in final_output:
            final_output = final_output.split("```json")[1].split("```")[0].strip()
        elif "```" in final_output:
            final_output = final_output.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(final_output)
        if isinstance(parsed, dict) and "triplets" in parsed:
            return parsed["triplets"]
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        logger.error(f"Failed to parse triplets from agent output: {e}\nOutput: {final_output}")
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
