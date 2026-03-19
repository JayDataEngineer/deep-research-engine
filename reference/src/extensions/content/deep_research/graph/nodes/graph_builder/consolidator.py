"""Schema Consolidator - Normalizes entity types using LLM.

LLMs produce inconsistent entity types ("Person", "PERSON", "people", "Human").
This module uses an LLM to map variants to canonical forms.
"""

from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import logging
import json
import re

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger
from .extractor import GraphDocument, CANONICAL_ENTITY_TYPES, CANONICAL_RELATIONSHIPS

logger = get_logger(__name__)


# =============================================================================
# CONSOLIDATION PROMPTS
# =============================================================================

CONSOLIDATION_PROMPT = """You are normalizing entity types and relationships in a knowledge graph.

Given lists of extracted node types and relationship types, map them to canonical forms.

CANONICAL ENTITY TYPES:
- Concept: Ideas, theories, frameworks, paradigms
- Finding: Specific claims, results, discoveries
- Method: Methodologies, approaches, techniques
- Tool: Technologies, software, instruments
- Person: People, researchers, authors
- Organization: Companies, institutions, groups
- Dataset: Data sources, corpora
- Topic: High-level subject areas
- Event: Conferences, milestones, dates
- Location: Physical places

CANONICAL RELATIONSHIPS:
- RELATES_TO: General connection
- SUPPORTS: A provides evidence for B
- CONTRADICTS: A conflicts with B
- DERIVED_FROM: A builds on B
- SUBSUMES: A contains/is broader than B
- APPLIES: A uses B
- ATTRIBUTED_TO: Finding → Person/Org
- FOUND_BY: Finding → Method
- MEASURED_IN: Finding → Dataset
- PUBLISHED_IN: Finding → Event/Location
- WORKS_AT, LOCATED_IN, PART_OF, USED_BY

Input node types: {node_types}
Input relationship types: {relationship_types}

Output a JSON object with this exact structure:
{{
  "nodes": {{
    "CanonicalType": ["variant1", "variant2", "variant3"]
  }},
  "relationships": {{
    "CANONICAL_REL": ["variant1", "variant2"]
  }}
}}

IMPORTANT: Only include variants that differ from the canonical form.
If a type is already canonical, do NOT include it in the mapping.
"""


class SchemaConsolidator:
    """Normalizes entity types and relationships using LLM.

    This cleans up inconsistencies from LLM extraction:
    - "Person", "PERSON", "people", "Human" → "Person"
    - "Organization", "ORG", "company" → "Organization"
    - "RELATES_TO", "related to", "relates_to" → "RELATES_TO"
    """

    def __init__(self, llm: Any = None, context: Any = None):
        """Initialize the consolidator.

        Args:
            llm: Language model to use for consolidation
            context: ExtensionContext (optional)
        """
        self.llm = llm
        self.context = context
        self._node_mappings: Dict[str, str] = {}
        self._rel_mappings: Dict[str, str] = {}

    async def consolidate(
        self,
        graph_documents: List[GraphDocument],
    ) -> List[GraphDocument]:
        """Consolidate schema across all graph documents.

        Args:
            graph_documents: List of GraphDocument objects with extracted entities

        Returns:
            List of GraphDocument objects with normalized types
        """
        if not graph_documents:
            return []

        # Collect all unique types
        node_types: Set[str] = set()
        rel_types: Set[str] = set()

        for doc in graph_documents:
            for entity in doc.entities:
                node_types.add(entity.type)
            for rel in doc.relationships:
                rel_types.add(rel.type)

        logger.info(f"[GRAPH_BUILDER] Found {len(node_types)} node types, {len(rel_types)} relationship types")

        # Check if consolidation is needed
        needs_consolidation = (
            not node_types.issubset(set(CANONICAL_ENTITY_TYPES)) or
            not rel_types.issubset(set(CANONICAL_RELATIONSHIPS))
        )

        if not needs_consolidation:
            logger.info("[GRAPH_BUILDER] All types are canonical, skipping consolidation")
            return graph_documents

        # Get mappings from LLM
        mappings = await self._get_mappings(list(node_types), list(rel_types))

        if mappings:
            self._node_mappings = mappings.get("nodes", {})
            self._rel_mappings = mappings.get("relationships", {})

            # Apply mappings
            graph_documents = self._apply_mappings(graph_documents)

            logger.info(f"[GRAPH_BUILDER] Consolidated to {len(set(e.type for doc in graph_documents for e in doc.entities))} node types")

        return graph_documents

    async def _get_mappings(
        self,
        node_types: List[str],
        rel_types: List[str],
    ) -> Optional[Dict[str, Dict[str, List[str]]]]:
        """Get type mappings from LLM.

        Returns a dict like:
        {
            "nodes": {"Person": ["PERSON", "people", "Human"]},
            "relationships": {"RELATES_TO": ["related to", "relates_to"]}
        }
        """
        # Build reverse mappings for quick lookup
        # First, try rule-based consolidation
        node_mappings = self._rule_based_node_mapping(node_types)
        rel_mappings = self._rule_based_rel_mapping(rel_types)

        # If we have unmapped types, use LLM
        unmapped_nodes = [t for t in node_types if t not in node_mappings and t not in CANONICAL_ENTITY_TYPES]
        unmapped_rels = [t for t in rel_types if t not in rel_mappings and t not in CANONICAL_RELATIONSHIPS]

        if unmapped_nodes or unmapped_rels:
            if self.llm:
                llm_mappings = await self._llm_mapping(unmapped_nodes, unmapped_rels)
                if llm_mappings:
                    node_mappings.update(llm_mappings.get("nodes", {}))
                    rel_mappings.update(llm_mappings.get("relationships", {}))

        # Convert to the expected format
        result = {
            "nodes": {},
            "relationships": {},
        }

        # Group by canonical type
        for variant, canonical in node_mappings.items():
            if canonical not in result["nodes"]:
                result["nodes"][canonical] = []
            result["nodes"][canonical].append(variant)

        for variant, canonical in rel_mappings.items():
            if canonical not in result["relationships"]:
                result["relationships"][canonical] = []
            result["relationships"][canonical].append(variant)

        return result

    def _rule_based_node_mapping(self, node_types: List[str]) -> Dict[str, str]:
        """Apply rule-based mapping for common variants."""
        mappings = {}

        for node_type in node_types:
            normalized = node_type.lower().replace("_", " ").strip()

            # Person variants
            if normalized in ["person", "people", "human", "human being", "individual", "researcher", "author"]:
                mappings[node_type] = "Person"

            # Organization variants
            elif normalized in ["organization", "org", "organisation", "company", "institution", "group", "firm", "enterprise"]:
                mappings[node_type] = "Organization"

            # Concept variants
            elif normalized in ["concept", "idea", "theory", "framework", "paradigm", "notion"]:
                mappings[node_type] = "Concept"

            # Finding variants
            elif normalized in ["finding", "result", "discovery", "conclusion", "claim", "outcome"]:
                mappings[node_type] = "Finding"

            # Method variants
            elif normalized in ["method", "methodology", "approach", "technique", "procedure", "process"]:
                mappings[node_type] = "Method"

            # Tool variants
            elif normalized in ["tool", "technology", "software", "instrument", "application", "platform"]:
                mappings[node_type] = "Tool"

            # Dataset variants
            elif normalized in ["dataset", "data", "corpus", "database", "data source"]:
                mappings[node_type] = "Dataset"

            # Topic variants
            elif normalized in ["topic", "subject", "theme", "area", "field", "domain"]:
                mappings[node_type] = "Topic"

            # Location variants
            elif normalized in ["location", "place", "city", "country", "region", "site"]:
                mappings[node_type] = "Location"

            # Event variants
            elif normalized in ["event", "conference", "meeting", "workshop", "symposium"]:
                mappings[node_type] = "Event"

        return mappings

    def _rule_based_rel_mapping(self, rel_types: List[str]) -> Dict[str, str]:
        """Apply rule-based mapping for relationship variants."""
        mappings = {}

        for rel_type in rel_types:
            normalized = rel_type.upper().replace(" ", "_").replace("-", "_").strip()

            # RELATES_TO variants
            if normalized in ["RELATES_TO", "RELATED_TO", "RELATES", "RELATED", "CONNECTED_TO", "LINKED_TO"]:
                mappings[rel_type] = "RELATES_TO"

            # SUPPORTS variants
            elif normalized in ["SUPPORTS", "SUPPORT", "SUPPORTED_BY", "CONFIRMS", "VALIDATES"]:
                mappings[rel_type] = "SUPPORTS"

            # CONTRADICTS variants
            elif normalized in ["CONTRADICTS", "CONTRADICT", "CONTRADICTED_BY", "CONFLICTS_WITH", "OPPOSES"]:
                mappings[rel_type] = "CONTRADICTS"

            # DERIVED_FROM variants
            elif normalized in ["DERIVED_FROM", "DERIVES_FROM", "BUILDS_ON", "BASED_ON", "EXTENDS"]:
                mappings[rel_type] = "DERIVED_FROM"

            # APPLIES variants
            elif normalized in ["APPLIES", "APPLY", "USES", "UTILIZES", "EMPLOYS"]:
                mappings[rel_type] = "APPLIES"

            # ATTRIBUTED_TO variants
            elif normalized in ["ATTRIBUTED_TO", "ATTRIBUTE_TO", "AUTHORED_BY", "CREATED_BY", "MADE_BY"]:
                mappings[rel_type] = "ATTRIBUTED_TO"

            # WORKS_AT variants
            elif normalized in ["WORKS_AT", "WORKS_FOR", "EMPLOYED_BY", "EMPLOYED_AT", "AFFILIATED_WITH"]:
                mappings[rel_type] = "WORKS_AT"

            # LOCATED_IN variants
            elif normalized in ["LOCATED_IN", "LOCATION", "IN_LOCATION", "SITUATED_IN", "BASED_IN"]:
                mappings[rel_type] = "LOCATED_IN"

        return mappings

    async def _llm_mapping(
        self,
        unmapped_nodes: List[str],
        unmapped_rels: List[str],
    ) -> Optional[Dict[str, Dict[str, str]]]:
        """Use LLM to map remaining unmapped types."""
        if not self.llm:
            return None

        try:
            prompt = CONSOLIDATION_PROMPT.format(
                node_types=unmapped_nodes,
                relationship_types=unmapped_rels,
            )

            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                content = str(self.llm(prompt))

            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                # Convert to simple variant -> canonical mapping
                result = {"nodes": {}, "relationships": {}}

                for canonical, variants in data.get("nodes", {}).items():
                    for variant in variants:
                        result["nodes"][variant] = canonical

                for canonical, variants in data.get("relationships", {}).items():
                    for variant in variants:
                        result["relationships"][variant] = canonical

                return result

        except Exception as e:
            logger.warning(f"[GRAPH_BUILDER] LLM consolidation failed: {e}")

        return None

    def _apply_mappings(self, graph_documents: List[GraphDocument]) -> List[GraphDocument]:
        """Apply type mappings to all graph documents."""
        for doc in graph_documents:
            # Apply node type mappings
            for entity in doc.entities:
                if entity.type in self._node_mappings:
                    original_type = entity.type
                    entity.type = self._node_mappings[entity.type]
                    logger.debug(f"[GRAPH_BUILDER] Mapped node type: {original_type} → {entity.type}")

            # Apply relationship type mappings
            for rel in doc.relationships:
                if rel.type in self._rel_mappings:
                    original_type = rel.type
                    rel.type = self._rel_mappings[rel.type]
                    logger.debug(f"[GRAPH_BUILDER] Mapped rel type: {original_type} → {rel.type}")

        return graph_documents
