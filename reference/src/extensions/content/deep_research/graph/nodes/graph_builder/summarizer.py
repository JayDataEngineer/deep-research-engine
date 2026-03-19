"""Community Summarizer - Generates LLM summaries for communities.

LLM analyzes the entities in each community and generates:
- A concise title (4 words max)
- A summary of what connects the entities

Then embeds the summaries for similarity search.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import re
import asyncio

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger
from .community_detector import CommunitySummary

logger = get_logger(__name__)


# =============================================================================
# SUMMARIZATION PROMPTS
# =============================================================================

COMMUNITY_SUMMARY_PROMPT = """Analyze these related concepts and entities from research:

{entity_list}

Generate a title and summary for what connects these entities.

Requirements:
1. Title: A concise title (max 4 words) describing the common theme
2. Summary: 2-3 sentences explaining what connects these concepts

Output format (exactly):
title: [your title here]
summary: [your summary here]"""

PARENT_COMMUNITY_PROMPT = """Analyze these community summaries that are related:

{community_summaries}

Generate a higher-level title and summary.

Requirements:
1. Title: A concise title (max 4 words) describing the overarching theme
2. Summary: 2-3 sentences synthesizing the key points

Output format (exactly):
title: [your title here]
summary: [your summary here]"""


class CommunitySummarizer:
    """Generates LLM summaries for each community.

    For each community:
    1. Get the entities in the community
    2. Format them for the LLM
    3. Generate title and summary
    4. Store in Neo4j
    """

    def __init__(
        self,
        neo4j_driver: Any = None,
        llm: Any = None,
        embeddings: Any = None,
        context: Any = None,
    ):
        """Initialize the summarizer.

        Args:
            neo4j_driver: Neo4j driver instance
            llm: Language model for summarization
            embeddings: Embeddings model for community embeddings
            context: ExtensionContext (optional)
        """
        self.driver = neo4j_driver
        self.llm = llm
        self.embeddings = embeddings
        self.context = context
        self._gds = None

    async def _get_gds(self):
        """Get GraphDataScience client."""
        if self._gds is not None:
            return self._gds

        try:
            from graphdatascience import GraphDataScience

            # Get connection details from driver if available
            if self.driver:
                # Extract from driver
                uri = getattr(self.driver, '_uri', 'bolt://localhost:7687')
                auth = getattr(self.driver, '_auth', ('neo4j', 'neo4j'))
            else:
                uri = 'bolt://localhost:7687'
                auth = ('neo4j', 'neo4j')

            self._gds = GraphDataScience(endpoint=uri, auth=auth)
            return self._gds

        except ImportError:
            logger.warning("[GRAPH_BUILDER] graphdatascience not available")
            return None

    async def summarize_communities(
        self,
        session_id: str,
        communities: List[CommunitySummary],
    ) -> List[CommunitySummary]:
        """Generate LLM summaries for each community.

        Args:
            session_id: Research session ID
            communities: List of CommunitySummary objects (without summaries)

        Returns:
            List of CommunitySummary objects with titles and summaries
        """
        if not communities:
            return []

        if not self.llm:
            logger.warning("[GRAPH_BUILDER] No LLM available for community summarization")
            return communities

        gds = await self._get_gds()
        updated_communities = []

        for community in communities:
            try:
                # Get entities for this community
                entities = await self._get_community_entities(session_id, community.community_id)

                if not entities:
                    continue

                # Format entity list
                entity_list = "\n".join([
                    f"- {e['type']}: {e['id']}" + (f" - {e['description'][:100]}" if e.get('description') else "")
                    for e in entities[:20]  # Limit to prevent token overflow
                ])

                # Generate summary with LLM
                prompt = COMMUNITY_SUMMARY_PROMPT.format(entity_list=entity_list)

                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                else:
                    content = str(self.llm(prompt))

                # Parse response
                title, summary = self._parse_summary_response(content)

                # Update community
                community.title = title
                community.summary = summary

                # Store in Neo4j
                await self._store_community_summary(session_id, community.community_id, title, summary)

                updated_communities.append(community)

                logger.debug(f"[GRAPH_BUILDER] Summarized community {community.community_id}: {title}")

            except Exception as e:
                logger.warning(f"[GRAPH_BUILDER] Failed to summarize community {community.community_id}: {e}")
                updated_communities.append(community)

        logger.info(f"[GRAPH_BUILDER] Summarized {len(updated_communities)} communities")
        return updated_communities

    async def summarize_parent_communities(
        self,
        session_id: str,
        communities: List[CommunitySummary],
    ) -> List[CommunitySummary]:
        """Generate summaries for parent communities (level > 0).

        Parent communities are summarized based on their child community summaries.
        """
        if not self.llm:
            return communities

        gds = await self._get_gds()
        if not gds:
            return communities

        # Get parent communities (level > 0)
        parent_communities = [c for c in communities if c.level > 0]

        for parent in parent_communities:
            try:
                # Get child community summaries
                child_summaries = await self._get_child_summaries(session_id, parent.community_id)

                if not child_summaries:
                    continue

                # Format for LLM
                summaries_text = "\n\n".join([
                    f"Community: {s['title']}\n{s['summary']}"
                    for s in child_summaries
                ])

                # Generate parent summary
                prompt = PARENT_COMMUNITY_PROMPT.format(community_summaries=summaries_text)

                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                else:
                    content = str(self.llm(prompt))

                # Parse response
                title, summary = self._parse_summary_response(content)

                # Update and store
                parent.title = title
                parent.summary = summary

                await self._store_community_summary(session_id, parent.community_id, title, summary)

            except Exception as e:
                logger.warning(f"[GRAPH_BUILDER] Failed to summarize parent community {parent.community_id}: {e}")

        return communities

    def _parse_summary_response(self, content: str) -> tuple[str, str]:
        """Parse title and summary from LLM response."""
        title = "Untitled"
        summary = ""

        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()

        # Fallback: try to extract from freeform text
        if not title or title == "Untitled":
            # Take first sentence as title
            sentences = re.split(r'[.!?]', content)
            if sentences:
                title = sentences[0].strip()[:50]

        if not summary:
            # Take rest of content as summary
            summary_match = re.search(r'summary:(.+)', content, re.IGNORECASE | re.DOTALL)
            if summary_match:
                summary = summary_match.group(1).strip()
            else:
                # Take everything after first line
                summary = " ".join(lines[1:]).strip()

        return title, summary

    async def _get_community_entities(self, session_id: str, community_id: str) -> List[Dict]:
        """Get entities for a community."""
        gds = await self._get_gds()
        if not gds:
            return []

        result = gds.run_cypher("""
            MATCH (c:Community {id: $community_id, session_id: $session_id})<-[:IN_COMMUNITY]-(e:Entity)
            RETURN e.id as id, e.type as type, e.description as description
        """, {"session_id": session_id, "community_id": community_id})

        return result.to_dict('records')

    async def _get_child_summaries(self, session_id: str, parent_id: str) -> List[Dict]:
        """Get summaries of child communities."""
        gds = await self._get_gds()
        if not gds:
            return []

        result = gds.run_cypher("""
            MATCH (child:Community)-[:PARENT_COMMUNITY]->(parent:Community {id: $parent_id, session_id: $session_id})
            WHERE child.summary IS NOT NULL
            RETURN child.title as title, child.summary as summary
        """, {"session_id": session_id, "parent_id": parent_id})

        return result.to_dict('records')

    async def _store_community_summary(
        self,
        session_id: str,
        community_id: str,
        title: str,
        summary: str,
    ):
        """Store community summary in Neo4j."""
        gds = await self._get_gds()
        if not gds:
            return

        gds.run_cypher("""
            MATCH (c:Community {id: $community_id, session_id: $session_id})
            SET c.title = $title, c.summary = $summary
        """, {
            "session_id": session_id,
            "community_id": community_id,
            "title": title,
            "summary": summary,
        })

    async def embed_communities(
        self,
        session_id: str,
        communities: List[CommunitySummary],
    ) -> int:
        """Create embeddings for community summaries.

        Args:
            session_id: Research session ID
            communities: List of CommunitySummary objects with summaries

        Returns:
            Number of communities embedded
        """
        if not self.embeddings:
            logger.warning("[GRAPH_BUILDER] No embeddings model available")
            return 0

        gds = await self._get_gds()
        if not gds:
            return 0

        embedded_count = 0

        for community in communities:
            if not community.summary:
                continue

            try:
                # Create text to embed
                text = f"{community.title}: {community.summary}"

                # Generate embedding
                if hasattr(self.embeddings, 'aembed_query'):
                    embedding = await self.embeddings.aembed_query(text)
                else:
                    embedding = self.embeddings.embed_query(text)

                # Store in Neo4j
                gds.run_cypher("""
                    MATCH (c:Community {id: $community_id, session_id: $session_id})
                    CALL db.create.setNodeVectorProperty(c, "embedding", $embedding)
                """, {
                    "session_id": session_id,
                    "community_id": community.community_id,
                    "embedding": embedding,
                })

                embedded_count += 1

            except Exception as e:
                logger.warning(f"[GRAPH_BUILDER] Failed to embed community {community.community_id}: {e}")

        logger.info(f"[GRAPH_BUILDER] Embedded {embedded_count} community summaries")
        return embedded_count
