"""Community Detector - Detects concept communities using Leiden algorithm.

Uses Neo4j GraphDataScience to detect communities of related entities.
Creates hierarchical community structure with IN_COMMUNITY relationships.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger

logger = get_logger(__name__)


class CommunitySummary(BaseModel):
    """Summary of a detected community."""
    community_id: str = Field(..., description="Community ID")
    level: int = Field(default=0, description="Hierarchy level (0=smallest)")
    title: str = Field(default="", description="LLM-generated title")
    summary: str = Field(default="", description="LLM-generated summary")
    entity_count: int = Field(default=0, description="Number of entities")
    key_entities: List[str] = Field(default_factory=list, description="Key entity IDs")


class CommunityDetector:
    """Detects concept communities using Leiden algorithm.

    The Leiden algorithm finds clusters of densely connected nodes.
    For concept-focused graphs, this groups related concepts together.

    Creates a hierarchical structure:
    - Level 0: Small communities (5-10 related entities)
    - Level 1: Medium communities (groups of level 0)
    - Level 2: Large communities (groups of level 1)
    """

    def __init__(
        self,
        neo4j_driver: Any = None,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "neo4j",
        context: Any = None,
    ):
        """Initialize the community detector.

        Args:
            neo4j_driver: Existing Neo4j driver (optional)
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            context: ExtensionContext (optional)
        """
        self.driver = neo4j_driver
        self.uri = uri
        self.username = username
        self.password = password
        self.context = context
        self._gds = None

    async def _get_gds(self):
        """Get or create GraphDataScience client."""
        if self._gds is not None:
            return self._gds

        try:
            from graphdatascience import GraphDataScience

            self._gds = GraphDataScience(
                endpoint=self.uri,
                auth=(self.username, self.password),
            )
            return self._gds

        except ImportError:
            raise RuntimeError(
                "graphdatascience package not installed. Install with: pip install graphdatascience\n"
                "Also ensure Neo4j GDS library is installed on the Neo4j server."
            )

    async def detect_communities(
        self,
        session_id: str,
        max_levels: int = 3,
        min_community_size: int = 2,
    ) -> List[CommunitySummary]:
        """Detect communities using Leiden algorithm.

        Args:
            session_id: Research session ID
            max_levels: Maximum hierarchy depth
            min_community_size: Minimum entities per community

        Returns:
            List of CommunitySummary objects
        """
        gds = await self._get_gds()
        projection_name = f"communities_{session_id.replace('-', '_')}"

        logger.info(f"[GRAPH_BUILDER] Starting community detection for session {session_id}")

        try:
            # 1. Clear old communities
            await self._clear_communities(session_id)

            # 2. Project the entity graph
            await self._project_graph(gds, session_id, projection_name)

            # 3. Run Leiden algorithm
            await self._run_leiden(gds, projection_name, max_levels, min_community_size)

            # 4. Create Community nodes
            await self._create_community_nodes(session_id)

            # 5. Get community summaries (without entities - that's done in summarizer)
            summaries = await self._get_community_summaries(session_id)

            logger.info(f"[GRAPH_BUILDER] Detected {len(summaries)} communities")

            return summaries

        except Exception as e:
            logger.error(f"[GRAPH_BUILDER] Community detection failed: {e}")
            raise

        finally:
            # Cleanup projection
            try:
                gds.graph.drop(projection_name, failIfMissing=False)
            except:
                pass

    async def _clear_communities(self, session_id: str):
        """Clear existing communities for this session."""
        gds = await self._get_gds()

        # Delete community nodes
        gds.run_cypher("""
            MATCH (c:Community {session_id: $session_id})
            DETACH DELETE c
        """, {"session_id": session_id})

        # Remove communities property from entities
        gds.run_cypher("""
            MATCH (e:Entity)
            WHERE $session_id IN e.session_ids
            SET e.communities = null
        """, {"session_id": session_id})

        logger.debug(f"[GRAPH_BUILDER] Cleared existing communities for session {session_id}")

    async def _project_graph(self, gds: Any, session_id: str, projection_name: str):
        """Project the entity graph for community detection.

        Projects only entities and their relationships (not chunks).
        """
        # Check if projection already exists
        try:
            existing = gds.graph.list()
            if projection_name in existing['graphName'].values:
                gds.graph.drop(projection_name)
        except:
            pass

        # Project using native projection
        projection_query = """
            MATCH (e:Entity)-[r]->(e2:Entity)
            WHERE $session_id IN e.session_ids
            AND $session_id IN e2.session_ids
            WITH e, e2, count(r) as weight
            WITH gds.graph.project(
                $name,
                e,
                e2,
                { relationshipProperties: { weight: weight } },
                { undirectedRelationshipTypes: ['*'] }
            ) AS projection
            RETURN projection.graphName AS name, projection.nodeCount AS nodes, projection.relationshipCount AS rels
        """

        result = gds.run_cypher(projection_query, {
            "session_id": session_id,
            "name": projection_name,
        })

        if len(result) > 0:
            row = result.iloc[0]
            logger.info(f"[GRAPH_BUILDER] Projected graph: {row['nodes']} nodes, {row['rels']} relationships")
        else:
            logger.warning("[GRAPH_BUILDER] No relationships found for community detection")

    async def _run_leiden(
        self,
        gds: Any,
        projection_name: str,
        max_levels: int,
        min_community_size: int,
    ):
        """Run Leiden algorithm on the projected graph."""
        try:
            graph = gds.graph.get(projection_name)

            # Run Leiden with intermediate communities for hierarchy
            result = gds.leiden.write(
                graph,
                writeProperty="communities",
                includeIntermediateCommunities=True,
                maxLevels=max_levels,
                minCommunitySize=min_community_size,
            )

            logger.info(f"[GRAPH_BUILDER] Leiden completed: {result}")

        except Exception as e:
            logger.error(f"[GRAPH_BUILDER] Leiden algorithm failed: {e}")
            raise

    async def _create_community_nodes(self, session_id: str):
        """Create Community nodes with IN_COMMUNITY relationships."""
        gds = await self._get_gds()

        # Create Level 0 communities
        gds.run_cypher("""
            MATCH (e:Entity)
            WHERE e.communities IS NOT NULL
            AND $session_id IN e.session_ids
            WITH e, e.communities[0] AS cid
            MERGE (c:Community {id: '0-' + toString(cid), session_id: $session_id})
            ON CREATE SET c.level = 0
            MERGE (e)-[:IN_COMMUNITY]->(c)
        """, {"session_id": session_id})

        # Create higher-level communities with PARENT_COMMUNITY relationships
        gds.run_cypher("""
            MATCH (e:Entity)
            WHERE e.communities IS NOT NULL
            AND size(e.communities) > 1
            AND $session_id IN e.session_ids
            UNWIND range(1, size(e.communities) - 1) AS level
            MERGE (curr:Community {
                id: toString(level) + '-' + toString(e.communities[level]),
                session_id: $session_id
            })
            ON CREATE SET curr.level = level
            MERGE (prev:Community {
                id: toString(level-1) + '-' + toString(e.communities[level-1]),
                session_id: $session_id
            })
            ON CREATE SET prev.level = level - 1
            MERGE (prev)-[:PARENT_COMMUNITY]->(curr)
        """, {"session_id": session_id})

        logger.debug(f"[GRAPH_BUILDER] Created Community nodes")

    async def _get_community_summaries(self, session_id: str) -> List[CommunitySummary]:
        """Get community summaries (without LLM - that's in summarizer)."""
        gds = await self._get_gds()

        result = gds.run_cypher("""
            MATCH (c:Community {session_id: $session_id})<-[:IN_COMMUNITY]-(e:Entity)
            WITH c, collect(e.id) as entities, count(e) as entity_count
            RETURN c.id as community_id, c.level as level, entity_count, entities
            ORDER BY level, entity_count DESC
        """, {"session_id": session_id})

        summaries = []
        for _, row in result.iterrows():
            summaries.append(CommunitySummary(
                community_id=row['community_id'],
                level=row['level'],
                entity_count=row['entity_count'],
                key_entities=row['entities'][:10],  # Top 10 entities
            ))

        return summaries

    async def get_community_entities(self, session_id: str, community_id: str) -> List[Dict]:
        """Get all entities in a community."""
        gds = await self._get_gds()

        result = gds.run_cypher("""
            MATCH (c:Community {id: $community_id, session_id: $session_id})<-[:IN_COMMUNITY]-(e:Entity)
            RETURN e.id as id, e.type as type, e.description as description
        """, {"session_id": session_id, "community_id": community_id})

        return result.to_dict('records')
