"""Graph Builder Node - Main orchestrator for knowledge graph construction.

This is a SINGLE ISOLATED NODE that:
1. Pulls findings from LangGraph Store (where deep_agents write via write_file("/memories/"))
2. Extracts entities and concepts using LLMGraphTransformer
3. Consolidates schema (normalizes entity types)
4. Builds Neo4j graph with session isolation
5. Creates entity embeddings
6. Detects communities using Leiden algorithm
7. Generates community summaries and embeddings

All Neo4j creation happens HERE - Flash Explorers should NOT write to Neo4j.
This node runs ONCE before synthesis.
"""

import time
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
import logging

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger
from ...decorators import research_node
from ...constants import Phase
from langchain_core.runnables import RunnableConfig

from ..common import (
    GlobalState,
    ResearchRuntimeConfig,
    get_run_logger_from_config,
    get_yaml_config_from_context,
    create_llm_from_config,
)

from langgraph.types import Command
from langgraph.runtime import Runtime

# Import sub-components
from .chunker import FindingChunker, EntityChunk
from .extractor import ConceptExtractor, GraphDocument
from .consolidator import SchemaConsolidator
from .builder import Neo4jGraphBuilder
from .embedder import EntityEmbedder
from .community_detector import CommunityDetector, CommunitySummary
from .summarizer import CommunitySummarizer

logger = get_logger(__name__)


# =============================================================================
# NODE OUTPUT STATE
# =============================================================================

class GraphBuilderOutput(BaseModel):
    """Output from the graph builder node."""
    graph_built: bool = Field(default=False, description="Whether graph was built successfully")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    entities_extracted: int = Field(default=0, description="Number of entities extracted")
    relationships_created: int = Field(default=0, description="Number of relationships created")
    communities_detected: int = Field(default=0, description="Number of communities detected")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# =============================================================================
# HELPER: Get LangGraph Store
# =============================================================================

async def _get_langgraph_store():
    """Get the LangGraph Store instance.

    Uses the same store that deep_agents use for write_file("/memories/").
    """
    try:
        from src.extensions.features.tools.context_engine.storage.langgraph_store import get_langgraph_store
        return await get_langgraph_store()
    except Exception as e:
        logger.warning(f"[GRAPH_BUILDER] Could not get LangGraph Store: {e}")
        return None


# =============================================================================
# MAIN NODE
# =============================================================================

@research_node(Phase.GRAPH_BUILDER)
async def graph_builder_node(
    state: GlobalState,
    writer: Optional[Any] = None,
    *,
    runtime: Optional[Runtime[ResearchRuntimeConfig]] = None,
    store: Optional[Any] = None,
    config: RunnableConfig | None = None,
) -> Command[Literal["synthesis"]]:
    """Build knowledge graph from research memories.

    This node:
    1. Pulls findings from LangGraph Store (deep_agent memories)
    2. Extracts entities/concepts using LLM
    3. Builds Neo4j graph
    4. Detects communities
    5. Creates embeddings

    Deep agents save findings via write_file("/memories/") tool.
    Graph Builder reads from the same LangGraph Store.
    """
    start_time = time.time()

    yaml_config = get_yaml_config_from_context()
    run_logger = get_run_logger_from_config(config)

    session_id = state.get("session_id")

    run_logger._write_to_file(f"[GRAPH_BUILDER] Starting graph build for session {session_id}")

    output = GraphBuilderOutput()
    communities = []

    try:
        # Get LangGraph Store (from parameter or initialize)
        if not store:
            store = await _get_langgraph_store()

        if not store:
            raise RuntimeError(
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║       LANGGRAPH STORE NOT AVAILABLE - CANNOT BUILD GRAPH     ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
                "\nGraph Builder requires LangGraph Store to pull memories.\n"
                "Deep agents must have write_file access to /memories/.\n"
            )

        # Get LLM for extraction and summarization
        llm = await create_llm_from_config(yaml_config, role="worker")

        # Get Neo4j connection details
        neo4j_config = yaml_config.get("context_engine", {}).get("neo4j", {})
        neo4j_uri = neo4j_config.get("uri", "bolt://localhost:7687")
        neo4j_user = neo4j_config.get("username", "neo4j")
        neo4j_password = neo4j_config.get("password", "neo4j")

        # Get embeddings model
        embeddings = None
        try:
            from src.shared.llm.factory import get_embeddings
            embeddings = get_embeddings()
        except:
            run_logger._write_to_file("[GRAPH_BUILDER] Warning: Could not get embeddings model")

        # =========================================================================
        # STEP 1: Get findings as chunks from LangGraph Store
        # =========================================================================
        run_logger._write_to_file("[GRAPH_BUILDER] Step 1: Getting memories from LangGraph Store")

        chunker = FindingChunker(context=None)
        chunks = await chunker.get_findings_as_chunks(store, session_id)

        output.chunks_processed = len(chunks)
        run_logger._write_to_file(f"[GRAPH_BUILDER] Found {len(chunks)} chunks for processing")

        if not chunks:
            run_logger._write_to_file("[GRAPH_BUILDER] No chunks found, skipping graph build")
            output.graph_built = True  # Not an error, just nothing to process

            elapsed = time.time() - start_time
            return Command(
                update={
                    "graph_builder_output": output.model_dump(),
                    "graph_built": True,
                    "current_phase": "synthesis",
                },
                goto="synthesis",
            )

        # =========================================================================
        # STEP 2: Extract entities/concepts
        # =========================================================================
        run_logger._write_to_file("[GRAPH_BUILDER] Step 2: Extracting entities and concepts")

        extractor = ConceptExtractor(llm=llm)
        graph_documents = await extractor.extract(chunks)

        entities_count = sum(len(doc.entities) for doc in graph_documents)
        output.entities_extracted = entities_count
        run_logger._write_to_file(f"[GRAPH_BUILDER] Extracted {entities_count} entities")

        # =========================================================================
        # STEP 3: Consolidate schema
        # =========================================================================
        run_logger._write_to_file("[GRAPH_BUILDER] Step 3: Consolidating schema")

        consolidator = SchemaConsolidator(llm=llm)
        graph_documents = await consolidator.consolidate(graph_documents)

        # =========================================================================
        # STEP 4: Build Neo4j graph
        # =========================================================================
        run_logger._write_to_file("[GRAPH_BUILDER] Step 4: Building Neo4j graph")

        builder = Neo4jGraphBuilder(
            uri=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
        )

        counts = await builder.build_graph(graph_documents, session_id)
        output.relationships_created = counts.get("relationships", 0)

        run_logger._write_to_file(f"[GRAPH_BUILDER] Built graph: {counts}")

        # =========================================================================
        # STEP 5: Create entity embeddings
        # =========================================================================
        if embeddings:
            run_logger._write_to_file("[GRAPH_BUILDER] Step 5: Creating entity embeddings")

            embedder = EntityEmbedder(
                neo4j_driver=builder.driver,
                embeddings=embeddings,
            )

            entity_embedded = await embedder.embed_entities(session_id)
            run_logger._write_to_file(f"[GRAPH_BUILDER] Embedded {entity_embedded} entities")

        # =========================================================================
        # STEP 6: Detect communities
        # =========================================================================
        run_logger._write_to_file("[GRAPH_BUILDER] Step 6: Detecting communities")

        community_detector = CommunityDetector(
            neo4j_driver=builder.driver,
            uri=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
        )

        communities = await community_detector.detect_communities(session_id)
        output.communities_detected = len(communities)

        run_logger._write_to_file(f"[GRAPH_BUILDER] Detected {len(communities)} communities")

        # =========================================================================
        # STEP 7: Generate community summaries
        # =========================================================================
        if communities and llm:
            run_logger._write_to_file("[GRAPH_BUILDER] Step 7: Generating community summaries")

            summarizer = CommunitySummarizer(
                neo4j_driver=builder.driver,
                llm=llm,
                embeddings=embeddings,
            )

            communities = await summarizer.summarize_communities(session_id, communities)

            # Summarize parent communities (level > 0)
            communities = await summarizer.summarize_parent_communities(session_id, communities)

            # Embed community summaries
            if embeddings:
                await summarizer.embed_communities(session_id, communities)

            run_logger._write_to_file(f"[GRAPH_BUILDER] Summarized {len(communities)} communities")

        # =========================================================================
        # STEP 8: Create vector indexes
        # =========================================================================
        run_logger._write_to_file("[GRAPH_BUILDER] Step 8: Creating vector indexes")

        await builder.create_vector_indexes(session_id)

        # =========================================================================
        # Cleanup and return
        # =========================================================================
        await builder.close()

        output.graph_built = True
        elapsed = time.time() - start_time

        run_logger._write_to_file(f"[GRAPH_BUILDER] Graph build completed in {elapsed:.2f}s")
        run_logger._write_to_file(f"[GRAPH_BUILDER] Stats: {output.model_dump()}")

        return Command(
            update={
                "graph_builder_output": output.model_dump(),
                "graph_built": True,
                "community_summaries": [c.model_dump() for c in communities],
                "current_phase": "synthesis",
            },
            goto="synthesis",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)

        run_logger._write_to_file(f"[GRAPH_BUILDER] FAILED after {elapsed:.2f}s: {error_msg}")
        logger.error(f"[GRAPH_BUILDER] Graph build failed: {e}", exc_info=True)

        output.error = error_msg

        # Continue to synthesis even on failure - synthesis can work without graph
        return Command(
            update={
                "graph_builder_output": output.model_dump(),
                "graph_built": False,
                "graph_error": error_msg,
                "current_phase": "synthesis",
            },
            goto="synthesis",
        )


# =============================================================================
# ALTERNATIVE: Class-based node for more control
# =============================================================================

class GraphBuilderNode:
    """Class-based Graph Builder Node for more control over initialization.

    Use this when you need to:
    - Reuse components across multiple calls
    - Have custom configuration
    - Test components independently
    """

    def __init__(
        self,
        llm: Any = None,
        embeddings: Any = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j",
    ):
        """Initialize the graph builder node with components."""
        self.llm = llm
        self.embeddings = embeddings
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Initialize components (lazy - only when needed)
        self._chunker = None
        self._extractor = None
        self._consolidator = None
        self._builder = None
        self._embedder = None
        self._community_detector = None
        self._summarizer = None

    async def build(self, store: Any, session_id: str) -> GraphBuilderOutput:
        """Build the knowledge graph.

        Args:
            store: LangGraph BaseStore instance
            session_id: Research session ID

        Returns:
            GraphBuilderOutput with statistics
        """
        output = GraphBuilderOutput()

        try:
            # Initialize components
            self._chunker = FindingChunker()
            self._extractor = ConceptExtractor(llm=self.llm)
            self._consolidator = SchemaConsolidator(llm=self.llm)
            self._builder = Neo4jGraphBuilder(
                uri=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
            )
            self._embedder = EntityEmbedder(
                neo4j_driver=self._builder.driver,
                embeddings=self.embeddings,
            )
            self._community_detector = CommunityDetector(
                neo4j_driver=self._builder.driver,
                uri=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
            )
            self._summarizer = CommunitySummarizer(
                neo4j_driver=self._builder.driver,
                llm=self.llm,
                embeddings=self.embeddings,
            )

            # Execute pipeline
            chunks = await self._chunker.get_findings_as_chunks(store, session_id)
            output.chunks_processed = len(chunks)

            if not chunks:
                output.graph_built = True
                return output

            graph_docs = await self._extractor.extract(chunks)
            output.entities_extracted = sum(len(d.entities) for d in graph_docs)

            graph_docs = await self._consolidator.consolidate(graph_docs)

            counts = await self._builder.build_graph(graph_docs, session_id)
            output.relationships_created = counts.get("relationships", 0)

            if self.embeddings:
                await self._embedder.embed_entities(session_id)

            communities = await self._community_detector.detect_communities(session_id)
            output.communities_detected = len(communities)

            if communities:
                communities = await self._summarizer.summarize_communities(session_id, communities)
                communities = await self._summarizer.summarize_parent_communities(session_id, communities)
                if self.embeddings:
                    await self._summarizer.embed_communities(session_id, communities)

            await self._builder.create_vector_indexes(session_id)
            await self._builder.close()

            output.graph_built = True

        except Exception as e:
            output.error = str(e)
            logger.error(f"[GRAPH_BUILDER] Build failed: {e}")

        return output
