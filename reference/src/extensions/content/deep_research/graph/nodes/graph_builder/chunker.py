"""Finding Chunker - Converts research memories into entity-extraction-ready chunks.

Pulls findings from LangGraph Store (where deep_agents write via write_file("/memories/"))
and formats them for the LLMGraphTransformer to extract entities and concepts.
"""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
import logging

# Setup path for imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=8)

from ....logging_utils import get_logger

logger = get_logger(__name__)


class EntityChunk(BaseModel):
    """A chunk of text ready for entity extraction.

    This represents a single unit of text that will be processed
    by the LLMGraphTransformer to extract entities and relationships.
    """

    id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="The text content to extract entities from")
    source_url: Optional[str] = Field(default=None, description="Original source URL")
    source_type: Optional[str] = Field(default=None, description="Type of source (web, pdf, api)")
    worker_id: Optional[str] = Field(default=None, description="ID of worker that produced this")
    artifact_type: Optional[str] = Field(default=None, description="Type of artifact (finding, claim, citation)")
    confidence: Optional[float] = Field(default=None, description="Confidence score if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "allow"}


class FindingChunker:
    """Converts research memories into entity-extraction-ready chunks.

    Pulls from:
    - LangGraph Store: Items stored by deep_agents via write_file("/memories/")

    The chunker does NOT do any chunking/splitting - it assumes the
    memories are already at an appropriate size for entity extraction.
    """

    def __init__(self, context: Any = None):
        """Initialize the chunker.

        Args:
            context: ExtensionContext (optional, for dependency injection)
        """
        self.context = context

    async def get_findings_as_chunks(
        self,
        store: Any,
        session_id: str,
        max_chunks: int = 100,
    ) -> List[EntityChunk]:
        """Pull findings from LangGraph Store and format for entity extraction.

        Args:
            store: LangGraph BaseStore instance (where deep_agents write memories)
            session_id: Research session ID to filter memories
            max_chunks: Maximum number of chunks to return (prevents overload)

        Returns:
            List of EntityChunk objects ready for entity extraction

        Raises:
            RuntimeError: If store is not available
        """
        if not store:
            raise RuntimeError(
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║        LANGGRAPH STORE NOT AVAILABLE - CANNOT GET CHUNKS     ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
                "\nGraph Builder requires LangGraph Store to pull memories.\n"
                "Deep agents must have write_file access to /memories/.\n"
            )

        chunks = []

        # Query memories from LangGraph Store
        try:
            # Build namespace prefix for this session
            # Use namespace_factory for consistent namespace construction
            from src.extensions.content.deep_research.storage import namespace_prefix
            ns_prefix = namespace_prefix(session_id, "memories")

            logger.info(f"[GRAPH_BUILDER] Querying LangGraph Store for namespace={ns_prefix}")

            # Use asearch without query to list all items in namespace
            # This returns all memories for this session
            results = await store.asearch(
                ns_prefix,
                query=None,  # No semantic filter - get all
                limit=max_chunks,
            )

            logger.info(f"[GRAPH_BUILDER] Found {len(results)} memories for session {session_id}")

            for item in results:
                chunk = self._store_item_to_chunk(item, session_id)
                if chunk:
                    chunks.append(chunk)

        except Exception as e:
            logger.error(f"[GRAPH_BUILDER] Failed to query LangGraph Store: {e}")
            raise RuntimeError(f"Failed to query memories from LangGraph Store: {e}")

        logger.info(f"[GRAPH_BUILDER] Total chunks for entity extraction: {len(chunks)}")
        return chunks[:max_chunks]

    def _store_item_to_chunk(
        self,
        item: Any,
        session_id: str,
    ) -> Optional[EntityChunk]:
        """Convert a LangGraph Store item to an EntityChunk.

        Args:
            item: SearchItem from LangGraph Store
            session_id: Session ID for the chunk

        Returns:
            EntityChunk or None if item is invalid
        """
        try:
            # Extract value from the store item
            value = item.value if hasattr(item, 'value') else item

            # Get content - could be string or list (StoreBackend format)
            content = value.get("content", "")
            if isinstance(content, list):
                text = " ".join(str(c) for c in content)
            else:
                text = str(content) if content else ""

            if not text or len(text.strip()) < 10:
                return None

            # Get metadata
            mem_type = value.get("type", "memory")
            source_url = value.get("url") or value.get("source_url")

            chunk = EntityChunk(
                id=f"memory_{item.key}",
                text=text.strip(),
                source_url=source_url,
                source_type="langgraph_store",
                worker_id=value.get("worker_id"),
                artifact_type=mem_type,
                confidence=value.get("confidence"),
                metadata={
                    "session_id": session_id,
                    "namespace": str(item.namespace) if hasattr(item, 'namespace') else "",
                    "created_at": str(item.created_at) if hasattr(item, 'created_at') else "",
                }
            )

            return chunk

        except Exception as e:
            logger.warning(f"[GRAPH_BUILDER] Failed to convert store item to chunk: {e}")
            return None
