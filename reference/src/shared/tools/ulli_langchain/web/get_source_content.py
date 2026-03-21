"""Get Source Content Tool using Command API pattern (LangGraph V1).

This follows LangChain handoffs documentation:
- Uses @tool decorator instead of BaseTool class
- Returns Command(update={...}) to update state
- No in-place state mutations
- State flows naturally through the graph

Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""

import hashlib
from .logging_config import get_logger
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

logger = get_logger(__name__)


class GetSourceContentInput(BaseModel):
    """Input for retrieving source content."""
    source_ids: List[str] = Field(
        ...,
        description="List of source IDs to retrieve. These IDs are stored in WorkerResult.source_refs"
    )
    max_length: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Maximum characters per source (default: 5000)"
    )
    include_metadata: bool = Field(
        default=False,
        description="Include metadata (tier, relevance_score, timestamp)"
    )


def generate_source_id(url: str) -> str:
    """Generate a deterministic source ID from URL.

    This ensures that if multiple workers scrape the same URL,
    they automatically map to the same source_id (deduplication).

    Args:
        url: The source URL

    Returns:
        MD5 hash of the URL (hex string)
    """
    return hashlib.md5(url.encode()).hexdigest()


async def _retrieve_sources_impl(
    source_ids: List[str],
    max_length: int,
    include_metadata: bool,
    research_id: str,
    db_path: str,
) -> str:
    """Retrieve source content from database.

    This is the core retrieval logic - extracted so it can be reused.

    Args:
        source_ids: List of source IDs to retrieve
        max_length: Maximum characters per source
        include_metadata: Include metadata information
        research_id: Research ID for database lookup
        db_path: Path to database

    Returns:
        Formatted source content
    """
    # Import ResearchDatabase
    from src.extensions.content.deep_research.storage.schema import ResearchDatabase

    db = ResearchDatabase(db_path)

    results = []
    for source_id in source_ids:
        # Get source from database
        row = db.get_source(source_id)
        if not row:
            # Try to get by research_id
            row = db.get_source_by_id(research_id, source_id)

        if not row:
            results.append(f"Source ID '{source_id}' not found")
            continue

        # Convert database row to dict format
        # EXPLICIT CONTENT SELECTION: Priority order with explicit failure
        # Priority: summary > snippet > full_content
        # This matches the Information Funnel tier architecture:
        # - Tier A: Has summary (AI-generated abstract)
        # - Tier B: Has snippet (search result preview)
        # - Tier C: Has full_content (raw scrape)
        # FAIL EXPLICITLY: If no content available, raise error (anti-fragile system)
        if row.get("summary"):
            content = row["summary"]
            logger.debug(f"[GET_SOURCE_CONTENT] {source_id}: Using summary (Tier A)")
        elif row.get("snippet"):
            content = row["snippet"]
            logger.debug(f"[GET_SOURCE_CONTENT] {source_id}: Using snippet (Tier B)")
        elif row.get("full_content"):
            content = row["full_content"]
            logger.debug(f"[GET_SOURCE_CONTENT] {source_id}: Using full_content (Tier C)")
        else:
            # FAIL EXPLICITLY: No content available - this indicates data corruption
            raise ValueError(
                f"╔═══════════════════════════════════════════════════════════════════╗\n"
                f"║           SOURCE CONTENT ERROR: {source_id:34} ║\n"
                f"╚═══════════════════════════════════════════════════════════════════╝\n"
                f"\n"
                f"Source '{source_id}' has no content available.\n"
                f"\n"
                f"Database state:\n"
                f"  - summary: {row.get('summary', 'MISSING')}\n"
                f"  - snippet: {row.get('snippet', 'MISSING')}\n"
                f"  - full_content: {row.get('full_content', 'MISSING')}\n"
                f"\n"
                f"This indicates data corruption or incomplete ingestion.\n"
                f"Anti-fragile systems should prevent this - check your ingestion pipeline.\n"
                f"\n"
            )

        source = {
            "source_id": row["id"],
            "url": row["url"],
            "title": row["title"] or "Untitled",
            "content": content,
            "snippet": row["snippet"],
            "summary": row["summary"],
            "tier": row["tier"],
            "relevance_score": row["relevance_score"],
            "timestamp": row["retrieved_at"],
        }

        # Build output
        output = []
        output.append(f"**{source['title']}**")
        output.append(f"Source ID: {source_id}")
        output.append(f"URL: {source['url']}")

        if include_metadata:
            output.append(f"Tier: {source['tier']}")
            output.append(f"Relevance: {source['relevance_score']:.2f}")
            output.append(f"Retrieved: {source['timestamp']}")

        output.append("")  # Blank line

        # Add content (truncated if needed)
        content = source['content']
        if len(content) > max_length:
            content = content[:max_length] + f"...[truncated, {len(content)} total chars]"

        output.append(content)
        output.append("")  # Blank line between sources
        output.append("-" * 60)
        output.append("")

        results.append("\n".join(output))

    return "\n".join(results) if results else "No sources found"


@tool
async def get_source_content(
    source_ids: List[str],
    max_length: int = 5000,
    include_metadata: bool = False,
    runtime: Optional[ToolRuntime] = None,
) -> Command:
    """Retrieve full content from stored sources.

    Use this tool when you need to:
    - Verify a specific claim or fact from a source
    - Get more detailed information from a Tier B snippet
    - Access the full markdown content of a source
    - Expand on a brief mention to get the complete context

    The source IDs are provided in the research findings as "source_id: [id]" or
    in the source_refs field.

    Args:
        source_ids: List of source IDs to retrieve (e.g., ["abc123", "def456"])
        max_length: Maximum characters per source (default: 5000, max: 50000)
        include_metadata: Include tier/relevance/timestamp info (default: false)
        runtime: ToolRuntime context for accessing state

    Returns:
        Command with ToolMessage (formatted source content)
    """
    if runtime is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: Tool runtime context not available",
                        tool_call_id=""
                    )
                ]
            }
        )

    # Get research_id from state
    research_id = runtime.state.get("research_id", "")
    if not research_id:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: No research_id found in state",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )

    db_path = runtime.state.get("db_path", "data/research/sources.db")

    try:
        # Retrieve sources
        result = await _retrieve_sources_impl(
            source_ids=source_ids,
            max_length=max_length,
            include_metadata=include_metadata,
            research_id=research_id,
            db_path=db_path
        )

        logger.info(f"[GET_SOURCE_CONTENT] Retrieved {len(source_ids)} sources")

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )

    except Exception as e:
        logger.error(f"[GET_SOURCE_CONTENT] Error: {e}", exc_info=True)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error retrieving sources: {str(e)}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )


# Export the tool function
__all__ = ["get_source_content", "generate_source_id"]
