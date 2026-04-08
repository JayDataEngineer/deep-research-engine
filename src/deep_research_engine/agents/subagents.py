"""Sub-agents for ingestion pipeline.

Sort Agent: Categorizes and organizes raw content into structured sections.
Entity Extractor Agent: Extracts and consolidates named entities from text.

Uses the DeepAgents pattern with LangChain-compatible tool integration.
"""

from typing import Any

from deepagents import create_deep_agent
from loguru import logger

from ..shared.infra_client import InfraClient
from ..tools.storage import (
    graph_query,
    initialize_storage_tools,
    vector_search_datastore,
)

# ---------------------------------------------------------------------------
# Deep Agent Tools (shared across all sub-agents)
# ---------------------------------------------------------------------------


def get_deep_agent_tools(config_data: dict[str, Any] | None = None) -> list:
    """Return the full set of Deep Agent tools.

    These are the same tools used by workers and the drafter node:
    - vector_search_datastore: Search vector DB
    - graph_query: Query knowledge graph (Neo4j)
    - initialize_storage_tools: Set up storage backends
    """
    if config_data:
        initialize_storage_tools(config_data)
    return [vector_search_datastore, graph_query]


# ---------------------------------------------------------------------------
# Sort Sub-Agent
# ---------------------------------------------------------------------------

SORT_AGENT_PROMPT = (
    "You are a Content Sort Agent. Your job is to analyze raw ingested content "
    "and organize it into clear, structured categories.\n\n"
    "SORTING RULES:\n"
    "1. GROUP by content type: messages, voice transcripts, photo descriptions, video summaries\n"
    "2. ORDER chronologically within each group using timestamps\n"
    "3. MERGE duplicate or near-duplicate entries\n"
    "4. FLAG low-confidence or incomplete items with [?] markers\n"
    "5. EXTRACT key metadata: sender, timestamp, media type, entity mentions\n\n"
    "OUTPUT FORMAT:\n"
    "## Messages\n"
    "- [timestamp] **sender**: content [entities]\n\n"
    "## Voice Transcripts\n"
    "- [timestamp] **speaker**: transcript [entities]\n\n"
    "## Photo Descriptions\n"
    "- [timestamp] **description** [entities]\n\n"
    "## Video Summaries\n"
    "- [timestamp] **summary** [entities]\n\n"
    "Be thorough but concise. Preserve all original information — only reorganize it."
)


def _ensure_llm_loaded(model: str) -> None:
    """Ensure the LLM is loaded before running a sub-agent."""
    try:
        infra = InfraClient()
        infra.load("llm", model, wait=True)
    except Exception as e:
        logger.warning(f"Could not reload LLM {model} for sub-agent (may be cloud): {e}")


def create_sort_agent(
    model: str = "gemma4-e4b-vision-fast",
    tools: list | None = None,
) -> Any:
    """Create a Sort SubAgent using DeepAgents pattern.

    Args:
        model: Model identifier (passed to LiteLLM/deepagents).
        tools: Optional list of tools. Defaults to no tools (pure text processing).

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    # Sort agent is pure text processing — no storage tools needed by default
    if tools is None:
        tools = []

    return create_deep_agent(
        model=model,
        system_prompt=SORT_AGENT_PROMPT,
        tools=tools,
    )


async def run_sort_agent(
    content: str,
    model: str = "gemma4-e4b-vision-fast",
    config_data: dict[str, Any] | None = None,
) -> str:
    """Run the Sort Agent on raw content.

    Args:
        content: Raw ingested text to sort.
        model: Model to use.
        config_data: Optional config for storage tools.

    Returns:
        Sorted and structured content string.
    """
    # Ensure LLM is loaded (may have been unloaded by ASR)
    _ensure_llm_loaded(model)

    # Sort agent doesn't need storage tools — pure text processing
    tools: list = []
    agent = create_sort_agent(model=model, tools=tools)

    # Chunk large content to avoid truncation (process in 50K char chunks)
    chunk_size = 50_000
    if len(content) <= chunk_size:
        chunks = [content]
    else:
        # Split at paragraph boundaries to avoid cutting mid-sentence
        chunks = []
        pos = 0
        while pos < len(content):
            end = min(pos + chunk_size, len(content))
            if end < len(content):
                # Find nearest paragraph break
                para_break = content.rfind("\n\n", pos, end)
                if para_break > pos:
                    end = para_break + 2
            chunks.append(content[pos:end])
            pos = end
        logger.info(f"Split sort input into {len(chunks)} chunks ({len(content)} chars total)")

    sorted_parts = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing sort chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        result = await agent.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        f"Sort and organize the following raw content (part {i + 1} of {len(chunks)}):\n\n{chunk}\n\n"
                        f"Apply the sorting rules from your system prompt.",
                    )
                ]
            }
        )
        sorted_parts.append(result["messages"][-1].content)

    if len(sorted_parts) == 1:
        sorted_content = sorted_parts[0]
    else:
        # Run a final merge pass to deduplicate across chunks
        merged_input = "\n\n".join(sorted_parts)
        merge_agent = create_sort_agent(model=model, tools=[])
        merge_result = await merge_agent.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        f"Merge these sorted sections into a single organized document, "
                        f"deduplicating any overlapping entries:\n\n{merged_input}",
                    )
                ]
            }
        )
        sorted_content = merge_result["messages"][-1].content

    logger.info(f"Sort Agent completed: {len(sorted_content)} chars output")
    return sorted_content

    sorted_content = result["messages"][-1].content
    logger.info(f"Sort Agent completed: {len(sorted_content)} chars output")
    return sorted_content


# ---------------------------------------------------------------------------
# Entity Extractor Sub-Agent
# ---------------------------------------------------------------------------

ENTITY_EXTRACTOR_PROMPT = (
    "You are an Entity Extraction Agent. Your job is to analyze text and extract "
    "all named entities into a structured, deduplicated format.\n\n"
    "ENTITY TYPES TO EXTRACT:\n"
    "1. **People**: Full names, aliases, usernames, handles\n"
    "2. **Locations**: Cities, states, countries, addresses, landmarks\n"
    "3. **Organizations**: Groups, companies, channels, communities\n"
    "4. **Dates/Times**: Specific dates, time ranges, relative dates\n"
    "5. **Topics**: Key subjects, events, projects, movements\n"
    "6. **Identifiers**: Phone numbers, license plates, social media handles, URLs\n\n"
    "EXTRACTION RULES:\n"
    "1. EXTRACT every entity mentioned, even in passing\n"
    "2. DEDUPLICATE — merge variations (e.g., 'WLM Montana' = 'White Lives Matter Montana')\n"
    "3. PRESERVE context — note what the entity was mentioned in relation to\n"
    "4. FLAG uncertain entities with [?]\n"
    "5. DO NOT infer or hallucinate entities not present in the text\n\n"
    "OUTPUT FORMAT:\n"
    "## People\n"
    "- Name: [context/relation]\n\n"
    "## Locations\n"
    "- Location: [context/relation]\n\n"
    "## Organizations\n"
    "- Organization: [context/relation]\n\n"
    "## Dates/Times\n"
    "- Date: [context/relation]\n\n"
    "## Topics\n"
    "- Topic: [context/relation]\n\n"
    "## Identifiers\n"
    "- Identifier: [context/relation]\n\n"
    "Be comprehensive. Extract ALL entities from the provided text."
)


def create_entity_extractor_agent(
    model: str = "gemma4-e4b-vision-fast",
    tools: list | None = None,
) -> Any:
    """Create an Entity Extractor SubAgent using DeepAgents pattern.

    Args:
        model: Model identifier (passed to LiteLLM/deepagents).
        tools: Optional list of tools. Defaults to no tools (pure text processing).

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    # Entity extraction is pure text processing — no storage tools needed
    if tools is None:
        tools = []

    return create_deep_agent(
        model=model,
        system_prompt=ENTITY_EXTRACTOR_PROMPT,
        tools=tools,
    )


async def run_entity_extractor_agent(
    content: str,
    model: str = "gemma4-e4b-vision-fast",
    config_data: dict[str, Any] | None = None,
) -> str:
    """Run the Entity Extractor Agent on text content.

    Args:
        content: Text to extract entities from.
        model: Model to use.
        config_data: Optional config for storage tools.

    Returns:
        Structured entity extraction result string.
    """
    # Ensure LLM is loaded (may have been unloaded by ASR)
    _ensure_llm_loaded(model)

    # Entity extraction doesn't need storage tools — pure text processing
    tools: list = []
    agent = create_entity_extractor_agent(model=model, tools=tools)

    # Chunk large content to avoid truncation
    chunk_size = 50_000
    if len(content) <= chunk_size:
        chunks = [content]
    else:
        chunks = []
        pos = 0
        while pos < len(content):
            end = min(pos + chunk_size, len(content))
            if end < len(content):
                para_break = content.rfind("\n\n", pos, end)
                if para_break > pos:
                    end = para_break + 2
            chunks.append(content[pos:end])
            pos = end
        logger.info(f"Split entity extraction into {len(chunks)} chunks")

    entity_parts = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Extracting entities from chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        result = await agent.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        f"Extract all named entities from the following text (part {i + 1} of {len(chunks)}):\n\n{chunk}\n\n"
                        f"Apply the extraction rules from your system prompt.",
                    )
                ]
            }
        )
        entity_parts.append(result["messages"][-1].content)

    if len(entity_parts) == 1:
        entities = entity_parts[0]
    else:
        # Merge and deduplicate entity lists
        merged = "\n\n".join(entity_parts)
        merge_agent = create_entity_extractor_agent(model=model, tools=[])
        merge_result = await merge_agent.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        f"Merge and deduplicate these entity lists into a single comprehensive list:\n\n{merged}",
                    )
                ]
            }
        )
        entities = merge_result["messages"][-1].content

    logger.info(f"Entity Extractor completed: {len(entities)} chars output")
    return entities


# ---------------------------------------------------------------------------
# Pipeline Integration: Run both agents sequentially
# ---------------------------------------------------------------------------


async def run_sort_then_extract(
    content: str,
    model: str = "gemma4-e4b-vision-fast",
    config_data: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Run Sort Agent first, then Entity Extractor on the sorted output.

    This two-pass approach produces cleaner entities because the content
    is already organized and deduplicated before entity extraction.

    Args:
        content: Raw ingested text.
        model: Model to use for both agents.
        config_data: Optional config for storage tools.

    Returns:
        Dict with 'sorted_content' and 'entities' keys.
    """
    logger.info("Running Sort Agent...")
    sorted_content = await run_sort_agent(content, model=model, config_data=config_data)

    logger.info("Running Entity Extractor Agent...")
    entities = await run_entity_extractor_agent(
        sorted_content, model=model, config_data=config_data
    )

    return {
        "sorted_content": sorted_content,
        "entities": entities,
    }
