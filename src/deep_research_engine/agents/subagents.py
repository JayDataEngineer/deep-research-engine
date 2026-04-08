"""Sub-agents for ingestion pipeline.

Sort Agent: Categorizes and organizes raw content into structured sections.
Entity Extractor Agent: Extracts and consolidates named entities from text.
Entity Finder Agent: Builds per-entity profiles from sorted content.

Uses the DeepAgents pattern with LangChain-compatible tool integration.
"""

from typing import Any

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
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


def _make_chat_model(model: str = "gemma4-e4b-vision-fast") -> ChatOpenAI:
    """Create a ChatOpenAI instance pointing at the local orchestrator.

    Maps infra model keys to the orchestrator's OpenAI-compatible endpoint,
    avoiding LangChain's provider resolution which can't handle bare names.
    """
    import os

    # Map infra model key to the API model name the orchestrator expects
    model_name_map = {
        "gemma4-e4b-vision-fast": "gemma-4-e4b",
        "gemma4-e4b-vision": "gemma-4-e4b",
        "qwen35-35b-a3b-vision-fast": "qwen35-35b-a3b-vision-fast",
    }
    api_model = model_name_map.get(model, model)

    # Build the base URL — use the same discovery logic as InfraClient
    base_url = os.environ.get(
        "INFRA_ORCHESTRATOR_URL",
        "http://localhost:18880",
    ).rstrip("/")
    api_key = os.environ.get("LITELLM_PROXY_KEY", "sk-litellm-e2e-test-master-key")

    return ChatOpenAI(
        model=api_model,
        base_url=f"{base_url}/v1",
        api_key=api_key,
        default_headers={"Host": "api.orchestrator.local"},
        temperature=0.3,
        max_tokens=16000,
    )


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
        model=_make_chat_model(model),
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
        model=_make_chat_model(model),
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


# ---------------------------------------------------------------------------
# Entity Finder Sub-Agent (per-entity deep profile builder)
# ---------------------------------------------------------------------------


ENTITY_FINDER_PROMPT = (
    "You are an Entity Profile Agent. You receive a single entity (person or event) "
    "and a body of text. Your job is to find EVERY mention of that entity in the text "
    "and build a comprehensive profile.\n\n"
    "PROFILE STRUCTURE:\n"
    "1. **Name/Alias**: All names, nicknames, handles used for this entity\n"
    "2. **Timeline**: Chronological list of every mention with date + context\n"
    "3. **Relationships**: People connected to this entity and how\n"
    "4. **Key Facts**: Specific details — locations, roles, events, affiliations\n"
    "5. **Sentiment/Tone**: How the entity is discussed (positive, negative, neutral)\n"
    "6. **Open Questions**: Things mentioned but unclear or incomplete\n\n"
    "RULES:\n"
    "- Include ONLY information found in the provided text\n"
    "- Quote directly when possible\n"
    "- Note contradictions or changes over time\n"
    "- If the entity is barely mentioned, report what little exists\n"
    "- Be thorough — a single passing mention still has context\n\n"
    "OUTPUT: A structured markdown profile for this entity."
)


def _extract_entity_names(entities_text: str) -> list[str]:
    """Extract entity names from the Entity Extractor Agent's output.

    Parses the ## People section to get individual names.
    """
    names: list[str] = []
    in_people_section = False
    for line in entities_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## People"):
            in_people_section = True
            continue
        if stripped.startswith("## ") and in_people_section:
            break  # Next section
        if in_people_section and stripped.startswith("- "):
            # Extract name before any colon or bracket
            name = stripped[2:].split(":")[0].split("[")[0].strip()
            # Remove trailing [?] uncertainty marker
            name = name.replace("[?]", "").strip()
            if name and len(name) > 1:
                names.append(name)
    return names


def create_entity_finder_agent(
    model: str = "gemma4-e4b-vision-fast",
    tools: list | None = None,
) -> Any:
    """Create an Entity Finder SubAgent using DeepAgents pattern.

    Args:
        model: Model identifier.
        tools: Optional tools. Defaults to none.

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    if tools is None:
        tools = []
    return create_deep_agent(
        model=_make_chat_model(model),
        system_prompt=ENTITY_FINDER_PROMPT,
        tools=tools,
    )


async def run_entity_finder(
    entity_name: str,
    content: str,
    model: str = "gemma4-e4b-vision-fast",
) -> str:
    """Run the Entity Finder Agent for a single entity.

    Searches through content for all mentions of the given entity
    and produces a structured profile.

    Args:
        entity_name: Name of the entity to profile.
        content: Sorted content to search through.
        model: Model to use.

    Returns:
        Structured markdown profile for the entity.
    """
    _ensure_llm_loaded(model)

    agent = create_entity_finder_agent(model=model)

    # For very large content, search for relevant chunks containing the entity
    # to avoid sending the entire document for each entity
    relevant_chunks = _find_relevant_chunks(entity_name, content, chunk_size=60_000)

    if not relevant_chunks:
        return f"## {entity_name}\n\nNo mentions found in the content.\n"

    relevant_text = "\n\n".join(relevant_chunks)
    logger.info(
        f"Entity Finder: searching for '{entity_name}' in {len(relevant_text)} chars "
        f"({len(relevant_chunks)} chunks)"
    )

    result = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    f"Build a comprehensive profile for the entity: **{entity_name}**\n\n"
                    f"Search through the following text for every mention:\n\n{relevant_text}",
                )
            ]
        }
    )

    profile = result["messages"][-1].content
    logger.info(f"Entity Finder: profile for '{entity_name}' — {len(profile)} chars")
    return profile


def _find_relevant_chunks(entity_name: str, content: str, chunk_size: int = 60_000) -> list[str]:
    """Find content chunks that mention the entity.

    Uses case-insensitive substring matching to locate relevant sections.
    Includes surrounding context (the full chunk containing each mention).

    Returns:
        List of text chunks that mention the entity.
    """
    name_lower = entity_name.lower()
    # Also check first name only (for "John Smith" → also match "John")
    name_parts = name_lower.split()
    search_terms = [name_lower]
    if len(name_parts) > 1:
        search_terms.append(name_parts[0])

    # Split into paragraphs and filter
    paragraphs = content.split("\n\n")
    relevant: list[str] = []
    current_chunk: list[str] = []
    chunk_len = 0

    for para in paragraphs:
        para_lower = para.lower()
        is_relevant = any(term in para_lower for term in search_terms)

        if is_relevant:
            # Include surrounding context (1 paragraph before and after)
            current_chunk.append(para)
            chunk_len += len(para)
        else:
            # If we're building a relevant chunk, include adjacent paragraphs for context
            if current_chunk:
                current_chunk.append(para)
                chunk_len += len(para)
                # Flush chunk if it's getting large
                if chunk_len >= chunk_size:
                    relevant.append("\n\n".join(current_chunk))
                    current_chunk = []
                    chunk_len = 0

        # Flush if chunk is large
        if chunk_len >= chunk_size:
            relevant.append("\n\n".join(current_chunk))
            current_chunk = []
            chunk_len = 0

    # Flush remaining
    if current_chunk:
        relevant.append("\n\n".join(current_chunk))

    return relevant


async def run_entity_finder_batch(
    entities_text: str,
    sorted_content: str,
    model: str = "gemma4-e4b-vision-fast",
    max_entities: int = 10,
) -> dict[str, str]:
    """Run the Entity Finder Agent for all extracted entities.

    Takes the output of the Entity Extractor Agent and builds profiles
    for each person found.

    Args:
        entities_text: Entity Extractor Agent output (## People section).
        sorted_content: Sorted content from the Sort Agent.
        model: Model to use.
        max_entities: Maximum number of entities to profile.

    Returns:
        Dict mapping entity names to their profiles.
    """
    names = _extract_entity_names(entities_text)
    if not names:
        logger.info("Entity Finder: no person names found in entities text")
        return {}

    # Limit to top N entities
    names = names[:max_entities]
    logger.info(f"Entity Finder: building profiles for {len(names)} entities")

    _ensure_llm_loaded(model)

    profiles: dict[str, str] = {}
    for i, name in enumerate(names):
        logger.info(f"Profiling entity {i + 1}/{len(names)}: {name}")
        try:
            profile = await run_entity_finder(name, sorted_content, model=model)
            profiles[name] = profile
        except Exception as e:
            logger.warning(f"Entity Finder failed for '{name}': {e}")
            profiles[name] = f"## {name}\n\nProfile extraction failed: {e}\n"

    return profiles


def format_entity_profiles(profiles: dict[str, str]) -> str:
    """Format entity profiles into a single markdown document for the synthesis LLM."""
    if not profiles:
        return ""

    parts = ["# Entity Profiles\n"]
    parts.extend(f"\n---\n\n{profile}\n" for profile in profiles.values())

    return "\n".join(parts)
