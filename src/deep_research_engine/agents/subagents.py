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
    "IMPORTANT: Use the graph_query tool to look up existing entities in the knowledge graph "
    "BEFORE sorting. This helps you understand known people, their relationships, and how "
    "to group content around them. Query common names from the text to see what's already known.\n\n"
    "SORTING RULES:\n"
    "1. GROUP by content type: messages, voice transcripts, photo descriptions, video summaries\n"
    "2. ORDER chronologically within each group using timestamps\n"
    "3. MERGE duplicate or near-duplicate entries\n"
    "4. FLAG low-confidence or incomplete items with [?] markers\n"
    "5. EXTRACT key metadata: sender, timestamp, media type, entity mentions\n"
    "6. CROSS-REFERENCE entities with the knowledge graph — annotate known entities with "
    "their graph relationships (e.g., [KNOWN: works_at X, related_to Y])\n\n"
    "OUTPUT FORMAT:\n"
    "## Messages\n"
    "- [timestamp] **sender**: content [entities] [graph context if available]\n\n"
    "## Voice Transcripts\n"
    "- [timestamp] **speaker**: transcript [entities] [graph context if available]\n\n"
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
    config_data: dict[str, Any] | None = None,
) -> Any:
    """Create a Sort SubAgent using DeepAgents pattern.

    Args:
        model: Model identifier (passed to LiteLLM/deepagents).
        tools: Optional list of tools. Defaults to graph_query for entity lookup.
        config_data: Config for initializing storage tools.

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    if tools is None:
        tools = get_deep_agent_tools(config_data)

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

    agent = create_sort_agent(model=model, config_data=config_data)

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
        merge_agent = create_sort_agent(model=model, config_data=config_data)
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
    "IMPORTANT: Use the graph_query tool to look up existing entities in the knowledge graph. "
    "This helps you resolve ambiguous names and understand entity relationships. "
    "When you find a name, query the graph to see if it's already known and what "
    "relationships it has. This ensures extracted entities align with existing graph data.\n\n"
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
    "4. CROSS-REFERENCE with the knowledge graph — annotate known entities with "
    "[KNOWN_IN_GRAPH] and include their relationships\n"
    "5. FLAG uncertain entities with [?]\n"
    "6. DO NOT infer or hallucinate entities not present in the text\n\n"
    "OUTPUT FORMAT:\n"
    "## People\n"
    "- Name: [context/relation] [KNOWN_IN_GRAPH: relationships if applicable]\n\n"
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
    config_data: dict[str, Any] | None = None,
) -> Any:
    """Create an Entity Extractor SubAgent using DeepAgents pattern.

    Args:
        model: Model identifier (passed to LiteLLM/deepagents).
        tools: Optional list of tools. Defaults to graph_query for entity lookup.
        config_data: Config for initializing storage tools.

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    if tools is None:
        tools = get_deep_agent_tools(config_data)

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

    agent = create_entity_extractor_agent(model=model, config_data=config_data)

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
        merge_agent = create_entity_extractor_agent(model=model, config_data=config_data)
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
    "IMPORTANT: Use the graph_query tool to look up this entity in the knowledge graph. "
    "Query their name to find existing relationships, known aliases, and connected entities. "
    "This context enriches the profile and connects new mentions to existing knowledge.\n\n"
    "PROFILE STRUCTURE:\n"
    "1. **Name/Alias**: All names, nicknames, handles used for this entity\n"
    "2. **Timeline**: Chronological list of every mention with date + context\n"
    "3. **Relationships**: People connected to this entity and how (cross-reference with graph)\n"
    "4. **Key Facts**: Specific details — locations, roles, events, affiliations\n"
    "5. **Sentiment/Tone**: How the entity is discussed (positive, negative, neutral)\n"
    "6. **Open Questions**: Things mentioned but unclear or incomplete\n"
    "7. **Graph Context**: Any existing knowledge graph relationships found\n\n"
    "RULES:\n"
    "- Include ONLY information found in the provided text or knowledge graph\n"
    "- Quote directly when possible\n"
    "- Note contradictions or changes over time\n"
    "- If the entity is barely mentioned, report what little exists\n"
    "- Be thorough — a single passing mention still has context\n"
    "- Mark graph-sourced info with [FROM_GRAPH] so downstream knows the origin\n\n"
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
    config_data: dict[str, Any] | None = None,
) -> Any:
    """Create an Entity Finder SubAgent using DeepAgents pattern.

    Args:
        model: Model identifier.
        tools: Optional tools. Defaults to graph_query for entity lookup.
        config_data: Config for initializing storage tools.

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    if tools is None:
        tools = get_deep_agent_tools(config_data)
    return create_deep_agent(
        model=_make_chat_model(model),
        system_prompt=ENTITY_FINDER_PROMPT,
        tools=tools,
    )


async def run_entity_finder(
    entity_name: str,
    content: str,
    model: str = "gemma4-e4b-vision-fast",
    config_data: dict[str, Any] | None = None,
) -> str:
    """Run the Entity Finder Agent for a single entity.

    Searches through content for all mentions of the given entity
    and produces a structured profile enriched with graph context.

    Args:
        entity_name: Name of the entity to profile.
        content: Sorted content to search through.
        model: Model to use.
        config_data: Config for initializing storage tools.

    Returns:
        Structured markdown profile for the entity.
    """
    _ensure_llm_loaded(model)

    agent = create_entity_finder_agent(model=model, config_data=config_data)

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
    config_data: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Run the Entity Finder Agent for all extracted entities.

    Takes the output of the Entity Extractor Agent and builds profiles
    for each person found, enriched with knowledge graph context.

    Args:
        entities_text: Entity Extractor Agent output (## People section).
        sorted_content: Sorted content from the Sort Agent.
        model: Model to use.
        max_entities: Maximum number of entities to profile.
        config_data: Config for initializing storage tools.

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
            profile = await run_entity_finder(
                name, sorted_content, model=model, config_data=config_data,
            )
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


# ---------------------------------------------------------------------------
# Phase 6: Identity Resolver Sub-Agent
# ---------------------------------------------------------------------------


IDENTITY_RESOLVER_PROMPT = (
    "You are an Identity Resolution Agent. Your job is to analyze multimodal data "
    "and produce a unified identity mapping that links speaker IDs, face clusters, "
    "and entity names to real people.\n\n"
    "INPUT DATA YOU RECEIVE:\n"
    "- ASR transcript segments with speaker labels and timestamps\n"
    "- Visual face cluster events from photos/videos\n"
    "- Text messages with sender names and timestamps\n"
    "- Pre-NER detected person names\n"
    "- Existing graph identities (queried from knowledge graph)\n\n"
    "RESOLUTION STRATEGY:\n"
    "1. FIRST, use the graph_query tool to look up any existing persons in the "
    "knowledge graph. Match new speaker/cluster IDs to known names.\n"
    "2. Cross-reference speaker IDs across files — speakers with similar context "
    "(discussing same people, places, events) are likely the same person.\n"
    "3. Match face clusters to entity names when the timestamp and context align.\n"
    "4. Use text message senders as anchor identities — if a sender name appears "
    "in the same temporal context as a speaker ID, they are likely the same person.\n"
    "5. Pre-NER names are high-confidence — prioritize matching to these.\n\n"
    "RULES:\n"
    "- Only map IDs to names when you have reasonable confidence\n"
    "- Flag uncertain mappings with [?] suffix\n"
    "- Unmapped IDs keep their raw form (Speaker_0, Identity_Cluster_1, etc.)\n"
    "- Multiple raw IDs can map to the same resolved name\n\n"
    "OUTPUT: A JSON object mapping raw IDs to resolved names.\n"
    "Example: {\"Speaker_0\": \"Alice\", \"Speaker_1\": \"Bob\", "
    "\"Identity_Cluster_2\": \"Alice\", \"file0:Speaker_0\": \"Alice\"}\n"
)


def create_identity_resolver_agent(
    model: str = "gemma4-e4b-vision-fast",
    tools: list | None = None,
) -> Any:
    """Create an Identity Resolver SubAgent using DeepAgents pattern.

    Uses graph_query tool to iteratively look up existing identities
    from the knowledge graph during resolution.

    Args:
        model: Model identifier.
        tools: Optional tools. Defaults to graph_query + vector_search.

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    if tools is None:
        tools = get_deep_agent_tools()
    return create_deep_agent(
        model=_make_chat_model(model),
        system_prompt=IDENTITY_RESOLVER_PROMPT,
        tools=tools,
    )


def _format_asr_summary(asr_segments: list[dict[str, Any]], max_segments: int = 100) -> str:
    """Format ASR segments into a compact summary for the identity resolver.

    Smart truncation: keeps first N and last N segments, plus any segments
    that mention pre-NER names.
    """
    if not asr_segments:
        return "No ASR segments available."

    if len(asr_segments) <= max_segments:
        lines = [f"[{s['start']:.1f}s] {s['speaker']}: {s['text']}" for s in asr_segments]
        return "\n".join(lines)

    # Keep first 50, last 50
    head = asr_segments[: max_segments // 2]
    tail = asr_segments[-(max_segments // 2) :]
    lines = [f"[{s['start']:.1f}s] {s['speaker']}: {s['text']}" for s in head]
    lines.append(f"... ({len(asr_segments) - max_segments} segments omitted) ...")
    lines.extend(f"[{s['start']:.1f}s] {s['speaker']}: {s['text']}" for s in tail)
    return "\n".join(lines)


def _format_visual_summary(visual_events: list[dict[str, Any]], max_events: int = 100) -> str:
    """Format visual events into a compact summary."""
    if not visual_events:
        return "No visual events available."
    events = visual_events[:max_events]
    lines = [f"[{v.get('timestamp', 'N/A')}] {v['cluster']} in {v.get('path', 'unknown')}" for v in events]
    if len(visual_events) > max_events:
        lines.append(f"... ({len(visual_events) - max_events} more events)")
    return "\n".join(lines)


def _format_text_summary(text_messages: list[dict[str, Any]], max_messages: int = 100) -> str:
    """Format text messages into a compact summary."""
    if not text_messages:
        return "No text messages available."
    messages = text_messages[:max_messages]
    lines = [f"[{t['timestamp']}] {t['sender']}: {t['text'][:200]}" for t in messages]
    if len(text_messages) > max_messages:
        lines.append(f"... ({len(text_messages) - max_messages} more messages)")
    return "\n".join(lines)


async def run_identity_resolver(
    asr_segments: list[dict[str, Any]],
    visual_events: list[dict[str, Any]],
    text_messages: list[dict[str, Any]],
    pre_named_entities: list[str] | None = None,
    model: str = "gemma4-e4b-vision-fast",
    config_data: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Run the Identity Resolver Agent for Phase 6.

    Uses DeepAgents with graph_query tool to iteratively resolve speaker IDs,
    face clusters, and entity names to real people.

    Args:
        asr_segments: ASR transcript segments with speaker labels.
        visual_events: Face cluster events from photos/videos.
        text_messages: Text messages with sender names.
        pre_named_entities: Pre-NER detected person names.
        model: Model to use.
        config_data: Config for storage tools.

    Returns:
        Identity mapping from raw IDs to resolved names.
    """
    _ensure_llm_loaded(model)

    tools = get_deep_agent_tools(config_data)
    agent = create_identity_resolver_agent(model=model, tools=tools)

    # Build context with smart chunking
    asr_summary = _format_asr_summary(asr_segments)
    visual_summary = _format_visual_summary(visual_events)
    text_summary = _format_text_summary(text_messages)

    # Build entity names section
    entity_section = ""
    if pre_named_entities:
        entity_section = (
            "PRE-NER DETECTED NAMES (high-confidence — match IDs to these first):\n"
            + "\n".join(f"- {name}" for name in pre_named_entities)
        )

    # Build user message
    parts = [
        "Resolve the following identities. Use graph_query to look up existing persons first.\n",
        f"## ASR Segments ({len(asr_segments)} total)\n{asr_summary}\n",
        f"## Visual Events ({len(visual_events)} total)\n{visual_summary}\n",
        f"## Text Messages ({len(text_messages)} total)\n{text_summary}\n",
    ]
    if entity_section:
        parts.append(f"## Pre-NER Names\n{entity_section}\n")
    parts.append(
        "Produce a JSON identity mapping. Only include IDs you can confidently resolve.\n"
        'Format: {"raw_id": "Resolved Name", ...}'
    )

    user_message = "\n\n".join(parts)

    result = await agent.ainvoke({"messages": [("user", user_message)]})
    response_text = result["messages"][-1].content

    # Parse JSON mapping from response
    mapping = _parse_identity_mapping(response_text)
    logger.info(f"Identity Resolver: {len(mapping)} identities resolved")
    return mapping


def _parse_identity_mapping(response: str) -> dict[str, str]:
    """Parse identity mapping from LLM response text.

    Handles both raw JSON and markdown-fenced JSON blocks.
    """
    import json
    import re

    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1).strip())
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON object
    brace_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    # Try line-by-line parsing: "Speaker_0": "Alice"
    mapping: dict[str, str] = {}
    for line in response.split("\n"):
        # Match patterns like: "Speaker_0": "Alice" or Speaker_0: Alice
        m = re.match(r'["\']?([^"\']+)["\']?\s*:\s*["\']([^"\']+)["\']', line.strip())
        if m:
            mapping[m.group(1).strip()] = m.group(2).strip()

    return mapping


# ---------------------------------------------------------------------------
# Phase 7: Context Enrichment Sub-Agent
# ---------------------------------------------------------------------------


CONTEXT_ENRICHMENT_PROMPT = (
    "You are a Context Enrichment Agent. Your job is to process enriched pipeline "
    "items and extract structured entity information while managing context window "
    "constraints.\n\n"
    "You receive batches of content items (messages, transcripts, descriptions) and "
    "must extract named entities from each batch.\n\n"
    "EXTRACTION RULES:\n"
    "1. Extract: People, Locations, Organizations, Dates, Topics\n"
    "2. Deduplicate entities across items within the batch\n"
    "3. Track which items mention which entities\n"
    "4. Flag uncertain extractions with [?]\n\n"
    "OUTPUT FORMAT:\n"
    "For each batch, output a JSON object with:\n"
    "- 'entities': Combined entity list by category\n"
    "- 'item_entities': Mapping of item index to entities found\n\n"
    "Be comprehensive. Extract ALL entities from the provided items."
)


def create_context_enrichment_agent(
    model: str = "gemma4-e4b-vision-fast",
    tools: list | None = None,
) -> Any:
    """Create a Context Enrichment SubAgent using DeepAgents pattern.

    Processes items in batches for entity extraction with context management.

    Args:
        model: Model identifier.
        tools: Optional tools. Defaults to none (pure text processing).

    Returns:
        A DeepAgent instance that can be invoked with .ainvoke().
    """
    if tools is None:
        tools = []
    return create_deep_agent(
        model=_make_chat_model(model),
        system_prompt=CONTEXT_ENRICHMENT_PROMPT,
        tools=tools,
    )


async def run_context_enrichment(
    items: list[dict[str, Any]],
    resolved_identity_map: dict[str, str],
    audio_results: dict[str, list[dict[str, Any]]],
    video_descriptions: dict[str, str],
    identity_map: dict[str, list[str]],
    model: str = "gemma4-e4b-vision-fast",
    batch_size: int = 20,
) -> dict[str, Any]:
    """Run Context Enrichment for Phase 7 using batched subagent calls.

    Instead of per-item NER calls, processes items in batches to reduce
    LLM overhead and provide cross-item entity deduplication.

    Args:
        items: Pipeline items to enrich.
        resolved_identity_map: Identity mapping from Phase 6.
        audio_results: ASR results keyed by audio path.
        video_descriptions: VLM video descriptions keyed by path.
        identity_map: Face cluster identity map.
        model: Model to use.
        batch_size: Items per batch.

    Returns:
        Dict with 'processed_messages', 'entity_extractions',
        'structured_media_items'.
    """
    _ensure_llm_loaded(model)

    processed_messages: list[str] = []
    entity_extractions: list[dict[str, Any]] = []
    structured_media_items: list[dict[str, Any]] = []

    # Build content strings for each item
    item_contents = _build_item_contents(
        items, resolved_identity_map, audio_results,
        video_descriptions, identity_map,
    )

    # Process in batches for entity extraction
    for batch_start in range(0, len(item_contents), batch_size):
        batch = item_contents[batch_start: batch_start + batch_size]
        batch_text = "\n\n---\n\n".join(
            f"ITEM {i}:\n{content}"
            for i, content in enumerate(batch)
        )

        logger.info(
            f"Context Enrichment: batch {batch_start // batch_size + 1}, "
            f"{len(batch)} items, {len(batch_text)} chars"
        )

        agent = create_context_enrichment_agent(model=model)
        result = await agent.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        f"Extract entities from these {len(batch)} items. "
                        f"Return JSON with 'entities' and 'item_entities' keys.\n\n{batch_text}",
                    )
                ]
            }
        )
        batch_result = _parse_batch_entities(result["messages"][-1].content)
        entity_extractions.append(batch_result)

    # Build processed messages from enriched items
    for idx, item in enumerate(items):
        content = item_contents[idx] if idx < len(item_contents) else ""
        if not content:
            continue

        sender = item.get("sender", "Unknown")
        timestamp = item.get("timestamp", "Unknown")
        item_type = item.get("type", "text")
        media_path_rel = item.get("path")

        # Add entities from batch extraction
        batch_idx = idx // batch_size
        item_idx = idx % batch_size
        if batch_idx < len(entity_extractions):
            batch_ents = entity_extractions[batch_idx]
            item_ents = batch_ents.get("item_entities", {}).get(str(item_idx), {})
            people = item_ents.get("people", [])
            if people:
                content += f" [Entities - People: {', '.join(people)}]"

        processed_messages.append(f"**{sender}** [{timestamp}]: {content}")

        # Track structured items
        is_media = (
            item_type in ("voice", "video", "photo")
            or item.get("forwarded")
            or bool(item.get("text"))
        )
        if is_media and media_path_rel:
            s_item: dict[str, Any] = {
                "type": item_type,
                "path": media_path_rel,
                "timestamp": timestamp,
                "sender": sender,
            }
            structured_media_items.append(s_item)

    return {
        "processed_messages": processed_messages,
        "entity_extractions": entity_extractions,
        "structured_media_items": structured_media_items,
    }


def _build_item_contents(
    items: list[dict[str, Any]],
    resolved_identity_map: dict[str, str],
    audio_results: dict[str, list[dict[str, Any]]],
    video_descriptions: dict[str, str],
    identity_map: dict[str, list[str]],
) -> list[str]:
    """Build enriched content strings for each item.

    Applies resolved identities, transcripts, VLM descriptions, and
    face recognition results to produce a single content string per item.
    """
    contents: list[str] = []
    for item in items:
        parts: list[str] = []
        msg_text = item.get("text") or ""
        media_path_rel = item.get("path")
        item_type = item.get("type", "text")

        if msg_text:
            parts.append(msg_text)

        if media_path_rel:
            if item_type == "voice":
                # Find matching audio result
                from pathlib import Path
                media_path = Path(media_path_rel)
                segments = audio_results.get(str(media_path), [])
                if not segments:
                    # Try with different path formats
                    for key in audio_results:
                        if media_path_rel in key or str(media_path) in key:
                            segments = audio_results[key]
                            break

                if segments:
                    # Apply resolved names
                    transcript_parts = []
                    for seg in segments:
                        raw_spk = seg.get("speaker", "")
                        resolved = resolved_identity_map.get(
                            raw_spk,
                            resolved_identity_map.get(seg.get("mf_speaker_id", ""), raw_spk),
                        )
                        name = seg.get("resolved_name", resolved)
                        transcript_parts.append(f"{name}: {seg.get('text', '')}")

                    transcript = " ".join(transcript_parts)
                    parts.append(f"[Voice Message Transcript: {transcript}]")
                    unique_speakers = {
                        resolved_identity_map.get(
                            s.get("speaker", ""),
                            resolved_identity_map.get(s.get("mf_speaker_id", ""), s.get("speaker", "")),
                        )
                        for s in segments
                    }
                    if len(unique_speakers) > 1:
                        parts.append(f"[Speakers: {', '.join(unique_speakers)}]")

            elif item_type == "photo":
                if media_path_rel in identity_map:
                    faces = identity_map[media_path_rel]
                    resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                    parts.append(f"[People in Photo: {', '.join(set(resolved_faces))}]")

            elif item_type == "video":
                if media_path_rel in identity_map:
                    faces = identity_map[media_path_rel]
                    resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                    parts.append(f"[People in Video: {', '.join(set(resolved_faces))}]")
                if media_path_rel in video_descriptions:
                    parts.append(f"[Video Summary: {video_descriptions[media_path_rel]}]")

        contents.append(" ".join(parts))
    return contents


def _parse_batch_entities(response: str) -> dict[str, Any]:
    """Parse batch entity extraction result from LLM response."""
    import json
    import re

    # Try JSON extraction
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", response, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: return empty structure
    return {"entities": {}, "item_entities": {}}
