"""Drafter Node - LangGraph node for drafting using subagents.

Spawns flash subagents and specialist sub-sub agents to gather research and write a draft.
For ingestion data (big_text_dump), uses direct multi-pass synthesis instead.
"""

import asyncio
from pathlib import Path
from typing import Any

from deepagents import SubAgent, create_deep_agent

from ..config import get_phase_config
from ..shared import get_langchain_llm
from ..shared.infra_client import InfraClient
from ..state import EngineState
from ..tools.storage import graph_query, initialize_storage_tools, vector_search_datastore
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..workers.base import WorkerRegistry

logger = get_loguru_logger("nodes.drafter")


@node_context("drafter")
async def drafter_node(state: EngineState) -> dict[str, Any]:
    """Drafter node - The 'King' that orchestrates research via 'Flash' subagents."""
    logger.info("Starting King Drafter node (DeepAgents Orchestration)")

    # 1. Get workflow config and check if enabled
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "drafter")
    if not phase_config:
        # Check "boss" for legacy compatibility
        phase_config = get_phase_config(config_data, "boss")

    node_enabled = phase_config.get("enabled", True)

    if not node_enabled:
        logger.info("Drafter node disabled. Skipping.")
        return {"__goto__": "critic"}

    # Reload LLM — may have been unloaded by ASR during ingestion
    king_model_name = phase_config.get("model")
    if not king_model_name:
        king_model_name = "gemma4-e4b-vision-fast"
    try:
        from ..shared.infra_client import InfraClient

        infra = InfraClient()
        infra.load("llm", king_model_name, wait=True)
        logger.info(f"LLM reloaded for drafter: {king_model_name}")
    except Exception as e:
        logger.warning(f"Failed to reload LLM for drafter (may use cloud model): {e}")

    # Initialize storage tools with current config
    initialize_storage_tools(config_data)

    # 2. Map workers to SubAgent specs
    subagents: list[SubAgent] = []

    # Add workers from config
    config_workers = phase_config.get("workers", [])
    if isinstance(config_workers, list):
        for w_cfg in config_workers:
            if not w_cfg.get("enabled", True):
                continue

            w_type = w_cfg.get("type")
            if not w_type:
                continue

            try:
                # Use WorkerRegistry to get worker class and its metadata
                worker_class = WorkerRegistry.get(w_type)

                # Convert our generic worker config to deepagents SubAgent spec.
                # Only include "model" if the config specifies one — otherwise
                # deepagents inherits the king model (BaseChatModel).  Passing
                # model=None causes resolve_model(None) to crash.
                spec: dict[str, Any] = {
                    "name": w_type,
                    "description": f"{worker_class.description}. Task: {w_cfg.get('task')}",
                    "system_prompt": (
                        f"Task: {w_cfg.get('task')}. "
                        f"Only use information provided in the input data. "
                        f"Be thorough but concise."
                    ),
                    # Don't pass worker_class.required_tools here — those are
                    # string names meant for the old BaseWorker.execute() pattern.
                    # DeepAgents will inherit the king's tools instead.
                }
                w_model = w_cfg.get("model")
                if w_model:
                    spec["model"] = w_model
                subagents.append(spec)
            except Exception as e:
                logger.warning(f"Failed to register subagent for worker type '{w_type}': {e}")
                # Fallback to a generic subagent if registry fails.
                # NOTE: omit "model" key so deepagents falls back to the king model.
                # Passing model=None causes resolve_model(None) to crash.
                subagents.append(
                    {
                        "name": w_type,
                        "description": f"Specialist worker for {w_type}. Task: {w_cfg.get('task')}",
                        "system_prompt": f"You are a specialist in {w_type}. Your goal is: {w_cfg.get('task')}. Be thorough but concise.",
                    }
                )

    # Add default social archive specialist if applicable
    if (
        state.get("workflow_name") == "social_archive"
        or state.get("context", {}).get("big_text_dump")
    ) and not any(s["name"] == "social_archive" for s in subagents):
        subagents.append(
            {
                "name": "social_archive",
                "description": "Expert at analyzing social archive data dumps to extract relationship profiles and key memories.",
                "system_prompt": "Analyze the social archive data to extract relationship profiles and key memories. Only state facts supported by the data. Do not invent personality traits or relationships not evidenced in the messages.",
                "model": "qwen35-35b-a3b-vision",
            }
        )

    # 3. Create the King Drafter
    king_prompt = (
        "You are the lead drafter responsible for coordinating subagent research and producing the final report.\n\n"
        "CORE RESPONSIBILITIES:\n"
        "1. DELEGATE: You don't do the primary research yourself. You delegate specific tasks to your Flash Subagents.\n"
        "2. SYNTHESIZE: Once subagents return their findings, integrate them into a coherent, comprehensive report.\n"
        "3. FORMAT: Ensure the output is well-structured markdown with clear headings, citations, and insights.\n\n"
        "CONSTRAINTS:\n"
        "- Only include information from subagent results and provided data\n"
        "- Do not add facts, conclusions, or context not present in the source material\n"
        "- If information is missing or uncertain, say so explicitly\n\n"
        "OUTPUT RULES:\n"
        "- Respond with plain text/markdown only\n"
        "- Do NOT use tool calls for your final response\n"
        "- When generating JSON arguments, use valid JSON syntax (arrays as [...], not strings)\n\n"
        f"Current Goal: {state.get('input_data', {}).get('query', 'Perform comprehensive research and analysis.')}\n"
    )

    # Get model from config or use local LiteLLM
    if king_model_name:
        # Use ChatOpenAI pointing at the orchestrator (avoids provider resolution errors)
        from ..agents.subagents import _make_chat_model

        king_model = _make_chat_model(king_model_name)
    else:
        # Use shared DRY method for LiteLLM-backed ChatOpenAI
        king_model = get_langchain_llm(config_data=config_data)

    # Build user message from ingestion output or original query
    user_query = state.get("input_data", {}).get("query", "Start the research process.")

    # Check for ingestion output (big_text_dump from telegram/pipeline)
    # For ingestion data, use direct synthesis — skip the King agent entirely.
    big_dump_path = state.get("context", {}).get("big_text_dump")
    if big_dump_path:
        dump_file = Path(big_dump_path)
        if dump_file.exists():
            dump_content = dump_file.read_text()
            logger.info(f"Loaded ingestion output: {len(dump_content)} chars from {dump_file.name}")

            # Try sort+extract sub-agents first (uses DeepAgents pattern)
            sorted_content = None
            entities_content = None
            entity_profiles_content = None
            try:
                from ..agents.subagents import (
                    format_entity_profiles,
                    run_entity_finder_batch,
                    run_sort_then_extract,
                )

                sorted_result = await run_sort_then_extract(
                    dump_content,
                    model=king_model_name if king_model_name else "gemma4-e4b-vision-fast",
                    config_data=config_data,
                )
                sorted_content = sorted_result["sorted_content"]
                entities_content = sorted_result["entities"]
                logger.info(
                    f"Sort Agent: {len(sorted_content)} chars, "
                    f"Entity Extractor: {len(entities_content)} chars"
                )
                # Write sorted output for reference
                pipeline_dir = dump_file.parent / "pipeline"
                pipeline_dir.mkdir(parents=True, exist_ok=True)
                sorted_path = pipeline_dir / "sorted_content.md"
                sorted_path.write_text(sorted_content)
                entities_path = pipeline_dir / "extracted_entities.md"
                entities_path.write_text(entities_content)

                # Phase 3: Entity Finder — build per-entity profiles
                if entities_content and sorted_content:
                    try:
                        profiles = await run_entity_finder_batch(
                            entities_text=entities_content,
                            sorted_content=sorted_content,
                            model=king_model_name if king_model_name else "gemma4-e4b-vision-fast",
                            max_entities=10,
                            config_data=config_data,
                        )
                        if profiles:
                            entity_profiles_content = format_entity_profiles(profiles)
                            profiles_path = pipeline_dir / "entity_profiles.md"
                            profiles_path.write_text(entity_profiles_content)
                            logger.info(
                                f"Entity Finder: {len(profiles)} profiles, "
                                f"{len(entity_profiles_content)} chars total"
                            )
                    except Exception as e:
                        logger.warning(f"Entity Finder batch failed (non-fatal): {e}")
            except Exception as e:
                logger.warning(f"Sort/Extract sub-agents failed, using raw dump: {e}")
                logger.exception(e)

            # For ingestion data, use direct synthesis instead of King delegation.
            # The King pattern works for web research but produces raw dumps for
            # pre-existing data because the subagents have nothing to fetch.
            final_message = await _synthesize_ingestion_data(
                infra_client=InfraClient(),
                model=king_model_name,
                dump_content=dump_content,
                sorted_content=sorted_content,
                entities_content=entities_content,
                entity_profiles=entity_profiles_content,
                user_query=user_query,
                dump_file=dump_file,
            )

            king_result = {
                "success": True,
                "output": final_message,
                "output_type": "markdown",
                "worker_id": "king_drafter",
            }
            logger.info("Ingestion synthesis completed.")
            return {"worker_results": [king_result], "__goto__": "critic"}
        else:
            logger.warning(f"Ingestion dump file not found: {big_dump_path}")
            user_message = f"Execute the research plan for: {user_query}"
    else:
        user_message = f"Execute the research plan for: {user_query}"

    # 4. Create and invoke the King (only reached for non-ingestion workflows)
    king = create_deep_agent(
        model=king_model,
        system_prompt=king_prompt,
        subagents=subagents,
        tools=[vector_search_datastore, graph_query],
        debug=True,
    )
    logger.info(f"Invoking King Drafter with {len(subagents)} subagents...")

    result = await king.ainvoke({"messages": [("user", user_message)]})

    # Extract the final response from the King
    final_message = result["messages"][-1].content

    # Create result for accumulation
    king_result = {
        "success": True,
        "output": final_message,
        "output_type": "markdown",
        "worker_id": "king_drafter",
    }

    logger.info("King Drafter has completed the orchestration.")
    return {"worker_results": [king_result], "__goto__": "critic"}


async def _synthesize_ingestion_data(
    infra_client: InfraClient,
    model: str,
    dump_content: str,
    sorted_content: str | None,
    entities_content: str | None,
    user_query: str,
    dump_file: Path,
    entity_profiles: str | None = None,
) -> str:
    """Synthesize ingestion data into a research report using chunked summarization.

    For large ingestion dumps, does multi-pass summarization:
    1. Split content into model-context-sized chunks
    2. Summarize each chunk
    3. Combine summaries into final synthesis

    Entity profiles (if available) are included as structured context to help
    the synthesis LLM produce entity-rich output.
    """
    # Use sorted content if available (more organized), else raw dump
    analysis_content = sorted_content or dump_content

    # Build context header with entities and profiles
    context_parts: list[str] = []
    if entities_content:
        context_parts.append(f"## EXTRACTED ENTITIES\n\n{entities_content}")
    if entity_profiles:
        context_parts.append(f"## ENTITY PROFILES\n\n{entity_profiles}")
    if context_parts:
        header = "\n\n".join(context_parts)
        analysis_content = f"{header}\n\n## CONTENT\n\n{analysis_content}"

    # Chunk size: 100K chars (~25K tokens) leaves room for system prompt + output in 72K context
    chunk_size = 100_000
    max_chars = 100_000

    if len(analysis_content) <= max_chars:
        # Small enough for single-pass synthesis
        synthesis_prompt = (
            "Synthesize the following ingested data into a "
            "comprehensive, well-structured research report.\n\n"
            "REQUIREMENTS:\n"
            "1. Create clear sections with headers (##)\n"
            "2. Summarize key themes, relationships, and patterns\n"
            "3. Preserve important details and specific facts from the data\n"
            "4. Include a summary at the top\n"
            "5. Use markdown formatting\n"
            "6. Do not add information not present in the data\n\n"
            f"RESEARCH GOAL: {user_query}\n\n"
            f"DATA:\n{analysis_content}"
        )

        response = await asyncio.to_thread(
            infra_client.chat,
            messages=synthesis_prompt,
            model=model,
            max_tokens=16000,
        )
        return _extract_text_from_chat(response)

    # Multi-pass: summarize chunks, then synthesize summaries
    logger.info(f"Large content ({len(analysis_content)} chars) — using multi-pass synthesis")

    chunks = _split_at_paragraphs(analysis_content, chunk_size)
    logger.info(f"Split into {len(chunks)} chunks for parallel summarization")

    summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        summary_prompt = (
            "Summarize the following content, preserving all key facts, names, dates, "
            "relationships, and important details. Use markdown formatting.\n\n"
            f"CONTENT (part {i + 1} of {len(chunks)}):\n{chunk}"
        )
        try:
            response = await asyncio.to_thread(
                infra_client.chat,
                messages=summary_prompt,
                model=model,
                max_tokens=8000,
            )
            summaries.append(_extract_text_from_chat(response))
        except Exception as e:
            logger.warning(f"Chunk {i + 1} summarization failed: {e}")
            # Include truncated original as fallback
            summaries.append(
                f"[Part {i + 1} summarization failed, first 5000 chars]:\n{chunk[:5000]}"
            )

    # Final synthesis pass
    combined_summaries = "\n\n---\n\n".join(
        f"## Part {i + 1} Summary\n{summary}" for i, summary in enumerate(summaries)
    )

    # Write intermediate summaries for debugging
    pipeline_dir = dump_file.parent / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = pipeline_dir / "chunk_summaries.md"
    summaries_path.write_text(combined_summaries)

    final_prompt = (
        "You are a research analyst. Combine these partial summaries into a single, "
        "comprehensive research report.\n\n"
        "REQUIREMENTS:\n"
        "1. Merge overlapping information\n"
        "2. Create a logical structure with clear sections\n"
        "3. Include an executive summary at the top\n"
        "4. Preserve specific facts, names, and details\n"
        "5. Use proper markdown formatting\n\n"
        f"RESEARCH GOAL: {user_query}\n\n"
        f"PARTIAL SUMMARIES:\n{combined_summaries}"
    )

    response = await asyncio.to_thread(
        infra_client.chat,
        messages=final_prompt,
        model=model,
        max_tokens=16000,
    )
    return _extract_text_from_chat(response)


def _split_at_paragraphs(text: str, max_chunk_size: int) -> list[str]:
    """Split text at paragraph boundaries to avoid cutting mid-sentence."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chunk_size, len(text))
        if end < len(text):
            # Find nearest paragraph break
            para_break = text.rfind("\n\n", pos, end)
            if para_break > pos + max_chunk_size // 2:
                end = para_break + 2
            else:
                # Fall back to newline
                nl_break = text.rfind("\n", pos, end)
                if nl_break > pos + max_chunk_size // 2:
                    end = nl_break + 1
        chunks.append(text[pos:end])
        pos = end
    return chunks


def _extract_text_from_chat(response: dict) -> str:
    """Extract text content from InfraClient chat response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return str(response)
