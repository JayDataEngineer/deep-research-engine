"""Synthesis Node - uses universal draft_extract pattern.

IMPORTANT: This node does NOT save synthesis output to disk.
Only the FINAL synthesis (after critic approval) is saved in the formatter node.
Intermediate syntheses during iterations are NOT saved.
"""

import time
import re
from typing import Literal, Optional, Any, List, Dict
from pydantic import BaseModel, Field

from ..common import (
    GlobalState,
    ResearchRuntimeConfig,
    get_run_logger_from_config,
    get_yaml_config_from_context,
    create_llm_from_config,
    get_llms_for_node,
    require_run_logger,
    get_prompt_from_config,
)
from ....logging_utils import log_component

from ...decorators import research_node, Phase
from .base import draft_extract

# Import streaming helpers for SSE events
from ...helpers.streaming import (
    get_writer,
    emit_phase_change,
    emit_research_complete,
)

from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import List, Optional as TypingOptional


class Citation(BaseModel):
    """Citation schema for synthesis output."""
    title: str = Field(description="Source title")
    url: str = Field(description="Source URL")
    excerpt: TypingOptional[str] = Field(default=None, description="Relevant excerpt")
    tier: TypingOptional[str] = Field(default=None, description="Source tier (A/B/C)")


class SynthesisOutput(BaseModel):
    """Synthesis output schema with structured fields."""
    title: str = Field(description="Concise title for the synthesis")
    summary: str = Field(description="Executive summary (2-3 sentences)")
    content: str = Field(description="Main synthesis content with sections")
    sections: List[str] = Field(default_factory=list, description="Content sections")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score (0-1)")


def render_template(template: str, config: Dict[str, Any]) -> str:
    """Render template placeholders like {yaml_config.brand_context.identity}.

    Supports nested dict access using dot notation.
    """
    if not template:
        return template

    def get_nested_value(config: Dict[str, Any], path: str) -> str:
        """Get nested value from config using dot-separated path."""
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return f"{{{path}}}"  # Return placeholder if not found
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value) if value is not None else f"{{{path}}}"

    # Match {yaml_config.some.path} pattern
    pattern = r'\{yaml_config\.([^}]+)\}'

    def replace_match(match):
        path = match.group(1)
        return get_nested_value(config, path)

    return re.sub(pattern, replace_match, template)


@research_node(Phase.SYNTHESIS)
async def unified_synthesis_node(
    state: GlobalState,
    writer: Optional[Any] = None,
    *,
    runtime: Optional[Runtime[ResearchRuntimeConfig]] = None,
    store: Optional[Any] = None,
    config: RunnableConfig | None = None,
) -> Command[Literal["critic"]]:
    """Synthesis using universal draft_extract pattern."""
    start_time = time.time()

    yaml_config = get_yaml_config_from_context()

    # Defensive check - yaml_config should always be a dict
    if yaml_config is None or not isinstance(yaml_config, dict):
        run_logger = get_run_logger_from_config(config)
        if run_logger:
            log_component(run_logger, "SYNTHESIS", f"ERROR: yaml_config is {type(yaml_config).__name__}, not dict. This is a context variable bug.")
        # Return early with error
        elapsed = time.time() - start_time
        response = "ERROR: Synthesis configuration unavailable. Please retry the research."
        return Command(
            update={
                "final_response": response,
                "synthesis_elapsed": elapsed,
                "current_phase": "critic",
            },
            goto="critic",
        )

    run_logger = get_run_logger_from_config(config)
    require_run_logger(run_logger, "unified synthesis node")

    # Get stream writer for SSE events (from positional arg or config)
    stream_writer = get_writer(writer, config)

    # Emit phase change event at start
    emit_phase_change(stream_writer, "synthesis", "Synthesis", "Generating final response...", "active")

    query = state.get("query")
    session_id = state.get("session_id")

    architect_findings = state.get("architect_findings")
    worker_outputs = state.get("worker_outputs", [])

    # Log what we have
    log_component(run_logger, "SYNTHESIS", f"architect_findings: {type(architect_findings)}, worker_outputs: {len(worker_outputs)}")

    # Check if we have ACTUAL research content (not just empty objects)
    has_architect_findings = (
        architect_findings is not None and
        hasattr(architect_findings, '__len__') and
        len(architect_findings) > 0
    )
    has_worker_outputs = len(worker_outputs) > 0
    has_research = has_architect_findings or has_worker_outputs

    if not has_research:
        log_component(run_logger, "SYNTHESIS", "WARNING: No research findings available, returning error")
        elapsed = time.time() - start_time
        # Return error response - critic should reject this
        response = f"ERROR: No research findings available for synthesis. The research phase did not complete properly."
        return Command(
            update={
                "final_response": response,
                "synthesis_elapsed": elapsed,
                "current_phase": "critic",
            },
            goto="critic",
        )

    # Get LLMs (main for synthesis, tool for binding)
    main_llm, tool_llm = await get_llms_for_node(yaml_config, main_role="synthesis")

    # Check if we have existing data in context engine
    has_memories = False
    memory_count = 0
    if store and session_id:
        try:
            from src.extensions.content.deep_research.storage import namespace_prefix
            ns_prefix = namespace_prefix(session_id, "memories")
            existing_memories = await store.asearch(ns_prefix)
            memory_count = len(list(existing_memories)) if existing_memories else 0
            has_memories = memory_count > 0
        except Exception:
            pass

    # Check if we have architect_findings from state
    architect_findings = state.get("architect_findings")

    prompts = yaml_config.get("prompts", {}).get("synthesis", {})
    # Handle both dict and string prompts (for flexibility in config.yaml)
    if isinstance(prompts, str):
        role = "Synthesizer"
        output_type = "research report"
        template = None
    else:
        role = prompts.get("synthesis_role", "Synthesizer")
        output_type = prompts.get("output_type", "research report")
        template = prompts.get("template")

    # Use Jinja template from config.yaml instead of hardcoded string
    try:
        system_prompt = get_prompt_from_config(
            yaml_config,
            "synthesis.default",
            query=query,
            session_id=session_id,
            synthesis_role=role,
        )
    except RuntimeError:
        # Fallback if template not found - use minimal hardcoded prompt
        log_component(run_logger, "SYNTHESIS", "WARNING: synthesis.default template not found, using fallback")
        system_prompt = f"""You are the {role}.
Session: {session_id}

YOUR JOB: Use your tools to SEARCH Context Engine for research findings, then synthesize.

CRITICAL INSTRUCTIONS:
1. DO NOT generate placeholder text
2. Use tools to FIND actual research findings
3. Synthesize findings into {output_type} content"""

    # Universal draft_extract pattern
    # Synthesis gets tools when there's data to synthesize (memories or architect_findings)
    synthesis_should_have_tools = has_memories or architect_findings is not None
    result = await draft_extract(
        query=f"Synthesize all research findings for: {query}",
        system_prompt=system_prompt,
        llm=main_llm,
        output_schema=SynthesisOutput,
        extractor_llm=tool_llm,
        has_data=synthesis_should_have_tools,  # Synthesis gets tools to find findings
    )

    response = result["extracted"].content

    # =========================================================================
    # POST-SYNTHESIS FORMATTING ROUTING
    # =========================================================================
    # If output.mode is set (e.g., "twitter", "podcast"), route through
    # content_formatter to transform raw research into formatted output.
    # Otherwise, return raw synthesis content.
    # =========================================================================
    output_config = yaml_config.get("output", {})
    output_mode = output_config.get("mode", "text")

    log_component(run_logger, "SYNTHESIS", f"output_config: {output_config}, output_mode: {output_mode}")

    # Format routing: only format if mode is not "text" (default raw output)
    if output_mode and output_mode != "text":
        log_component(run_logger, "SYNTHESIS", f"Routing to content_formatter with mode: {output_mode}")

        try:
            # Direct import of content_formatter base function
            from src.extensions.content.content_formatter.base import format_content
            from src.constants import CONTENT_FORMATTER_FORMATS_DIR

            # Get brand/character info if available
            brand_context = yaml_config.get("brand_context", {})
            character = brand_context.get("identity") if brand_context else None

            # Format the raw research content
            formatted_result = await format_content(
                format_name=output_mode,
                content=response,
                base_config={},  # Use defaults
                formats_dir=CONTENT_FORMATTER_FORMATS_DIR,
                session_id=session_id,
                character=character,
            )

            # Extract formatted content from Pydantic model
            if hasattr(formatted_result, 'content'):
                response = formatted_result.content
            elif hasattr(formatted_result, 'model_dump'):
                dumped = formatted_result.model_dump()
                # Try common field names
                response = dumped.get('content') or dumped.get('text') or dumped.get('tweets') or str(dumped)
            else:
                response = str(formatted_result)

            log_component(run_logger, "SYNTHESIS", f"Formatted output: {len(response)} chars")
        except Exception as e:
            log_component(run_logger, "SYNTHESIS", f"WARNING: Formatting failed: {e}, returning raw content")
    # =========================================================================

    # Validate response is not placeholder text
    placeholder_phrases = [
        "I'll help you",
        "Synthesizing research",
        "I will search",
        "Let me find",
        "searching for",
        "query the",
    ]
    is_placeholder = any(phrase.lower() in response.lower() for phrase in placeholder_phrases)

    if is_placeholder:
        log_component(run_logger, "SYNTHESIS", "WARNING: Response appears to be placeholder text")
        # Try to get more content from the draft
        draft_content = result.get("draft", "")
        if len(draft_content) > len(response):
            response = draft_content
            log_component(run_logger, "SYNTHESIS", f"Using draft content instead ({len(response)} chars)")

    # Render template with config placeholders
    if template:
        # First render all yaml_config placeholders
        rendered_template = render_template(template, yaml_config)
        # Then replace {findings} with actual content
        if "{findings}" in rendered_template:
            response = rendered_template.replace("{findings}", response)

    elapsed = time.time() - start_time
    log_component(run_logger, "SYNTHESIS", f"Completed in {elapsed:.2f}s, output length: {len(response)} chars")

    # Emit phase change complete event
    emit_phase_change(stream_writer, "synthesis", "Synthesis", "Research complete", "complete")

    # Emit research complete event for frontend
    emit_research_complete(
        writer=stream_writer,
        final_response=response,
        sources=[],  # Sources are embedded in the response
        citations=[]  # Citations are embedded in the response
    )

    # NOTE: Synthesis output is NOT saved here to avoid saving intermediate syntheses.
    # Only the FINAL synthesis (after critic approval) is saved in the formatter node.

    return Command(
        update={
            "final_response": response,
            "synthesis_elapsed": elapsed,
            "current_phase": "critic",
        },
        goto="critic",
    )
