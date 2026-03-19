"""Universal Draft + Extract pattern for ALL inference nodes.

Boss, Critic, Synthesis all use this same pattern:
1. DRAFT: LLM with optional tools - can search memories/graph if data exists
2. EXTRACT: Simple LLM with structured_output - easy formatting

IMPORTANT: We use bind_tools() directly instead of create_deep_agent() because
deepagents adds built-in filesystem tools (ls, read_file, glob, grep, etc.) that
we don't want. Our INFERENCE_TOOLS are memory/graph search tools only.

The prompt should tell the model IF data exists, and the model decides whether
to use tools. Don't force tool usage in the prompt.
"""

from typing import Type, Any, Dict, List, Optional
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage

# Import Context Engine memory search (LangGraph Store)
from src.extensions.features.tools.context_engine.tools import search_memories

# Import Graph Search tools for knowledge graph queries
from src.extensions.content.deep_research.tools.graph_search import (
    graph_entity_search,
    graph_community_search,
    graph_traverse,
    graph_get_entity_context,
)

# Universal inference tools for ALL nodes
# - search_memories: Semantic search over agent artifacts
# - graph_entity_search: Vector search on knowledge graph entities
# - graph_community_search: Search community summaries (high-level topics)
# - graph_traverse: Multi-hop graph traversal
# - graph_get_entity_context: Get entity details and relationships
INFERENCE_TOOLS = [
    search_memories,
    graph_entity_search,
    graph_community_search,
    graph_traverse,
    graph_get_entity_context,
]


async def draft_extract(
    query: str,
    system_prompt: str,
    llm: Any,
    output_schema: Type[BaseModel],
    extractor_llm: Any = None,
    middleware: Optional[List] = None,  # Ignored - kept for API compatibility
    tools: Optional[List] = None,  # Optional: custom tools (default: INFERENCE_TOOLS)
    max_steps: int = 25,  # Max tool call iterations
    has_data: bool = True,  # Whether data exists (affects tool binding)
) -> Dict[str, Any]:
    """Universal Draft + Extract pattern with optional inference tools.

    Used by: Boss, Critic, Synthesis - ALL inference nodes.

    Pattern:
    - DRAFT: LLM with tools (if has_data) → raw thinking output
    - EXTRACT: Simple LLM (structured_output) → validated data

    Tools are bound directly to the LLM using bind_tools(), NOT via deep agents.
    This gives us our memory/graph search tools without the deepagents filesystem
    tools (ls, read_file, glob, grep, etc.) that cause runaway tool calls.

    Args:
        query: The task/query - contains the original human query
        system_prompt: System instructions for the agent
        llm: LLM instance for drafting
        output_schema: Pydantic schema for extraction
        extractor_llm: LLM instance for extraction (default: same as llm)
        middleware: Ignored - kept for API compatibility
        tools: Custom tools (default: INFERENCE_TOOLS if has_data else no tools)
        max_steps: Max iterations for tool calls (prevents runaway)
        has_data: Whether memories/data exist. If False, no tools are bound.

    Returns:
        Dict with 'draft' (raw content) and 'extracted' (validated Pydantic model)
    """
    extractor_llm = extractor_llm or llm

    # ═══════════════════════════════════════════════════════════════
    # DRAFT: LLM with optional tools
    # Only bind tools if data exists - otherwise the model wastes time searching
    # ═══════════════════════════════════════════════════════════════

    # Determine which tools to use
    if tools is not None:
        # Custom tools provided
        agent_tools = tools
    elif has_data:
        # Data exists - provide inference tools so model can search
        agent_tools = INFERENCE_TOOLS
    else:
        # No data - don't give tools, model will just reason
        agent_tools = []

    # Bind tools to LLM if we have any
    if agent_tools:
        draft_llm = llm.bind_tools(agent_tools)
    else:
        draft_llm = llm

    # Simple LLM call - model decides whether to use tools
    draft_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    draft_response = await draft_llm.ainvoke(draft_messages)

    # Handle tool calls if the model made any
    # Execute tools in a loop until done or max_steps reached
    tool_call_count = 0
    current_messages = draft_messages[:]

    while (
        hasattr(draft_response, 'tool_calls') and
        draft_response.tool_calls and
        tool_call_count < max_steps
    ):
        tool_call_count += 1

        # Add the assistant message with tool calls
        current_messages.append(draft_response)

        # Execute each tool call
        from langchain_core.messages import ToolMessage
        for tool_call in draft_response.tool_calls:
            try:
                # Find the tool by name
                tool = None
                for t in agent_tools:
                    if hasattr(t, 'name') and t.name == tool_call['name']:
                        tool = t
                        break
                    elif callable(t) and hasattr(t, '__name__') and t.__name__ == tool_call['name']:
                        tool = t
                        break

                if tool:
                    # Execute the tool
                    tool_input = tool_call.get('args', {})
                    tool_result = await tool.ainvoke(tool_input)

                    # Create tool message
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call['id']
                    )
                    current_messages.append(tool_message)
                else:
                    # Tool not found - return error
                    tool_message = ToolMessage(
                        content=f"Error: Tool '{tool_call['name']}' not found",
                        tool_call_id=tool_call['id']
                    )
                    current_messages.append(tool_message)
            except Exception as e:
                # Tool execution failed
                tool_message = ToolMessage(
                    content=f"Error executing tool: {str(e)}",
                    tool_call_id=tool_call['id']
                )
                current_messages.append(tool_message)

        # Get next response from LLM with tool results
        draft_response = await draft_llm.ainvoke(current_messages)

    # Extract content from response
    if hasattr(draft_response, 'content'):
        draft_content = str(draft_response.content)
    else:
        draft_content = str(draft_response)

    # ═══════════════════════════════════════════════════════════════
    # EXTRACT: Simple LLM with structured_output
    # Extraction is EASY - just format the draft into schema
    # ═══════════════════════════════════════════════════════════════

    extractor = extractor_llm.with_structured_output(output_schema)

    # Combine into single SystemMessage - draft is system-generated, NOT human
    combined_extraction = f"Extract structured data from this content.\n\n---\n\n{draft_content}"
    extracted = await extractor.ainvoke([
        SystemMessage(content=combined_extraction)
    ])

    return {
        "draft": draft_content,
        "extracted": extracted,
    }
