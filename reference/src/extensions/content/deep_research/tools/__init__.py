"""Tools module for deep research.

ARCHITECTURE:
- Tool definitions are in the CENTRAL registry: src.shared.tools.registry
- This module only provides helper functions and artifact tools
- NO duplication of tool definitions

CENTRAL REGISTRY USAGE:
    from src.shared.tools.registry import load_tools, get_tool

    # Load tools by name
    tools = load_tools(["web_search", "web_get"])

    # Get single tool
    tool = get_tool("pdf_deep_agent")

V1 TOOLS STATE INJECTION:
    For V1 tools that require state injection (BaseTool with set_state method),
    use inject_state_to_tools() helper.

GRAPH SEARCH TOOLS:
    For querying the Neo4j knowledge graph built by graph_builder_node:
    - graph_entity_search: Vector search on entities
    - graph_community_search: Search community summaries
    - graph_traverse: Multi-hop graph traversal
    - graph_get_entity_context: Get entity details

THINK TOOL:
    For strategic reflection during research:
    - think_tool: Basic reflection tool
    - think_tool_with_persistence: Persists to Context Engine
    - THINK_TOOL_INSTRUCTIONS: Prompt template for agents
"""

from typing import Dict, Any, List
from langchain_core.tools import BaseTool


def inject_state_to_tools(
    tools: List[BaseTool],
    state: Dict[str, Any]
) -> None:
    """Inject state into V1 tools that support it.

    V1 tools inherit from BaseTool and have a set_state() method.
    V2 tools (with @tool decorator) get state via runtime parameter.

    Args:
        tools: List of tools to inject state into
        state: GlobalState dict containing raw_sources and other state
    """
    for tool in tools:
        if hasattr(tool, 'set_state'):
            tool.set_state(state)


# Re-export graph search tools for convenience
from .graph_search import (
    graph_entity_search,
    graph_community_search,
    graph_traverse,
    graph_get_entity_context,
    GRAPH_SEARCH_TOOLS,
)

# Import think_tool for strategic reflection
from .think_tool import (
    think_tool,
    think_tool_with_persistence,
    create_think_tool,
    THINK_TOOL_INSTRUCTIONS,
)


__all__ = [
    "inject_state_to_tools",
    # Graph search tools
    "graph_entity_search",
    "graph_community_search",
    "graph_traverse",
    "graph_get_entity_context",
    "GRAPH_SEARCH_TOOLS",
    # Think tool
    "think_tool",
    "think_tool_with_persistence",
    "create_think_tool",
    "THINK_TOOL_INSTRUCTIONS",
]
