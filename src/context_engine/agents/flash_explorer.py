"""Flash Explorer Agent - Fast research specialist.

Uses FAST model (Haiku) for quick, focused research tasks.
Can be spawned by Main Agent via task tool for parallel context gathering.

MCP Tools Integration:
- Uses LiteLLM's MCP tool support for web search, scraping, etc.
- Tools: search_web, scrape_url, map_domain, crawl_site, scrape_structured
"""

from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger

from context_engine.config import get_model
from context_engine.store.tools import create_search_vectors_tool


# =============================================================================
# MCP TOOL WRAPPERS (via LiteLLM)
# =============================================================================

def create_mcp_web_search_tool() -> Any:
    """Create web search tool using LiteLLM's MCP integration.

    This tool is called through LiteLLM which routes to webtools-mcp server.

    Returns:
        Tool definition for MCP web search
    """
    @tool
    async def search_web(query: str) -> str:
        """Search the web for current information.

        Uses webtools-mcp server through LiteLLM for web search with reranking.

        Args:
            query: Search query

        Returns:
            Search results with snippets and URLs

        Note:
            This tool requires LiteLLM to be configured with webtools_server MCP.
            See: http://webtools.local/mcp
        """
        # This is a placeholder - the actual tool call happens through LiteLLM's
        # MCP routing. When deepagents sees this tool name, it will route to MCP.
        # The tool definition tells the LLM about the capability.
        return f"[MCP Tool: search_web] Query: {query}"

    return search_web


def create_mcp_scrape_tool() -> Any:
    """Create URL scraping tool using LiteLLM's MCP integration.

    Returns:
        Tool definition for MCP URL scraping
    """
    @tool
    async def scrape_url(url: str) -> str:
        """Scrape and extract content from a URL.

        Uses webtools-mcp server through LiteLLM.

        Args:
            url: URL to scrape

        Returns:
            Extracted content from the page
        """
        return f"[MCP Tool: scrape_url] URL: {url}"

    return scrape_url


def create_mcp_tools() -> list[Any]:
    """Create all MCP tools for flash explorers.

    Returns:
        List of MCP tool wrappers
    """
    return [
        create_mcp_web_search_tool(),
        create_mcp_scrape_tool(),
    ]


# =============================================================================
# FLASH EXPLORER AGENT
# =============================================================================

def create_flash_explorer_agent(
    store: BaseStore,
    session_id: str,
    tools: list[Any] | None = None,
    system_prompt: str | None = None,
    use_mcp: bool = True,
) -> Any:
    """Create a flash explorer agent for quick research tasks.

    Uses FAST model (Haiku) via LiteLLM.
    Designed for parallel execution on focused topics.

    Args:
        store: LangGraph BaseStore for context access
        session_id: Current session ID
        tools: Optional additional tools
        system_prompt: Optional custom system prompt
        use_mcp: Whether to include MCP tools (default: True)

    Returns:
        Compiled StateGraph agent
    """
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import StoreBackend
    except ImportError as e:
        logger.error(f"deepagents not installed: {e}")
        msg = "deepagents package required. Install with: uv add deepagents"
        raise ImportError(msg) from e

    # Get FAST model (Haiku)
    model = get_model("fast")

    # Core tools
    search_vectors = create_search_vectors_tool(store, session_id)

    # Build tool list
    agent_tools = [search_vectors]

    # Add MCP tools if enabled
    if use_mcp:
        agent_tools.extend(create_mcp_tools())

    # Add any additional tools
    agent_tools.extend(tools or [])

    # Default prompt if none provided
    if system_prompt is None:
        from context_engine.prompts.langfuse import FLASH_EXPLORER_PROMPT
        system_prompt = FLASH_EXPLORER_PROMPT

    # Create agent with StoreBackend for read/write access
    agent = create_deep_agent(
        model=model,
        tools=agent_tools,
        system_prompt=system_prompt,
        subagents=[],  # Flash explorers don't spawn more subagents
        store=store,
        backend=lambda rt: StoreBackend(
            rt,
            namespace=lambda ctx: (ctx.runtime.context.get("session_id", session_id), "memories", "insights"),
        ),
    )

    logger.debug(f"Created flash explorer agent for session {session_id}")
    return agent


def create_flash_explorer_subagent(
    store: BaseStore,
    session_id: str,
    tools: list[Any] | None = None,
    description: str | None = None,
    use_mcp: bool = True,
) -> dict[str, Any]:
    """Create flash explorer subagent definition for delegation.

    This returns a subagent definition dict that can be passed
    to the Main Agent's subagents parameter.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID
        tools: Optional additional tools
        description: Optional custom description for the subagent
        use_mcp: Whether to include MCP tools (default: True)

    Returns:
        Subagent definition dict
    """
    from context_engine.prompts.langfuse import FLASH_EXPLORER_PROMPT

    # Get FAST model (Haiku)
    model = get_model("fast")

    # Core tools
    search_vectors = create_search_vectors_tool(store, session_id)

    # Build tool list
    agent_tools = [search_vectors]

    # Add MCP tools if enabled
    if use_mcp:
        agent_tools.extend(create_mcp_tools())

    # Add any additional tools
    agent_tools.extend(tools or [])

    # Default description for task delegation
    if description is None:
        description = (
            "Quick research on a specific topic. Can search web and scrape URLs. "
            "Give ONE focused topic. Returns findings with sources."
        )

    return {
        "name": "flash-explorer",
        "description": description,
        "system_prompt": FLASH_EXPLORER_PROMPT,
        "model": model,
        "tools": agent_tools,
    }


# Convenience function for common use case
def create_web_research_flash_explorer(
    store: BaseStore,
    session_id: str,
) -> dict[str, Any]:
    """Create flash explorer with web research capabilities.

    Includes MCP tools for web search and URL scraping.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        Subagent definition dict with MCP tools
    """
    return create_flash_explorer_subagent(
        store=store,
        session_id=session_id,
        use_mcp=True,
        description=(
            "Quick web research on a specific topic. "
            "Can search the web (search_web) and scrape URLs (scrape_url). "
            "Give ONE focused topic. Returns findings with sources."
        ),
    )
