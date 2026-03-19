"""Main Deep Agent - Context Builder.

Full StoreBackend access (read + write + search).
Uses SMART model (Sonnet) via LiteLLM.

Subagents can be delegated to via the task tool:
- Flash Explorers: Quick research on focused topics
- Web Search Specialist: Web search and scraping via MCP
- Document Specialist: Document analysis and summarization
"""

from typing import Any

from langgraph.store.base import BaseStore
from loguru import logger

from context_engine.config import get_model
from context_engine.store.tools import create_search_vectors_tool


def create_main_agent(
    store: BaseStore,
    session_id: str,
    prompts: dict[str, str],
    subagents: list[dict[str, Any]] | None = None,
) -> Any:
    """Create main context builder agent.

    Uses SMART model (Sonnet) via LiteLLM.
    Has full store access (read + write + search).

    Args:
        store: LangGraph BaseStore for persistence
        session_id: Current session ID for namespacing
        prompts: Dict of prompts from PromptManager
        subagents: Optional list of subagent definitions.
                   If None, uses default specialists.

    Returns:
        Compiled StateGraph agent
    """
    # Import deepagents here to avoid import issues if not installed
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import StoreBackend
    except ImportError as e:
        logger.error(f"deepagents not installed: {e}")
        msg = "deepagents package required. Install with: uv add deepagents"
        raise ImportError(msg) from e

    # Get SMART model (Sonnet) via LiteLLM
    model = get_model("smart")

    # Custom tools
    search_vectors = create_search_vectors_tool(store, session_id)

    # Get system prompt
    system_prompt = prompts.get("main_agent", "")

    # Use default specialists if none provided
    if subagents is None:
        subagents = create_default_specialists(store, session_id)

    # Create agent with StoreBackend for full filesystem access
    agent = create_deep_agent(
        model=model,
        tools=[search_vectors],
        system_prompt=system_prompt,
        subagents=subagents,
        store=store,
        backend=lambda rt: StoreBackend(
            rt,
            namespace=lambda ctx: (ctx.runtime.context.get("session_id", session_id), "memories"),
        ),
    )

    logger.info(f"Created main agent for session {session_id} with {len(subagents)} subagents")
    return agent


def create_default_specialists(
    store: BaseStore,
    session_id: str,
) -> list[dict[str, Any]]:
    """Create default specialist subagents for the main agent.

    Includes:
    - Flash Explorer (web research via MCP)
    - Web Search Specialist
    - Document Specialist

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        List of subagent definitions
    """
    from context_engine.agents.flash_explorer import create_web_research_flash_explorer
    from context_engine.specialists.document import create_document_specialist
    from context_engine.specialists.web_search import create_web_search_specialist

    return [
        # Flash Explorer with MCP tools
        create_web_research_flash_explorer(store, session_id),

        # Web Search Specialist (comprehensive web research)
        create_web_search_specialist(store, session_id),

        # Document Specialist (analyze stored documents)
        create_document_specialist(store, session_id),
    ]


def create_minimal_specialists(
    store: BaseStore,
    session_id: str,
) -> list[dict[str, Any]]:
    """Create minimal specialist set - just flash explorers.

    Use this for lighter weight context building without
    full specialist capabilities.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        List with just flash explorer subagent
    """
    from context_engine.agents.flash_explorer import create_flash_explorer_subagent

    return [
        create_flash_explorer_subagent(
            store=store,
            session_id=session_id,
            use_mcp=True,
        ),
    ]


# Legacy compatibility - this function is now in flash_explorer.py
# Keeping here for backward compatibility
def create_flash_explorer_subagent(
    store: BaseStore,
    session_id: str,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Create flash explorer subagent definition.

    .. deprecated::
        Use context_engine.agents.flash_explorer.create_flash_explorer_subagent instead.

    This function is kept for backward compatibility.
    """
    from context_engine.agents.flash_explorer import create_flash_explorer_subagent as _create

    return _create(
        store=store,
        session_id=session_id,
        tools=tools,
        use_mcp=True,
    )
