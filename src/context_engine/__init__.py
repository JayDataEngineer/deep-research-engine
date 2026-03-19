"""Context Engine - Minimal context building with deepagents + LangGraph Store."""

__version__ = "0.2.0"

from context_engine.state import ContextEngineState, create_initial_state

# Lazy imports for store-dependent modules
# These require psycopg/libpq which may not be available
__all__ = [
    # State (always available)
    "ContextEngineState",
    "create_initial_state",
    # Agents (lazy - require deepagents + store)
    "create_critic_agent",
    "create_flash_explorer_agent",
    "create_flash_explorer_subagent",
    "create_main_agent",
    "create_web_research_flash_explorer",
    "parse_critic_response",
    # Pipeline (lazy - requires store)
    "build_context",
    "run_pipeline",
    # Specialists (lazy)
    "create_all_mcp_tools",
    "create_document_specialist",
    "create_mcp_crawl_tool",
    "create_mcp_scrape_tool",
    "create_mcp_web_search_tool",
    "create_web_search_specialist",
    # Prompts (always available)
    "CRITIC_AGENT_PROMPT",
    "FLASH_EXPLORER_PROMPT",
    "MAIN_AGENT_PROMPT",
    "PromptManager",
    "get_prompt_manager",
    "load_prompts",
]


def __getattr__(name: str):
    """Lazy import for store-dependent modules."""
    if name in ("create_critic_agent", "create_main_agent",
                "create_flash_explorer_agent", "create_flash_explorer_subagent",
                "create_web_research_flash_explorer", "parse_critic_response"):
        from context_engine.agents import (
            create_critic_agent,
            create_flash_explorer_agent,
            create_flash_explorer_subagent,
            create_main_agent,
            create_web_research_flash_explorer,
            parse_critic_response,
        )
        return locals()[name]

    if name in ("build_context", "run_pipeline"):
        from context_engine.pipeline import build_context, run_pipeline
        return locals()[name]

    if name in ("create_all_mcp_tools", "create_mcp_crawl_tool",
                "create_mcp_scrape_tool", "create_mcp_web_search_tool",
                "create_web_search_specialist", "create_document_specialist"):
        from context_engine.specialists import (
            create_all_mcp_tools,
            create_document_specialist,
            create_mcp_crawl_tool,
            create_mcp_scrape_tool,
            create_mcp_web_search_tool,
            create_web_search_specialist,
        )
        return locals()[name]

    if name in ("CRITIC_AGENT_PROMPT", "FLASH_EXPLORER_PROMPT", "MAIN_AGENT_PROMPT",
                "PromptManager", "get_prompt_manager", "load_prompts"):
        from context_engine.prompts import (
            CRITIC_AGENT_PROMPT,
            FLASH_EXPLORER_PROMPT,
            MAIN_AGENT_PROMPT,
            PromptManager,
            get_prompt_manager,
            load_prompts,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
