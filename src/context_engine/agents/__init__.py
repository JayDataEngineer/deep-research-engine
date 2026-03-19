"""Agents module - Main, Critic, Flash Explorers."""

__all__ = [
    "create_critic_agent",
    "create_flash_explorer_agent",
    "create_flash_explorer_subagent",
    "create_main_agent",
    "create_web_research_flash_explorer",
    "parse_critic_response",
]


def __getattr__(name: str):
    """Lazy import for store-dependent modules."""
    if name == "create_critic_agent":
        from context_engine.agents.critic import create_critic_agent
        return create_critic_agent

    if name == "parse_critic_response":
        from context_engine.agents.critic import parse_critic_response
        return parse_critic_response

    if name == "create_main_agent":
        from context_engine.agents.main import create_main_agent
        return create_main_agent

    if name in ("create_flash_explorer_agent", "create_flash_explorer_subagent", "create_web_research_flash_explorer"):
        from context_engine.agents.flash_explorer import (
            create_flash_explorer_agent,
            create_flash_explorer_subagent,
            create_web_research_flash_explorer,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
