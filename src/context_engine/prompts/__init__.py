"""Prompts module - LangFuse integration."""

from context_engine.prompts.langfuse import (
    CRITIC_AGENT_PROMPT,
    FLASH_EXPLORER_PROMPT,
    MAIN_AGENT_PROMPT,
    PromptManager,
    get_prompt_manager,
    load_prompts,
)

__all__ = [
    "CRITIC_AGENT_PROMPT",
    "FLASH_EXPLORER_PROMPT",
    "MAIN_AGENT_PROMPT",
    "PromptManager",
    "get_prompt_manager",
    "load_prompts",
]
