"""LangFuse prompt management.

All prompts are stored in LangFuse - the single source of truth.

Usage:
    from deep_research_engine.prompts import get_prompt, PromptRegistry

    prompt = await get_prompt(PromptRegistry.RESEARCH_PLANNER)
    compiled = prompt.compile(strategy="complexity_based", min_workers=1)
"""

from .client import PromptManager, PromptWrapper, get_langfuse_client, get_prompt
from .registry import PromptRegistry

__all__ = [
    "PromptManager",
    "PromptRegistry",
    "PromptWrapper",
    "get_langfuse_client",
    "get_prompt",
]
