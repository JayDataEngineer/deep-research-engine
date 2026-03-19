"""Context window management strategies for LLM conversations.

This module provides hot-swappable strategies for managing context window limits.

Strategies:
- basic: Simple FIFO clipping of oldest messages
- sliding_window: Keeps recent window + system prompt
- keep_recent: Keeps N most recent messages
- summarize: Summarizes old messages using LLM

Usage:
    from shared.context_management import wrap_llm_with_context_management

    llm = ChatOpenAI(model="gpt-4")
    managed_llm = wrap_llm_with_context_management(
        llm,
        strategy="sliding_window",
        max_tokens=200000
    )
"""

from .context_strategies import (
    manage_context,
    count_tokens,
    ContextStrategy,
    BASIC_STRATEGY,
    SLIDING_WINDOW_STRATEGY,
    KEEP_RECENT_STRATEGY,
    SUMMARIZE_STRATEGY,
)

from .context import (
    wrap_llm_with_context_management,
    create_context_management_node,
    ContextManagedLLM,
    ContextMiddlewareConfig,
)

__all__ = [
    # Strategies
    "manage_context",
    "count_tokens",
    "ContextStrategy",
    "BASIC_STRATEGY",
    "SLIDING_WINDOW_STRATEGY",
    "KEEP_RECENT_STRATEGY",
    "SUMMARIZE_STRATEGY",
    # Middleware (modern API)
    "wrap_llm_with_context_management",
    "create_context_management_node",
    "ContextManagedLLM",
    "ContextMiddlewareConfig",
]
