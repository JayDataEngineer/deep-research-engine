"""Shared LLM abstraction - LiteLLM backend.

This module provides a unified LLM interface through LiteLLM proxy.
All calls are automatically traced to Langfuse via LiteLLM.

Migration from legacy:
- ChatZAI, StandardChatLLM, LlamaCppLLM are deprecated
- Use get_llm() for all LLM needs
- Session tracking via session_id, user_id, tags params

Usage:
    from src.shared import get_llm, get_tool_router_llm

    llm = await get_llm(model="qwen-35-27-fast", enable_thinking=True)
    response = await llm.ainvoke([{"role": "user", "content": "Hello"}])
"""

# Core factory functions (async)
from .litellm_factory import (
    get_llm,
    get_tool_router_llm,
    get_main_chat_llm,
    LiteLLMWrapper,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
)

# Config classes (kept for compatibility)
from .config import LLMConfig, ToolRouterConfig, MainChatConfig

# Base class (kept for compatibility)
from .base import BaseLLM

# Pydantic models (kept for compatibility)
from .schema import (
    ChatMessage,
    LLMRequest,
    ToolDefinition,
    LLMChunk,
    LLMResponse,
    LLMProviderConfig,
    MessageConverter,
)

# Middleware (kept for compatibility)
from .wrapper import LLMWrapper
from .retry import with_llm_retry, RetryConfig, RetryMiddleware


__all__ = [
    # Core classes
    "BaseLLM",
    "LLMConfig",
    "ToolRouterConfig",
    "MainChatConfig",
    # Factory functions (all async)
    "get_llm",
    "get_tool_router_llm",
    "get_main_chat_llm",
    # LiteLLM wrapper
    "LiteLLMWrapper",
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    # Middleware
    "LLMWrapper",
    "with_llm_retry",
    "RetryConfig",
    "RetryMiddleware",
    # Pydantic models
    "ChatMessage",
    "LLMRequest",
    "ToolDefinition",
    "LLMChunk",
    "LLMResponse",
    "LLMProviderConfig",
    "MessageConverter",
]
