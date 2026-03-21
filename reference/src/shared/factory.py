"""LLM Factory - LiteLLM backend.

This module provides a unified LLM interface through LiteLLM proxy.
All calls are automatically traced to Langfuse via LiteLLM.

Migration from legacy factory:
- get_llm() -> Same interface, now uses LiteLLM
- get_tool_router_llm() -> Same interface, now uses LiteLLM
- get_chat_zai() -> DEPRECATED, use get_llm() instead

LiteLLM provides:
- Single endpoint for all models: http://litellm.local/v1
- Automatic Langfuse tracing (no manual callbacks needed)
- Rate limiting handled by LiteLLM
- MCP tools integration for web scraping/search

Usage:
    from .factory import get_llm, get_tool_router_llm

    # Basic usage
    llm = await get_llm(model="qwen-35-27-fast")
    response = await llm.ainvoke([{"role": "user", "content": "Hello"}])

    # With session tracking for Langfuse
    llm = await get_llm(
        model="qwen-35-27-fast",
        session_id="research-123",
        user_id="user-456",
        tags=["deep_research"]
    )

Model Names (LiteLLM):
    - qwen-35-27-fast: Local fast model (no thinking)
    - qwen-35-27: Local thinking model
    - zai-glm-4-5: ZAI Cloud GLM-4.5
    - zai-glm-4-7: ZAI Cloud GLM-4.7
    - zai-glm-5: ZAI Cloud GLM-5
"""

# Import from the new LiteLLM factory
from .litellm_factory import (
    get_llm,
    get_tool_router_llm,
    get_main_chat_llm,
    LiteLLMWrapper,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    _map_model_name,
)

# Legacy imports for backward compatibility
from .base import BaseLLM
from .config import LLMConfig, ToolRouterConfig, MainChatConfig

# ChatZAI is removed - use LiteLLMWrapper instead
# For backward compatibility, alias LiteLLMWrapper as ChatZAI
ChatZAI = LiteLLMWrapper


async def get_chat_zai(
    model: str = "glm-4.7",
    temperature: float = 0.7,
    enable_thinking: bool = True,
    **kwargs
) -> BaseLLM:
    """DEPRECATED: Use get_llm() instead.

    This function is kept for backward compatibility but now delegates
    to get_llm() which uses LiteLLM.

    Args:
        model: Model name (will be mapped to LiteLLM name)
        temperature: Sampling temperature
        enable_thinking: Enable thinking mode
        **kwargs: Additional arguments

    Returns:
        BaseLLM instance via LiteLLM
    """
    import warnings
    warnings.warn(
        "get_chat_zai() is deprecated. Use get_llm() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return await get_llm(
        model=model,
        temperature=temperature,
        enable_thinking=enable_thinking,
        **kwargs
    )


__all__ = [
    # Core functions (all async)
    "get_llm",
    "get_tool_router_llm",
    "get_main_chat_llm",
    # Legacy compatibility
    "get_chat_zai",
    # Classes
    "BaseLLM",
    "LiteLLMWrapper",
    "ChatZAI",  # For import compatibility
    # Config
    "LLMConfig",
    "ToolRouterConfig",
    "MainChatConfig",
    # Constants
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
]
