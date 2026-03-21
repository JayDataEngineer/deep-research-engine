"""Shared LLM setup for all text/chat extensions.

This module provides a unified interface for getting LLM instances
across all chat-based extensions (companion, roleplay, deep_research, etc.).

All text extensions should import from here rather than directly from
src.shared.factory. This ensures consistent LLM configuration
and allows for centralized chat-specific LLM logic.

All LLM calls are routed through LiteLLM proxy for:
- Unified model routing (configured in litellm/config.yaml)
- Automatic Langfuse tracing
- Rate limiting

Usage:
    from .chat_llm import get_chat_llm

    llm = await get_chat_llm(model="qwen-35-27-fast", enable_thinking=True)
"""
from typing import Optional, TYPE_CHECKING
from .factory import get_llm

if TYPE_CHECKING:
    from .base import BaseLLM


async def get_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list] = None,
) -> "BaseLLM":
    """Get LLM instance for chat/text applications.

    This is a unified entry point for all chat-based extensions.
    All models are routed through LiteLLM proxy.

    Args:
        model: Model name (LiteLLM model ID, e.g., "qwen-35-27-fast", "zai-glm-4-7")
        temperature: Temperature for generation
        enable_thinking: Enable chain-of-thought reasoning
        max_tokens: Optional max tokens override
        provider: DEPRECATED - LiteLLM handles routing now
        session_id: Session ID for Langfuse grouping
        user_id: User ID for Langfuse attribution
        tags: Tags for Langfuse filtering

    Returns:
        BaseLLM instance configured for chat use

    Example:
        # Basic usage
        llm = await get_chat_llm(model="qwen-35-27-fast")

        # With Langfuse tracing
        llm = await get_chat_llm(
            model="zai-glm-4-7",
            session_id="chat-123",
            tags=["companion", "roleplay"]
        )

        # With thinking enabled
        llm = await get_chat_llm(model="qwen-35-27", enable_thinking=True)
    """
    return await get_llm(
        model=model,
        temperature=temperature,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        provider=provider,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
    )


__all__ = ["get_chat_llm"]
