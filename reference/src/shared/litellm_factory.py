"""Simplified LLM Factory - LiteLLM backend only.

This module provides a unified LLM interface through LiteLLM proxy.
All calls are automatically traced to Langfuse via LiteLLM configuration.

LiteLLM provides:
- Single endpoint for all models: http://litellm.local/v1
- Automatic Langfuse tracing (no manual callbacks needed)
- Rate limiting handled by LiteLLM
- MCP tools integration for web scraping/search

Usage:
    from .litellm_factory import get_llm, get_tool_router_llm

    llm = await get_llm(model="qwen-35-27-fast", enable_thinking=True)
    response = await llm.ainvoke([{"role": "user", "content": "Hello"}])

    # With session tracking for Langfuse
    llm = await get_llm(
        model="qwen-35-27-fast",
        session_id="research-123",
        user_id="user-456",
        tags=["deep_research"]
    )
"""

from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base import BaseLLM
from .config import LLMConfig
from .schema import LLMChunk
from .litellm_config import (
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    MODEL_MAPPING,
    THINKING_TO_FAST,
    map_model_name,
)
from .logging_config import get_logger

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "get_llm",
    "get_tool_router_llm",
    "get_main_chat_llm",
    "LiteLLMWrapper",
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
]


def _map_model_name(model: str, enable_thinking: bool = True) -> str:
    """Map old model names to LiteLLM names.

    Args:
        model: Original model name
        enable_thinking: Whether thinking mode is enabled

    Returns:
        LiteLLM-compatible model name
    """
    return map_model_name(model, enable_thinking)


class LiteLLMWrapper(BaseLLM):
    """Wrapper for LiteLLM using LangChain ChatOpenAI.

    This class wraps LiteLLM's OpenAI-compatible endpoint with the
    BaseLLM interface for seamless integration with existing code.

    All calls are automatically traced to Langfuse via LiteLLM's
    built-in Langfuse integration. Use session_id, user_id, and tags
    parameters for session tracking.
    """

    def __init__(
        self,
        config: LLMConfig,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize LiteLLM wrapper.

        Args:
            config: LLMConfig with model settings
            session_id: Optional session ID for Langfuse tracking
            user_id: Optional user ID for Langfuse tracking
            tags: Optional tags for Langfuse tracking
            **kwargs: Additional arguments passed to BaseLLM
        """
        # Initialize base class attributes
        super()._init_config_params(config, **kwargs)

        # LangChain serialization attribute
        self._llm_type = "litellm"

        # Initialize bound tools (for LangChain compatibility)
        self._bound_tools = None
        self._bound_tools_kwargs = {}
        self._middleware = []

        # Validate required attributes
        self._validate_required_attrs()

        # Build metadata for Langfuse session tracking
        extra_body = {}
        if session_id or user_id or tags:
            extra_body["metadata"] = {}
            if session_id:
                extra_body["metadata"]["session_id"] = session_id
            if user_id:
                extra_body["metadata"]["trace_user_id"] = user_id
            if tags:
                extra_body["metadata"]["tags"] = tags

        # Build model kwargs
        model_kwargs = {}
        if extra_body:
            model_kwargs["extra_body"] = extra_body

        # Create LangChain ChatOpenAI client pointing to LiteLLM
        self._llm = ChatOpenAI(
            model=config.model,
            api_key=LITELLM_API_KEY,
            base_url=LITELLM_BASE_URL,
            temperature=config.temperature,
            model_kwargs=model_kwargs,
            streaming=True,
        )

        logger.info(
            f"[LiteLLM] Initialized: model={config.model}, "
            f"base_url={LITELLM_BASE_URL}, temperature={config.temperature}, "
            f"session_id={session_id}, user_id={user_id}, tags={tags}"
        )

    async def astream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> AsyncIterator[LLMChunk]:
        """Stream chat completion using LangChain ChatOpenAI.

        Args:
            messages: Chat messages in dict format
            tools: Optional list of tools for function calling
            **kwargs: Additional arguments

        Yields:
            LLMChunk: Response chunks with content, reasoning, tool_calls
        """
        # Use bound tools if no tools explicitly provided
        if tools is None and self._bound_tools is not None:
            tools = self._bound_tools

        # Convert messages to LangChain format
        lc_messages = self._normalize_messages(messages)

        # Bind tools if provided
        if tools:
            llm_to_stream = self._llm.bind_tools(tools)
        else:
            llm_to_stream = self._llm

        # Stream response
        async for chunk in llm_to_stream.astream(lc_messages):
            # Extract content
            content = ""
            if hasattr(chunk, 'content') and chunk.content:
                content = str(chunk.content)

            # Extract reasoning (if available in additional_kwargs)
            reasoning = ""
            if hasattr(chunk, 'additional_kwargs') and isinstance(chunk.additional_kwargs, dict):
                reasoning = chunk.additional_kwargs.get("reasoning_content", "")

            # Extract tool calls
            tool_calls = []
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                tool_calls = [self._serialize_tool_call(tc) for tc in chunk.tool_calls]

            yield LLMChunk(
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
                is_complete=False,
                additional_kwargs={}
            )

        # Yield final chunk
        yield LLMChunk(
            content="",
            reasoning="",
            tool_calls=[],
            is_complete=True,
            additional_kwargs={}
        )

    async def ainvoke(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> LLMChunk:
        """Invoke chat completion (non-streaming).

        Args:
            messages: Chat messages in dict format
            tools: Optional list of tools for function calling
            **kwargs: Additional arguments

        Returns:
            LLMChunk: Complete response
        """
        # Use bound tools if no tools explicitly provided
        if tools is None and self._bound_tools is not None:
            tools = self._bound_tools

        # Convert messages to LangChain format
        lc_messages = self._normalize_messages(messages)

        # Bind tools if provided
        if tools:
            llm_to_invoke = self._llm.bind_tools(tools)
        else:
            llm_to_invoke = self._llm

        # Invoke
        response = await llm_to_invoke.ainvoke(lc_messages)

        # Extract content
        content = ""
        if hasattr(response, 'content') and response.content:
            content = str(response.content)

        # Extract reasoning
        reasoning = ""
        if hasattr(response, 'additional_kwargs') and isinstance(response.additional_kwargs, dict):
            reasoning = response.additional_kwargs.get("reasoning_content", "")

        # Extract tool calls
        tool_calls = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = [self._serialize_tool_call(tc) for tc in response.tool_calls]

        return LLMChunk(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            is_complete=True,
            additional_kwargs={}
        )

    def _normalize_messages(self, messages: List) -> List:
        """Normalize messages to LangChain format.

        Handles both dict messages and LangChain message objects.

        Args:
            messages: List of dict or LangChain message objects

        Returns:
            List of LangChain message objects
        """
        from langchain_core.messages import ToolMessage

        lc_messages = []
        for msg in messages:
            # If already a LangChain message object, use as-is
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                lc_messages.append(msg)
                continue

            # Otherwise, assume it's a dict and convert
            role = msg.get("role") if isinstance(msg, dict) else None
            content = msg.get("content", "") if isinstance(msg, dict) else ""

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", "")
                ))
            else:
                # Default to user message
                lc_messages.append(HumanMessage(content=content))

        return lc_messages

    def _serialize_tool_call(self, tool_call) -> Dict[str, Any]:
        """Serialize a LangChain tool call to dict.

        Args:
            tool_call: LangChain tool call object

        Returns:
            Dict with id, name, args keys
        """
        return {
            "id": str(getattr(tool_call, 'id', '')),
            "name": str(getattr(tool_call, 'name', '')),
            "args": getattr(tool_call, 'args', {})
        }

    def supports_tool_calling(self) -> bool:
        """LiteLLM supports tool calling via OpenAI-compatible API."""
        return True

    def with_structured_output(self, pydantic_model):
        """Get structured output using JSON schema with strict validation.

        This uses `method="json_schema"` and `strict=True` to force
        llama.cpp (via LiteLLM) to use GBNF grammar constraints at the
        hardware level, ensuring 100% schema compliance for complex schemas.

        This is more reliable than Tool Calling for local models, which
        can struggle with complex nested schemas.

        Args:
            pydantic_model: Pydantic model defining the output schema

        Returns:
            Runnable that returns structured Pydantic model instances

        Reference: https://docs.llamaindex.ai/en/stable/module_guides/structuring/
        """
        return self._llm.with_structured_output(
            pydantic_model,
            method="json_schema",  # Force schema instead of tool calling
            strict=True,            # CRITICAL: triggers llama.cpp GBNF grammar
        )


async def get_llm(
    model: str | None = None,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> BaseLLM:
    """Get LLM instance via LiteLLM proxy.

    All calls are automatically traced to Langfuse via LiteLLM.
    Use session_id, user_id, and tags for session tracking.

    Args:
        model: Model name (uses default from config if None)
        temperature: Temperature for generation
        enable_thinking: Enable chain-of-thought reasoning
        session_id: Optional session ID for Langfuse tracking
        user_id: Optional user ID for Langfuse tracking
        tags: Optional tags for Langfuse tracking
        **kwargs: Additional arguments passed to LiteLLMWrapper

    Returns:
        BaseLLM instance configured for LiteLLM

    Example:
        >>> llm = await get_llm(model="qwen-35-27-fast")
        >>> response = await llm.ainvoke([{"role": "user", "content": "Hello"}])
    """
    from src.config import settings

    if model is None:
        model = settings.llm_model

    # Map to LiteLLM model name
    litellm_model = _map_model_name(model, enable_thinking)

    config = LLMConfig(
        model=litellm_model,
        api_key=LITELLM_API_KEY,
        base_url=LITELLM_BASE_URL,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )

    return LiteLLMWrapper(
        config,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        **kwargs
    )


async def get_tool_router_llm(
    model: str | None = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    """Get LLM for tool routing (non-thinking, faster).

    Tool routing doesn't need thinking mode, so we use the
    fast variant for quicker responses.

    Args:
        model: Model name
        temperature: Sampling temperature
        **kwargs: Additional arguments

    Returns:
        BaseLLM instance with thinking disabled
    """
    return await get_llm(
        model=model,
        temperature=temperature,
        enable_thinking=False,
        **kwargs
    )


async def get_main_chat_llm(
    model: str | None = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    """Get LLM for main chat (thinking enabled).

    Main chat benefits from thinking mode to show reasoning.

    Args:
        model: Model name
        temperature: Sampling temperature
        **kwargs: Additional arguments

    Returns:
        BaseLLM instance with thinking enabled
    """
    return await get_llm(
        model=model,
        temperature=temperature,
        enable_thinking=True,
        **kwargs
    )


__all__ = [
    "get_llm",
    "get_tool_router_llm",
    "get_main_chat_llm",
    "LiteLLMWrapper",
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
]
