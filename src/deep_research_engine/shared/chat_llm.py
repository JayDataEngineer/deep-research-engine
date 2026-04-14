"""Shared LLM setup for the Deep Research Engine.

This module provides a unified interface for getting LLM instances.
All calls are routed through LiteLLM proxy for:
- Unified model routing
- Automatic Langfuse tracing
- Rate limiting

Usage:
    from deep_research_engine.shared import get_chat_llm
    llm = await get_chat_llm(model="qwen-35-27", enable_thinking=True)
    response = await llm.ainvoke([{"role": "user", "content": "Hello"}])

    # With structured output
    structured_llm = llm.with_structured_output(MyPydanticModel)
    result = await structured_llm.ainvoke([{"role": "user", "content": "..."}])
"""

from langchain_openai import ChatOpenAI
from loguru import logger

from .config import (
    DEFAULT_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    map_model_name,
)


class ChatLLM:
    """Wrapper around LangChain ChatOpenAI for LiteLLM.

    Provides a simple interface with:
    - LiteLLM proxy integration
    - Structured output support
    - Tool calling support
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ):
        """Initialize the ChatLLM.

        Args:
            model: Model name (LiteLLM model ID)
            temperature: Temperature for generation
            session_id: Optional session ID for Langfuse tracking
            user_id: Optional user ID for Langfuse tracking
            tags: Optional tags for Langfuse tracking
        """
        self.model = model
        self.temperature = temperature

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
            model=model,
            api_key=LITELLM_API_KEY,
            base_url=LITELLM_BASE_URL,
            temperature=temperature,
            model_kwargs=model_kwargs,
            streaming=True,
        )

        logger.debug(
            f"[ChatLLM] Initialized: model={model}, "
            f"base_url={LITELLM_BASE_URL}, temperature={temperature}"
        )

    async def ainvoke(self, messages: list[dict], **kwargs):
        """Invoke chat completion (non-streaming)."""
        return await self._llm.ainvoke(messages, **kwargs)

    async def astream(self, messages: list[dict], **kwargs):
        """Stream chat completion."""
        async for chunk in self._llm.astream(messages, **kwargs):
            yield chunk

    def bind_tools(self, tools: list, **kwargs):
        """Bind tools to this LLM."""
        return self._llm.bind_tools(tools, **kwargs)

    def with_structured_output(self, pydantic_model, **kwargs):
        """Get structured output using JSON schema.

        Uses `method="json_schema"` and `strict=True` to force
        llama.cpp (via LiteLLM) to use GBNF grammar constraints.
        """
        return self._llm.with_structured_output(
            pydantic_model, method="json_schema", strict=True, **kwargs
        )

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying LLM."""
        return getattr(self._llm, name)


async def get_chat_llm(
    model: str | None = None,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list | None = None,
) -> ChatLLM:
    """Get LLM instance for chat applications.

    All models are routed through LiteLLM proxy.

    Args:
        model: Model name (LiteLLM model ID, e.g., "qwen-35-27", "zai-glm-4-7")
        temperature: Temperature for generation
        enable_thinking: Enable chain-of-thought reasoning (maps to fast model if False)
        session_id: Session ID for Langfuse grouping
        user_id: User ID for Langfuse attribution
        tags: Tags for Langfuse filtering

    Returns:
        ChatLLM instance configured for chat use

    Example:
        # Basic usage
        llm = await get_chat_llm(model="qwen-35-27")

        # With Langfuse tracing
        llm = await get_chat_llm(
            model="zai-glm-4-7",
            session_id="chat-123",
            tags=["identity-resolution"],
        )

        # With thinking disabled (faster)
        llm = await get_chat_llm(model="qwen-35-27", enable_thinking=False)
    """
    if model is None:
        model = DEFAULT_MODEL

    # Map to LiteLLM model name
    litellm_model = map_model_name(model, enable_thinking)

    return ChatLLM(
        model=litellm_model,
        temperature=temperature,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
    )


def _resolve_model(model: str | None, config_data: dict | None = None) -> str:
    if model:
        return map_model_name(model)
    if config_data:
        config_models = config_data.get("llm", {}).get("models", {})
        configured = config_models.get("main") or DEFAULT_MODEL
        return map_model_name(configured)
    return map_model_name(DEFAULT_MODEL)


def get_langchain_llm(
    model: str | None = None,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    config_data: dict | None = None,
    streaming: bool = True,
) -> ChatOpenAI:
    """Get a raw LangChain ChatOpenAI instance.

    Routes directly to llama.cpp (bypasses orchestrator) to avoid the
    orchestrator restarting the container with a different model.

    Use this when you need the actual ChatOpenAI (e.g., for deepagents,
    with_structured_output, bind_tools). For general chat, prefer get_chat_llm().
    """
    # Point directly at llama.cpp
    base_url = os.environ.get(
        "LLAMACPP_DIRECT_URL",
        "http://172.17.0.13:8001/v1/chat/completions",
    ).rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]

    api_model = os.environ.get(
        "LLAMACPP_DIRECT_MODEL",
        "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf",
    )

    return ChatOpenAI(
        model=api_model,
        api_key="not-needed",
        base_url=base_url,
        temperature=temperature,
        streaming=streaming,
    )


__all__ = ["ChatLLM", "get_chat_llm"]
