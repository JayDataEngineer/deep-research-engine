"""Modern context management for LangChain/LangGraph applications.

This module provides two approaches for context management:
1. LLM Wrapper - Wraps any LLM to apply context management before each call
2. Graph Node - A LangGraph node that applies context management to state

The old AgentMiddleware pattern is deprecated and no longer supported in modern LangChain/LangGraph.

COMPLIANCE: Uses Pydantic models for all configuration and validation.
"""

import os
from functools import wraps
from typing import Callable, Literal, Optional, Any, List
from pydantic import BaseModel, Field, field_validator

from .context_strategies import manage_context, count_tokens, ContextStrategy


class ContextMiddlewareConfig(BaseModel):
    """Configuration for context management middleware.

    COMPLIANCE: Pydantic model for validation.
    """
    strategy: ContextStrategy = Field(default=ContextStrategy.BASIC, description="Context management strategy")
    max_tokens: int = Field(default=200000, ge=1, description="Model's context window (glm-4.7 default)")
    output_buffer: int = Field(default=20000, ge=0, description="Tokens to reserve for output")
    verbose: bool = Field(default=False, description="Log when context is clipped")


def _convert_messages_to_dict(messages: list) -> list[dict]:
    """Convert LangChain Message objects to dict format.

    Args:
        messages: List of LangChain Message objects (HumanMessage, AIMessage, etc.)

    Returns:
        List of dict messages with 'role' and 'content' keys
    """
    converted = []
    for msg in messages:
        # Handle both dict and Message objects
        if isinstance(msg, dict):
            converted.append(msg)
        elif hasattr(msg, 'type'):
            # LangChain Message object
            msg_type = msg.type
            content = msg.content if hasattr(msg, 'content') else str(msg)

            # Map LangChain types to OpenAI format
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "tool": "tool",
                "function": "tool",
            }
            role = role_map.get(msg_type, msg_type)
            converted.append({"role": role, "content": content})
        else:
            # Fallback
            converted.append({"role": "user", "content": str(msg)})
    return converted


def _reconstruct_messages_from_dict(messages: list[dict]) -> list:
    """Reconstruct LangChain Message objects from dict format.

    Args:
        messages: List of dict messages with 'role' and 'content' keys

    Returns:
        List of LangChain Message objects
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

    reconstructed = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            reconstructed.append(HumanMessage(content=content))
        elif role == "assistant":
            reconstructed.append(AIMessage(content=content))
        elif role == "system":
            reconstructed.append(SystemMessage(content=content))
        elif role == "tool":
            reconstructed.append(ToolMessage(content=content))
        else:
            # Fallback
            reconstructed.append(HumanMessage(content=content))

    return reconstructed


class ContextManagedLLM:
    """Wrapper that applies context management to LLM calls.

    This wrapper intercepts LLM invoke/stream calls and applies context management
    to ensure messages fit within token limits.

    COMPLIANCE: Works with BaseLLM interface from shared factory.

    Usage:
        from src.shared import get_llm
        from src.extensions.libraries.core_chat_engine.memory.context import wrap_llm_with_context_management

        base_llm = get_llm()  # Returns BaseLLM
        managed_llm = wrap_llm_with_context_management(
            base_llm,
            strategy="sliding_window",
            max_tokens=200000,
            output_buffer=20000
        )
        # Use managed_llm just like the original llm
        response = await managed_llm.ainvoke(messages)
    """

    def __init__(
        self,
        llm: Any,
        strategy: str | ContextStrategy = ContextStrategy.BASIC,
        max_tokens: int = 200000,
        output_buffer: int = 20000,
        verbose: bool = False,
    ):
        """Initialize the context-managed LLM.

        Args:
            llm: The BaseLLM instance to wrap (from get_llm())
            strategy: Context management strategy
            max_tokens: Model's context window
            output_buffer: Tokens to reserve for output
            verbose: Log when context is clipped
        """
        self.llm = llm
        self.strategy = ContextStrategy(strategy) if isinstance(strategy, str) else strategy
        self.max_tokens = max_tokens
        self.output_buffer = output_buffer
        self.verbose = verbose

        # Allow env var override
        env_strategy = os.getenv("CONTEXT_STRATEGY")
        if env_strategy:
            self.strategy = ContextStrategy(env_strategy.lower())

    def _apply_context_management(self, messages: list) -> tuple[list, int, int]:
        """Apply context management to messages.

        Returns:
            Tuple of (managed_messages, original_tokens, managed_tokens)
        """
        messages_dict = _convert_messages_to_dict(messages)
        original_tokens = count_tokens(messages_dict)
        safe_limit = self.max_tokens - self.output_buffer

        if original_tokens > safe_limit:
            context_kwargs = {}
            if self.strategy == ContextStrategy.SUMMARIZE:
                # For summarization, we need an LLM - use the wrapped one
                context_kwargs["llm"] = self.llm
                if self.verbose:
                    def _summary_callback(summary: str, orig_tok: int, sum_tok: int, num_msgs: int, dur: float) -> None:
                        import sys
                        print(f"[Context Management] Summarization in {dur:.2f}s", file=sys.stderr)
                        print(f"[Context Management] Summarized {num_msgs} messages ({orig_tok} → {sum_tok} tokens)", file=sys.stderr)
                        print(f"[Context Management] Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}", file=sys.stderr)
                    context_kwargs["summary_log_callback"] = _summary_callback

            managed_messages = manage_context(
                messages_dict,
                max_tokens=self.max_tokens,
                output_buffer=self.output_buffer,
                strategy=self.strategy,
                **context_kwargs
            )

            managed_tokens = count_tokens(managed_messages)

            if self.verbose and self.strategy != ContextStrategy.SUMMARIZE:
                import sys
                print(f"[Context Management] Clipped: {original_tokens} → {managed_tokens} tokens ({len(messages)} → {len(managed_messages)} messages)", file=sys.stderr)

            # Reconstruct LangChain messages
            return _reconstruct_messages_from_dict(managed_messages), original_tokens, managed_tokens

        return messages, original_tokens, original_tokens

    def invoke(self, messages: List, **kwargs: Any) -> Any:
        """Invoke the LLM with context management applied.

        Args:
            messages: List of messages (LangChain format or dict format)
            **kwargs: Additional arguments to pass to LLM

        Returns:
            LLM response with context management applied

        The context management is TRANSIENT - it only affects the current LLM call
        and does NOT modify the original messages list.
        """
        managed_messages, orig_tok, managed_tok = self._apply_context_management(messages)
        return self.llm.invoke(managed_messages, **kwargs)

    async def ainvoke(self, messages: List, **kwargs: Any) -> Any:
        """Async invoke the LLM with context management applied.

        Args:
            messages: List of messages (LangChain format or dict format)
            **kwargs: Additional arguments to pass to LLM

        Returns:
            LLM response with context management applied
        """
        managed_messages, orig_tok, managed_tok = self._apply_context_management(messages)
        return await self.llm.ainvoke(managed_messages, **kwargs)

    def stream(self, messages: List, **kwargs: Any) -> Any:
        """Stream the LLM with context management applied.

        Args:
            messages: List of messages (LangChain format or dict format)
            **kwargs: Additional arguments to pass to LLM

        Returns:
            Streaming response with context management applied
        """
        managed_messages, orig_tok, managed_tok = self._apply_context_management(messages)
        return self.llm.stream(managed_messages, **kwargs)

    async def astream(self, messages: List, **kwargs: Any) -> Any:
        """Async stream the LLM with context management applied.

        Args:
            messages: List of messages (LangChain format or dict format)
            **kwargs: Additional arguments to pass to LLM

        Returns:
            Async streaming response with context management applied
        """
        managed_messages, orig_tok, managed_tok = self._apply_context_management(messages)
        return self.llm.astream(managed_messages, **kwargs)

    def bind_tools(self, tools: List, **kwargs: Any) -> Any:
        """Proxy bind_tools to the wrapped LLM.

        Args:
            tools: List of tools to bind
            **kwargs: Additional arguments

        Returns:
            Self with tools bound (or proxy to wrapped LLM's bind_tools)
        """
        return self.llm.bind_tools(tools, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Proxy with_structured_output to the wrapped LLM.

        Args:
            schema: Output schema
            **kwargs: Additional arguments

        Returns:
            Structured output LLM (or proxy to wrapped LLM)
        """
        return self.llm.with_structured_output(schema, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attributes to the wrapped LLM."""
        return getattr(self.llm, name)


def wrap_llm_with_context_management(
    llm: Any,
    strategy: str | ContextStrategy = ContextStrategy.BASIC,
    max_tokens: int = 200000,
    output_buffer: int = 20000,
    verbose: bool = False,
) -> ContextManagedLLM:
    """Wrap an LLM with context management.

    This is the recommended modern pattern for applying context management
    to LLM calls in LangChain/LangGraph applications.

    COMPLIANCE: Works with BaseLLM interface from shared.llm factory.

    Args:
        llm: The BaseLLM instance to wrap (obtained from get_llm())
        strategy: Context management strategy ('basic', 'sliding_window', 'keep_recent', 'summarize')
        max_tokens: Model's context window (default: 200k for glm-4.7)
        output_buffer: Tokens to reserve for LLM output (default: 20k)
        verbose: Log when context is clipped/summarized (default: False)

    Returns:
        A ContextManagedLLM wrapper that behaves like the original LLM

    Example:
        from src.shared import get_llm
        from src.extensions.libraries.core_chat_engine.memory.context import wrap_llm_with_context_management

        base_llm = get_llm()  # Returns BaseLLM
        llm = wrap_llm_with_context_management(
            base_llm,
            strategy="sliding_window",
            max_tokens=200000,
            output_buffer=20000,
            verbose=True
        )

        # Use llm normally - context management is applied automatically
        response = await llm.ainvoke(messages)
    """
    return ContextManagedLLM(
        llm=llm,
        strategy=strategy,
        max_tokens=max_tokens,
        output_buffer=output_buffer,
        verbose=verbose,
    )


def create_context_management_node(
    strategy: str | ContextStrategy = ContextStrategy.BASIC,
    max_tokens: int = 200000,
    output_buffer: int = 20000,
    verbose: bool = False,
):
    """Create a LangGraph node that applies context management to state.

    Use this when you want context management as a separate node in your graph.

    Args:
        strategy: Context management strategy
        max_tokens: Model's context window
        output_buffer: Tokens to reserve for output
        verbose: Log when context is clipped

    Returns:
        A LangGraph node function

    Example:
        from src.extensions.libraries.core_chat_engine.memory.context import create_context_management_node

        # Add node to your graph
        graph.add_node("context_management", create_context_management_node())
        graph.add_edge(START, "context_management")
        graph.add_edge("context_management", "agent")
    """
    def context_node(state: dict) -> dict:
        """Apply context management to the messages in state."""
        messages = state.get("messages", [])

        # Convert and apply context management
        from .context_strategies import manage_context, count_tokens
        messages_dict = _convert_messages_to_dict(messages)
        original_tokens = count_tokens(messages_dict)
        safe_limit = max_tokens - output_buffer

        if original_tokens > safe_limit:
            context_kwargs = {}
            if isinstance(strategy, str) and strategy == "summarize" or strategy == ContextStrategy.SUMMARIZE:
                # For summarization in a graph node, we'd need access to an LLM
                # This is a limitation - the node approach works best with non-summarization strategies
                if verbose:
                    import sys
                    print("[Context Node] Note: Summarization strategy in node mode requires LLM access", file=sys.stderr)

            managed_messages = manage_context(
                messages_dict,
                max_tokens=max_tokens,
                output_buffer=output_buffer,
                strategy=strategy if isinstance(strategy, ContextStrategy) else ContextStrategy(strategy),
                **context_kwargs
            )

            managed_tokens = count_tokens(managed_messages)

            if verbose:
                import sys
                print(f"[Context Node] Clipped: {original_tokens} → {managed_tokens} tokens", file=sys.stderr)

            # Reconstruct and return updated state
            return {"messages": _reconstruct_messages_from_dict(managed_messages)}

        # No changes needed
        return {}

    return context_node
