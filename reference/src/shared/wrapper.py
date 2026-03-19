"""LLM Wrapper - Dynamic wrapper for applying middleware to LLM instances.

Allows middleware to be applied at the factory level rather than as decorators
in each LLM class. Follows the LangChain middleware pattern.

Usage:
    llm = get_llm(model="gpt-4o", middleware=[RetryMiddleware()])
    # Factory applies middleware via LLMWrapper

Or directly:
    wrapped = LLMWrapper(llm, [RetryMiddleware(), CacheMiddleware()])
"""

import logging
from typing import AsyncIterator, Optional, List, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .base import BaseLLM
    from .retry import RetryMiddleware

from src.shared.logging_config import get_logger

logger = get_logger(__name__)


class LLMWrapper:
    """Dynamic wrapper that applies middleware to LLM calls.

    Wraps a BaseLLM instance and intercepts ainvoke/astream calls,
    applying middleware before delegating to the wrapped LLM.
    """

    def __init__(
        self,
        llm: "BaseLLM",
        middleware: Optional[List["RetryMiddleware"]] = None
    ):
        """Initialize the wrapper with an LLM and middleware list.

        Args:
            llm: The LLM instance to wrap
            middleware: List of middleware to apply (in order)
        """
        self._llm = llm
        self._middleware = middleware or []

        # Copy all attributes from wrapped LLM for transparency
        # This allows the wrapper to be a drop-in replacement
        # CRITICAL: Don't copy private attributes (starting with _) to avoid overwriting our own
        for attr in dir(llm):
            if attr.startswith('_'):
                continue  # Don't copy private attributes
            if callable(getattr(llm, attr, None)):
                continue  # Don't wrap methods
            try:
                setattr(self, attr, getattr(llm, attr))
            except (AttributeError, TypeError):
                pass

        # Use object.__getattribute__ to access _middleware directly, bypassing __getattr__
        middleware = object.__getattribute__(self, '_middleware')
        logger.debug(f"[LLMWrapper] Wrapped LLM with {len(middleware)} middleware")

    async def ainvoke(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ):
        """Invoke chat completion with middleware applied.

        Middleware is applied in order - each middleware's call_with_retry
        wraps the next, with the actual LLM call at the center.

        Args:
            messages: Chat messages
            tools: Optional tools for tool calling
            **kwargs: Additional arguments

        Returns:
            LLMChunk response
        """
        # Build the call chain: middleware[n](middleware[n-1](...middleware[0](llm.ainvoke)...))
        call_func = self._llm.ainvoke

        # Apply middleware in reverse order so first middleware wraps outermost
        for middleware in reversed(self._middleware):
            if hasattr(middleware, 'call_with_retry'):
                def make_wrapper(mw, inner_func):
                    async def wrapped(*args, **inner_kwargs):
                        return await mw.call_with_retry(inner_func, *args, **inner_kwargs)
                    return wrapped
                call_func = make_wrapper(middleware, call_func)

        return await call_func(messages, tools=tools, **kwargs)

    async def astream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> AsyncIterator:
        """Stream chat completion with middleware applied.

        Note: For streaming, middleware only applies to the initial
        connection setup. Once streaming begins, chunks are yielded
        directly without middleware intervention.

        Args:
            messages: Chat messages
            tools: Optional tools for tool calling
            **kwargs: Additional arguments

        Yields:
            LLMChunk response chunks
        """
        # Build the call chain similar to ainvoke
        async def stream_func():
            async for chunk in self._llm.astream(messages, tools=tools, **kwargs):
                yield chunk

        # Apply middleware to the stream initiation
        call_func = stream_func

        for middleware in reversed(self._middleware):
            if hasattr(middleware, 'call_with_retry'):
                async def make_wrapper(mw, inner_func):
                    async def wrapped(*args, **inner_kwargs):
                        async for chunk in await mw.call_with_retry(inner_func, *args, **inner_kwargs):
                            yield chunk
                    return wrapped
                call_func = make_wrapper(middleware, call_func)

        # Execute the wrapped stream
        async for chunk in call_func():
            yield chunk

    def bind_tools(self, tools: List, **kwargs) -> "LLMWrapper":
        """Bind tools to the wrapped LLM.

        Returns a new wrapper with the same middleware but tools bound.

        Args:
            tools: Tools to bind
            **kwargs: Additional arguments

        Returns:
            New LLMWrapper instance with bound tools
        """
        wrapped_llm = self._llm.bind_tools(tools, **kwargs)
        return LLMWrapper(wrapped_llm, self._middleware)

    @property
    def profile(self):
        """Return LLM capability profile from wrapped LLM."""
        return self._llm.profile

    def supports_reasoning(self) -> bool:
        """Check if wrapped LLM supports reasoning."""
        return self._llm.supports_reasoning()

    def supports_tool_calling(self) -> bool:
        """Check if wrapped LLM supports tool calling."""
        return self._llm.supports_tool_calling()

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attribute access to wrapped LLM.

        CRITICAL: Check if attribute exists on self first to prevent recursion
        when the wrapped LLM is also an LLMWrapper (infinite loop).
        """
        # CRITICAL: Prevent infinite recursion if _llm isn't set yet
        if name == '_llm' or name == '_middleware':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        try:
            # Try to get from self first (using object.__getattribute__ to avoid recursion)
            return object.__getattribute__(self, name)
        except AttributeError:
            # Not found on self, proxy to wrapped LLM
            return getattr(self._llm, name)

    def __repr__(self) -> str:
        """String representation showing wrapped LLM and middleware."""
        middleware_names = [m.__class__.__name__ for m in self._middleware]
        return f"LLMWrapper(llm={self._llm.__class__.__name__}, middleware={middleware_names})"
