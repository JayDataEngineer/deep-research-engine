"""Base LLM interface - All LLM implementations must inherit from this.

Provides a generic interface for streaming chat completions with reasoning support.
Different LLM providers (OpenAI, Claude, ZAI, etc.) implement this interface.

COMPLIANCE: ALL methods use Pydantic models for type safety and validation.
COMPLIANCE: ALL helper methods used by subclasses are defined here.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, ValidationError

from src.shared.logging_config import get_logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from .config import LLMConfig
    from .schema import ChatMessage, LLMRequest, ToolDefinition


# Global semaphore for limiting concurrent LLM API calls
# This prevents hitting rate limits (429 errors) from too many parallel requests
# LiteLLM concurrency limit - adjust based on your LiteLLM proxy configuration
_LLM_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(5)
_LLM_CONCURRENCY_LIMIT = 5

# Small stagger delay (seconds) before acquiring semaphore to spread out requests
# This helps prevent burst rate limiting when multiple calls start simultaneously
# Adjust based on your LiteLLM/proxy rate limits
_LLM_STAGGER_DELAY = 0.5

# Granular timeout configuration for httpx/OpenAI clients
# Using separate timeouts allows faster detection of send failures vs read timeouts
_LLM_TIMEOUT_CONNECT = 10.0  # 10s to establish connection
_LLM_TIMEOUT_READ = 300.0    # 5s to read response (LLM can take long)
_LLM_TIMEOUT_WRITE = 30.0   # 30s to SEND request (catch stuck sends fast!)
_LLM_TIMEOUT_POOL = 10.0     # 10s to get connection from pool


def get_llm_timeout():
    """Get the httpx Timeout configuration for LLM clients.

    Returns:
        httpx.Timeout or float to pass to OpenAI client
    """
    try:
        from httpx import Timeout
        return Timeout(
            connect=_LLM_TIMEOUT_CONNECT,
            read=_LLM_TIMEOUT_READ,
            write=_LLM_TIMEOUT_WRITE,
            pool=_LLM_TIMEOUT_POOL,
        )
    except ImportError:
        # Fallback to single timeout if httpx not available
        return _LLM_TIMEOUT_READ


def get_llm_httpx_event_hooks():
    """Get httpx event hooks for active monitoring of LLM requests.

    NOTE: Event hooks disabled due to httpx compatibility issues.
    The async hooks cause 'coroutine object is not callable' errors.

    Returns:
        Empty dict (no hooks)
    """
    # Event hooks disabled - causing issues with httpx
    return {}


def get_llm_concurrency_limit() -> int:
    """Get the current LLM concurrency limit."""
    return _LLM_CONCURRENCY_LIMIT


def set_llm_concurrency_limit(limit: int) -> None:
    """Set the LLM concurrency limit.

    Args:
        limit: Maximum concurrent LLM API calls (default: 8)
    """
    global _LLM_CONCURRENCY_LIMIT, _LLM_CONCURRENCY_SEMAPHORE
    _LLM_CONCURRENCY_LIMIT = limit
    _LLM_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(limit)


# Request size monitoring thresholds (matching factory.py)
_MAX_REQUEST_SIZE_BYTES = 10_000_000  # 10MB - ABORT if larger
_WARN_REQUEST_SIZE_BYTES = 1_000_000  # 1MB - Warn if larger
_CRITICAL_REQUEST_SIZE_BYTES = 5_000_000  # 5MB - Critical warning


def _log_llm_request_size(
    messages: List[Dict[str, str]],
    tools: Optional[List],
    model: str
) -> None:
    """Log LLM request size and check for obviously wrong data.

    This helps us detect:
    - Checkpoint data being included (should only be internal)
    - Context bloat from repeated messages
    - Tool definitions that are too large
    - Files/base64 data in messages

    Args:
        messages: Chat messages
        tools: Optional list of tools
        model: Model name for logging

    Raises:
        ValueError: If request size exceeds maximum
    """
    import json

    try:
        # Build approximate request body
        request_body = {"messages": messages}
        if tools:
            request_body["tools"] = tools

        # Calculate size
        request_json = json.dumps(request_body, default=str)
        size_bytes = len(request_json.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)

        logger.info(f"Request size: {size_mb:.2f} MB ({size_bytes:,} bytes) for model={model}")

        # Check thresholds
        if size_bytes > _MAX_REQUEST_SIZE_BYTES:
            error_msg = (
                f"ABORTING: Request size ({size_mb:.2f} MB) exceeds maximum!\n"
                f"Check for: checkpoint data, internal state, file inclusions, or context bloat.\n"
                f"Messages: {len(messages)}\n"
                f"Tools: {len(tools) if tools else 0}"
            )
            logger.error(error_msg)
            raise ValueError(f"Request payload too large: {size_mb:.2f} MB")

        elif size_bytes > _CRITICAL_REQUEST_SIZE_BYTES:
            logger.warning(
                f"CRITICAL: Very large request ({size_mb:.2f} MB). "
                f"High probability of send timeout!"
            )

        elif size_bytes > _WARN_REQUEST_SIZE_BYTES:
            logger.warning(
                f"WARNING: Large request ({size_mb:.2f} MB). "
                f"May cause send timeouts."
            )

        # Log breakdown
        messages_size = len(json.dumps(messages, default=str).encode('utf-8'))
        tools_size = len(json.dumps(tools, default=str).encode('utf-8')) if tools else 0

        logger.debug(
            f"Breakdown: "
            f"messages={messages_size:,} bytes ({len(messages)} msgs), "
            f"tools={tools_size:,} bytes ({len(tools) if tools else 0} tools)"
        )

        # Check for obviously wrong things in messages
        for i, msg in enumerate(messages):
            # Handle both dict-like and message object types
            if hasattr(msg, "content"):
                content = str(msg.content) if msg.content is not None else ""
            elif isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)

            if isinstance(content, str):
                # Check for checkpoint data (JSON with "checkpoint" key)
                if '"checkpoint"' in content.lower() and '"thread_id"' in content.lower():
                    logger.warning(f"Message {i} contains checkpoint-like data!")

                # Check for base64 data (files)
                if len(content) > 100000 and content.startswith("data:"):
                    logger.warning(f"Message {i} contains base64 data ({len(content):,} chars)!")

    except Exception as e:
        logger.warning(f"Failed to calculate request size: {e}")


class LLMProfile(BaseModel):
    """LLM capability profile (for LangGraph compatibility).

    LangGraph expects a `.profile` attribute on LLM objects that describes
    the model's capabilities. This provides that information.
    """
    name: str = "custom_llm"
    supports_tool_calling: bool = True
    supports_structured_output: bool = False
    supports_image_input: bool = False
    supports_audio_input: bool = False
    supports_video_input: bool = False


class BaseLLM(ABC):
    """Base interface for all LLM implementations.

    All LLM providers (OpenAI, Claude, ZAI, etc.) must implement this interface.
    This provides a unified API for the rest of the codebase.

    Required attributes for all implementations:
    - config: LLMConfig instance with validated parameters
    - model: Model name (str)
    - api_key: API key (str)
    - base_url: Base URL (str)
    - temperature: Temperature setting (float)
    - enable_thinking: Whether thinking mode is enabled (bool)
    - max_tokens: Max output tokens (Optional[int])
    - _bound_tools: Stored tools from bind_tools() call (Optional[List])
    - _bound_tools_kwargs: Additional kwargs from bind_tools() (Dict)
    - _middleware: List of middleware to apply to calls (List)

    LangGraph/LangChain compatibility:
    - profile: LLMProfile object describing capabilities
    - _llm_type: String identifier for serialization
    - bind_tools(): LangChain-compatible method for tool binding

    COMPLIANCE: All I/O uses Pydantic models for validation.
    COMPLIANCE: All helper methods used by subclasses are defined here.
    """

    config: "LLMConfig"  # Pydantic config object
    _llm_type: str = "custom_llm"  # LangChain serialization attribute
    _bound_tools: Optional[List] = None  # Tools bound via bind_tools()
    _bound_tools_kwargs: Dict[str, Any] = {}  # Additional kwargs from bind_tools()
    _middleware: List = []  # Middleware to apply to ainvoke/astream calls

    @abstractmethod
    def __init__(self, config: "LLMConfig", **kwargs):
        """Initialize the LLM with validated config.

        Args:
            config: LLMConfig instance with validated parameters
            **kwargs: Provider-specific arguments not in config

        Subclasses MUST set these attributes:
            - self.model = config.model
            - self.api_key = config.api_key
            - self.base_url = config.base_url
            - self.temperature = config.temperature
            - self.enable_thinking = config.enable_thinking
            - self.max_tokens = kwargs.get('max_tokens', config.max_tokens)
            - self._llm_type = "litellm" (all LLMs now use LiteLLM proxy)
        """
        pass

    @abstractmethod
    async def astream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> AsyncIterator["LLMChunk"]:
        """Stream chat completion.

        Args:
            messages: Chat messages (dict format for backward compatibility)
            tools: Optional list of tools for tool calling
            **kwargs: Additional arguments

        Yields:
            LLMChunk: Response chunks with content, reasoning, tool_calls

        Note: Messages are accepted as List[Dict] for backward compatibility.
        Implementations should validate internally or use MessageConverter.
        """
        pass

    @abstractmethod
    async def ainvoke(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> "LLMChunk":
        """Invoke chat completion (non-streaming).

        Args:
            messages: Chat messages (dict format for backward compatibility)
            tools: Optional list of tools for tool calling
            **kwargs: Additional arguments

        Returns:
            LLMChunk: Complete response
        """
        pass

    @abstractmethod
    def _serialize_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """Serialize a tool call to a standardized dict format.

        Each LLM provider may return tool calls in different formats.
        This method normalizes them to a common dict structure:
        {
            "id": str,        # Tool call ID
            "name": str,      # Tool/function name
            "args": Dict      # Tool arguments
        }

        Args:
            tool_call: Provider-specific tool call object (LangChain, OpenAI, etc.)

        Returns:
            Dict with keys: id, name, args

        Examples:
            LiteLLMWrapper handles LangChain/OpenAI tool calls via LiteLLM
        """
        pass

    def supports_reasoning(self) -> bool:
        """Check if this LLM supports chain-of-thought reasoning.

        Returns:
            bool: True if model can provide reasoning_content
        """
        return self.config.enable_thinking and hasattr(self, 'enable_thinking')

    def supports_tool_calling(self) -> bool:
        """Check if this LLM supports tool calling.

        Returns:
            bool: True if model supports tool calling
        """
        return True

    def bind_tools(self, tools: List, **kwargs) -> "BaseLLM":
        """Bind tools to this LLM for LangChain compatibility.

        This method follows LangChain's bind_tools() pattern. It stores the tools
        for use in subsequent astream()/ainvoke() calls.

        Args:
            tools: List of LangChain BaseTool objects or tool schemas
            **kwargs: Additional LangChain arguments (e.g., tool_choice)

        Returns:
            self (for chaining), with tools stored for subsequent calls
        """
        # Store tools for use in astream/ainvoke
        self._bound_tools = tools
        # Store any additional kwargs (like tool_choice) for LangChain compatibility
        self._bound_tools_kwargs = kwargs
        return self

    @property
    def profile(self) -> LLMProfile:
        """Return LLM capability profile (for LangGraph compatibility).

        LangGraph expects this attribute on all LLM objects to determine
        capabilities like tool calling, structured output, etc.

        Returns:
            LLMProfile: Profile describing model capabilities
        """
        return LLMProfile(
            name=getattr(self, 'model', 'custom_llm'),
            supports_tool_calling=self.supports_tool_calling(),
            supports_structured_output=False,  # Custom LLMs use manual parsing
            supports_image_input=False,
            supports_audio_input=False,
            supports_video_input=False
        )

    def _validate_required_attrs(self) -> None:
        """Validate that all required attributes are set.

        Raises:
            AttributeError: If any required attribute is missing
        """
        required_attrs = ['model', 'api_key', 'base_url', 'temperature', 'enable_thinking', '_llm_type', '_bound_tools', '_bound_tools_kwargs', '_middleware']
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]

        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} is missing required attributes: {missing}. "
                f"Make sure __init__ sets: model, api_key, base_url, temperature, enable_thinking, _llm_type, _bound_tools, _bound_tools_kwargs, _middleware"
            )

    def _init_config_params(self, config: "LLMConfig", **kwargs) -> None:
        """Initialize common config parameters with kwargs override support.

        This helper method reduces code duplication across LLM implementations.
        It stores generation parameters from config and allows kwargs to override.

        Args:
            config: LLMConfig instance with generation parameters
            **kwargs: Additional provider-specific arguments that override config

        Sets the following attributes on self:
            - config: The LLMConfig instance
            - model, api_key, base_url, temperature, enable_thinking: Core config values
            - max_tokens, top_p, top_k, frequency_penalty, presence_penalty,
              repeat_penalty, min_p, stop: Generation parameters
            - _extra_kwargs: Remaining kwargs for API pass-through
        """
        # Store core config values
        self.config = config
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.temperature = config.temperature
        self.enable_thinking = config.enable_thinking

        # Store generation parameters from config for attribute access
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty
        self.repeat_penalty = config.repeat_penalty
        self.min_p = config.min_p
        self.stop = config.stop

        # Allow kwargs to override config values (for backwards compatibility)
        override_params = [
            'max_tokens', 'top_p', 'top_k', 'frequency_penalty',
            'presence_penalty', 'repeat_penalty', 'min_p', 'stop'
        ]
        for param in override_params:
            if param in kwargs:
                setattr(self, param, kwargs.pop(param))

        # Filter out internal parameters that should NOT be passed to API
        # These are used by the factory/wrappers but not by the LLM API itself
        internal_params = ['rate_limiter', 'middleware']
        for param in internal_params:
            kwargs.pop(param, None)

        # Store remaining kwargs for direct API pass-through
        self._extra_kwargs = kwargs

    async def _apply_middleware_to_ainvoke(
        self,
        actual_ainvoke: callable,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> "LLMChunk":
        """Apply middleware to ainvoke call.

        Subclasses should call this at the start of their ainvoke method,
        passing their actual implementation as actual_ainvoke.

        Args:
            actual_ainvoke: The actual ainvoke implementation (usually self._actual_ainvoke)
            messages: Chat messages
            tools: Optional tools for tool calling
            **kwargs: Additional arguments

        Returns:
            LLMChunk response
        """
        if not self._middleware:
            # No middleware, call directly
            return await actual_ainvoke(messages, tools, **kwargs)

        # Build the call chain: middleware[n](middleware[n-1](...middleware[0](actual_ainvoke)...))
        call_func = actual_ainvoke

        # Apply middleware in reverse order so first middleware wraps outermost
        for middleware in reversed(self._middleware):
            if hasattr(middleware, 'call_with_retry'):
                def make_wrapper(mw, inner_func):
                    async def wrapped(*args, **inner_kwargs):
                        return await mw.call_with_retry(inner_func, *args, **inner_kwargs)
                    return wrapped
                call_func = make_wrapper(middleware, call_func)

        return await call_func(messages, tools=tools, **kwargs)

    async def _apply_middleware_to_astream(
        self,
        actual_astream: callable,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> AsyncIterator["LLMChunk"]:
        """Apply middleware to astream call.

        Subclasses should call this at the start of their astream method,
        passing their actual implementation as actual_astream.

        Args:
            actual_astream: The actual astream implementation (usually self._actual_astream)
            messages: Chat messages
            tools: Optional tools for tool calling
            **kwargs: Additional arguments

        Yields:
            LLMChunk response chunks
        """
        if not self._middleware:
            # No middleware, stream directly
            async for chunk in actual_astream(messages, tools, **kwargs):
                yield chunk
            return

        # With middleware, use the streaming-aware retry
        # Start with the actual stream function
        stream_func = actual_astream

        # Apply middleware in reverse order (first middleware wraps outermost)
        for middleware in reversed(self._middleware):
            if hasattr(middleware, 'astream_with_retry'):
                # Use the streaming-aware retry method
                original_func = stream_func
                mw = middleware  # Capture for closure

                # Create a wrapper that uses astream_with_retry
                def make_wrapper(inner_func):
                    async def wrapped(*args, **inner_kwargs):
                        async for chunk in mw.astream_with_retry(inner_func, *args, **inner_kwargs):
                            yield chunk
                    return wrapped

                stream_func = make_wrapper(original_func)
            elif hasattr(middleware, 'call_with_retry'):
                # Fallback to call_with_retry (not ideal for streaming)
                logger.warning(
                    f"[BaseLLM] Middleware {middleware.__class__.__name__} doesn't have astream_with_retry, "
                    "using call_with_retry (may have issues with streaming)"
                )
                original_func = stream_func
                mw = middleware  # Capture for closure

                def make_wrapper_fallback(inner_func):
                    async def wrapped(*args, **inner_kwargs):
                        # For call_with_retry, we need to execute and then iterate
                        result = await mw.call_with_retry(inner_func, *args, **inner_kwargs)
                        if result is not None:
                            # Try to iterate if it's a generator
                            async for chunk in result:
                                yield chunk
                    return wrapped

                stream_func = make_wrapper_fallback(original_func)

        # Execute the wrapped stream
        async for chunk in stream_func(messages, tools, **kwargs):
            yield chunk


class RateLimitedLLM(BaseLLM):
    """Wrapper that adds concurrency and rate limiting to any BaseLLM implementation.

    This wrapper provides two levels of limiting:
    1. Concurrency limiting: Global semaphore limits concurrent API calls
    2. Rate limiting: Strict throttle ensures minimum time between call STARTS

    This prevents rate limit errors (429) from multiple parallel requests.

    IMPORTANT: Rate limiting is GLOBAL and shared across all LLM instances.
    Even if you create 100 LLM instances, they all share the same rate limit state.

    Middleware can be passed and will be applied to all calls.

    Example:
        llm = get_llm("gpt-4o")  # Returns RateLimitedLLM(wrapped_llm)
    """

    # GLOBAL storage for last call time per base_url (shared across ALL instances)
    # This is a class variable, so all RateLimitedLLM instances share it
    _last_call_times: Dict[str, float] = {}
    _rate_lock = asyncio.Lock()

# Component-specific logger (uses llm_monitor config from config.yaml)
    _logger = get_logger(__name__)

    def __init__(self, wrapped_llm: BaseLLM, middleware: Optional[List] = None, rate_limiter: Optional[Any] = None):
        """Initialize with a wrapped LLM instance.

        Args:
            wrapped_llm: The actual LLM instance to wrap (ChatZAI, StandardChatLLM, etc.)
            middleware: Optional list of middleware to apply to calls
            rate_limiter: Optional InMemoryRateLimiter for per-provider rate limiting

        NOTE: Rate limiting is GLOBAL - all RateLimitedLLM instances share the same
        _last_call_times dict, keyed by base_url. This ensures that even if you create
        multiple LLM instances, they all respect the same rate limit.
        """
        self._wrapped = wrapped_llm
        self._rate_limiter = rate_limiter
        # Minimum seconds between call starts (derived from rate limiter)
        self._min_wait_seconds = None
        if rate_limiter is not None and rate_limiter.requests_per_second > 0:
            self._min_wait_seconds = 1.0 / rate_limiter.requests_per_second

        # Copy all required attributes from wrapped LLM
        self.config = wrapped_llm.config
        self.model = wrapped_llm.model
        self.api_key = wrapped_llm.api_key
        self.base_url = wrapped_llm.base_url
        self.temperature = wrapped_llm.temperature
        self.enable_thinking = wrapped_llm.enable_thinking
        self.max_tokens = wrapped_llm.max_tokens
        self._llm_type = wrapped_llm._llm_type
        self._bound_tools = wrapped_llm._bound_tools
        self._bound_tools_kwargs = wrapped_llm._bound_tools_kwargs
        self._middleware = middleware or []

        # Copy generation parameters for attribute access
        self.top_p = getattr(wrapped_llm, 'top_p', None)
        self.top_k = getattr(wrapped_llm, 'top_k', None)
        self.frequency_penalty = getattr(wrapped_llm, 'frequency_penalty', None)
        self.presence_penalty = getattr(wrapped_llm, 'presence_penalty', None)
        self.repeat_penalty = getattr(wrapped_llm, 'repeat_penalty', None)
        self.min_p = getattr(wrapped_llm, 'min_p', None)
        self.stop = getattr(wrapped_llm, 'stop', None)

    async def _acquire_rate_limit(self):
        """Acquire rate limiter permission before making API request.

        Uses strict throttling: ensures minimum time between call STARTS.
        CRASH-ONLY: No fallbacks - rate limiting is REQUIRED.
        """
        if self._min_wait_seconds is None:
            return  # No rate limiting (local servers)

        # Use base_url as key for per-provider rate limiting
        key = self.base_url

        import time
        now = time.time()

        # Acquire lock only to read/update shared state, then release before sleeping
        async with self._rate_lock:
            last_call = self._last_call_times.get(key, 0)
            earliest_allowed_start = last_call + self._min_wait_seconds

            # Calculate how long we need to wait (if at all)
            if now >= earliest_allowed_start:
                # No wait needed - update last call time and proceed
                self._last_call_times[key] = now
                import time
                api_start_time = time.time()
                self._logger.info(f" Call proceeding immediately (last call was {now - last_call:.2f}s ago), API call starting at {api_start_time:.2f}")
                return  # Proceed immediately

            # Need to wait - calculate wake time and store it BEFORE releasing lock
            wait_time = earliest_allowed_start - now
            wake_time = earliest_allowed_start
            # Reserve our slot by updating last_call_time NOW
            self._last_call_times[key] = wake_time
            # Log: show position in queue (other requests waiting) vs actual time since last call
            queue_position = f"(queued behind {(wake_time - now) / self._min_wait_seconds:.0f} requests)" if wake_time > now + self._min_wait_seconds else ""
            self._logger.info(f" Throttling {key[:40]}...: waiting {wait_time:.2f}s {queue_position}")

        # Sleep OUTSIDE the lock so other workers can check/update their wait times
        await asyncio.sleep(wait_time)
        import time
        api_start_time = time.time()
        self._logger.info(f" Throttling complete for {key[:40]}..., API call starting at {api_start_time:.2f}")

    async def astream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> AsyncIterator["LLMChunk"]:
        """Stream chat completion with rate limiting, concurrency limiting, stagger, middleware, and request monitoring."""
        # Log request size for monitoring (before any limiting)
        _log_llm_request_size(messages, tools, self.model)

        # Acquire rate limit permission (blocks if rate exceeded)
        await self._acquire_rate_limit()

        # Add small random stagger before acquiring semaphore to spread out requests
        import random
        if _LLM_STAGGER_DELAY > 0:
            await asyncio.sleep(random.uniform(0, _LLM_STAGGER_DELAY))

        # Define the actual stream implementation (with semaphore and wrapped LLM)
        async def actual_stream(msgs, tools, **kw):
            async with _LLM_CONCURRENCY_SEMAPHORE:
                async for chunk in self._wrapped.astream(msgs, tools, **kw):
                    yield chunk

        # Apply middleware if configured
        async for chunk in self._apply_middleware_to_astream(actual_stream, messages, tools, **kwargs):
            yield chunk

    async def ainvoke(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        **kwargs
    ) -> "LLMChunk":
        """Invoke chat completion with rate limiting, concurrency limiting, stagger, middleware, and request monitoring."""
        # Log request size for monitoring (before any limiting)
        _log_llm_request_size(messages, tools, self.model)

        # Acquire rate limit permission (blocks if rate exceeded)
        await self._acquire_rate_limit()

        # Add small random stagger before acquiring semaphore to spread out requests
        import random
        if _LLM_STAGGER_DELAY > 0:
            await asyncio.sleep(random.uniform(0, _LLM_STAGGER_DELAY))

        # Define the actual invoke implementation (with semaphore and wrapped LLM)
        async def actual_invoke(msgs, tools, **kw):
            async with _LLM_CONCURRENCY_SEMAPHORE:
                return await self._wrapped.ainvoke(msgs, tools, **kw)

        # Apply middleware if configured
        return await self._apply_middleware_to_ainvoke(actual_invoke, messages, tools, **kwargs)

    def _serialize_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """Delegate to wrapped LLM."""
        return self._wrapped._serialize_tool_call(tool_call)

    def supports_reasoning(self) -> bool:
        """Delegate to wrapped LLM."""
        return self._wrapped.supports_reasoning()

    def supports_tool_calling(self) -> bool:
        """Delegate to wrapped LLM."""
        return self._wrapped.supports_tool_calling()

    def bind_tools(self, tools: List, **kwargs) -> "BaseLLM":
        """Bind tools and return self (for chaining)."""
        # Call wrapped LLM's bind_tools to update its state
        self._wrapped.bind_tools(tools, **kwargs)
        # Also update our own state
        self._bound_tools = tools
        self._bound_tools_kwargs = kwargs
        return self

    def __getattr__(self, name: str):
        """Delegate unknown attributes to wrapped LLM.

        This allows RateLimitedLLM to support all methods of the wrapped LLM,
        including with_structured_output and any other custom methods.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._wrapped, name)


