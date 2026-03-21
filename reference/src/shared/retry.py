"""LLM Retry Middleware - Exponential backoff retry logic for LLM calls.

Provides configurable retry behavior similar to LangChain's ModelRetryMiddleware:
- Configurable max retries
- Exponential backoff with jitter
- Retry on specific error types
- Logging for debugging

Usage (factory-level middleware):
    llm = get_llm(
        model="gpt-4o",
        middleware=[RetryMiddleware(max_retries=3)]
    )

Usage (direct decorator):
    @with_llm_retry(max_retries=3, initial_delay=1.0)
    async def ainvoke(self, messages, tools=None, **kwargs):
        ...

Example config:
    retry_config:
        max_retries: 3
        initial_delay: 1.0  # seconds
        backoff_factor: 2.0  # exponential multiplier
        max_delay: 60.0  # max seconds between retries
        retry_on_status: [500, 502, 503, 504]  # HTTP status codes to retry
"""

import asyncio
from .logging_config import get_logger
from functools import wraps
from typing import Type, Tuple, List, Optional, Callable, Any, TYPE_CHECKING, AsyncIterator
from enum import Enum

if TYPE_CHECKING:
    from .base import BaseLLM

logger = get_logger(__name__)


class RetryableError(Enum):
    """Error types that should trigger retries."""

    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"


# Default HTTP status codes to retry
DEFAULT_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# Default exception types to retry
DEFAULT_RETRY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# httpx exceptions - imported lazily to avoid import errors
def get_httpx_timeout_exceptions():
    """Get httpx timeout exceptions for retry."""
    try:
        from httpx import TimeoutException, ReadTimeout, WriteTimeout, ConnectTimeout
        return (TimeoutException, ReadTimeout, WriteTimeout, ConnectTimeout)
    except ImportError:
        return ()

# Combined retry exceptions
def get_all_retry_exceptions():
    """Get all exceptions that should trigger retry."""
    return DEFAULT_RETRY_EXCEPTIONS + get_httpx_timeout_exceptions()


class RetryConfig:
    """Configuration for LLM retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        jitter: Add random jitter to delay to avoid thundering herd (default: True)
        retry_on_status: HTTP status codes that trigger retry
        retry_on_exceptions: Exception types that trigger retry
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retry_on_status: Optional[List[int]] = None,
        retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on_status = retry_on_status or DEFAULT_RETRY_STATUS_CODES
        self.retry_on_exceptions = retry_on_exceptions or get_all_retry_exceptions()

    def __repr__(self):
        return (
            f"RetryConfig(max_retries={self.max_retries}, "
            f"initial_delay={self.initial_delay}, "
            f"backoff_factor={self.backoff_factor})"
        )


def _should_retry_from_exception(exc: Exception, config: RetryConfig) -> bool:
    """Check if an exception should trigger a retry.

    Args:
        exc: The exception that occurred
        config: Retry configuration

    Returns:
        True if the exception should trigger a retry
    """
    # Check if exception type is in retry list
    for exc_type in config.retry_on_exceptions:
        if isinstance(exc, exc_type):
            return True

    # Check for specific API error patterns
    exc_str = str(exc).lower()
    error_patterns = [
        "rate limit",
        "too many requests",
        "timeout",
        "connection",
        "network",
        "temporary",
        "unavailable",
        "internal server error",
    ]

    for pattern in error_patterns:
        if pattern in exc_str:
            return True

    return False


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay before next retry using exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff: initial_delay * (backoff_factor ^ attempt)
    delay = config.initial_delay * (config.backoff_factor ** attempt)

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter to avoid thundering herd
    if config.jitter:
        import random
        delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay

    return delay


def with_llm_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on_status: Optional[List[int]] = None,
    retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable:
    """Decorator that adds retry logic to async LLM calls.

    This implements exponential backoff retry similar to LangChain's ModelRetryMiddleware.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Exponential backoff multiplier (default: 2.0)
        max_delay: Maximum delay between retries (default: 60.0)
        jitter: Add random jitter to delays (default: True)
        retry_on_status: HTTP status codes to retry (default: [429, 500, 502, 503, 504])
        retry_on_exceptions: Exception types to retry

    Returns:
        Decorated function with retry logic

    Example:
        @with_llm_retry(max_retries=5, initial_delay=2.0)
        async def ainvoke(self, messages, tools=None, **kwargs):
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
        jitter=jitter,
        retry_on_status=retry_on_status,
        retry_on_exceptions=retry_on_exceptions,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    should_retry = _should_retry_from_exception(e, config)

                    if not should_retry or attempt >= config.max_retries:
                        # Don't retry or max retries exceeded
                        logger.error(
                            f"[LLM-Retry] Failed after {attempt + 1} attempts: {e.__class__.__name__}: {e}"
                        )
                        raise

                    # Calculate delay and wait
                    delay = _calculate_delay(attempt, config)
                    logger.warning(
                        f"[LLM-Retry] Attempt {attempt + 1}/{config.max_retries + 1} failed: "
                        f"{e.__class__.__name__}: {e}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper

    return decorator


class RetryMiddleware:
    """Retry middleware for LLM calls - LangChain-style middleware pattern.

    Can be applied at the factory level:
        llm = get_llm(
            model="gpt-4o",
            middleware=[RetryMiddleware(max_retries=3)]
        )

    Or wraps an existing LLM instance:
        wrapped_llm = RetryMiddleware(max_retries=3).wrap(llm)
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        """Initialize retry middleware.

        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay in seconds (default: 1.0)
            backoff_factor: Exponential backoff multiplier (default: 2.0)
            max_delay: Maximum delay between retries (default: 60.0)
            jitter: Add random jitter to delays (default: True)
        """
        self.config = RetryConfig(
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=jitter,
        )

    def wrap(self, llm: "BaseLLM") -> "BaseLLM":
        """Wrap an LLM instance with retry logic.

        Creates a dynamic wrapper that intercepts ainvoke and astream calls,
        applying retry logic to each.

        Args:
            llm: The LLM instance to wrap

        Returns:
            Wrapped LLM with retry logic applied
        """
        from .wrapper import LLMWrapper

        return LLMWrapper(llm, self)

    async def call_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Call an async function with retry logic.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if we should retry
                should_retry = _should_retry_from_exception(e, self.config)

                if not should_retry or attempt >= self.config.max_retries:
                    logger.error(
                        f"[LLM-Retry] Failed after {attempt + 1} attempts: {e.__class__.__name__}: {e}"
                    )
                    raise

                # Calculate delay and wait
                delay = _calculate_delay(attempt, self.config)
                logger.warning(
                    f"[LLM-Retry] Attempt {attempt + 1}/{self.config.max_retries + 1} failed: "
                    f"{e.__class__.__name__}: {e}. Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        raise last_exception

    async def astream_with_retry(
        self,
        stream_func: Callable,
        *args,
        **kwargs
    ) -> AsyncIterator:
        """Stream with retry logic.

        For streaming, we wrap the entire stream operation and retry on failure.
        If the stream fails partway through, we restart the entire stream.

        Args:
            stream_func: Async generator function to call
            *args: Positional arguments for stream_func
            **kwargs: Keyword arguments for stream_func

        Yields:
            Chunks from the stream

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Create the async generator
                async for chunk in stream_func(*args, **kwargs):
                    yield chunk
                return  # Success - completed the stream

            except Exception as e:
                last_exception = e

                # Check if we should retry
                should_retry = _should_retry_from_exception(e, self.config)

                if not should_retry or attempt >= self.config.max_retries:
                    logger.error(
                        f"[LLM-Retry] Stream failed after {attempt + 1} attempts: {e.__class__.__name__}: {e}"
                    )
                    raise

                # Calculate delay and wait
                delay = _calculate_delay(attempt, self.config)
                logger.warning(
                    f"[LLM-Retry] Stream attempt {attempt + 1}/{self.config.max_retries + 1} failed: "
                    f"{e.__class__.__name__}: {e}. Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        raise last_exception


# Default retry middleware instance
default_retry = RetryMiddleware()
