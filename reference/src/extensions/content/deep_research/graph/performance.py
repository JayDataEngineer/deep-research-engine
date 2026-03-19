"""Performance utilities for Deep Research.

This module provides:
- Async concurrency helpers for parallel operations
- Caching layer for expensive operations
- Bulk operation utilities

CRASH-ONLY: Performance failures should crash, not degrade silently.
"""

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar, Tuple
try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python 3.10 and earlier
from functools import wraps
from collections import OrderedDict

from ..logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])


# =============================================================================
# ASYNC CONCURRENCY HELPERS
# =============================================================================

T = TypeVar('T')


async def gather_with_crash(
    *coros,
    timeout: Optional[float] = None,
    continue_on_error: bool = False,
) -> List[Any]:
    """Gather coroutines with crash-only error handling.

    Unlike asyncio.gather which returns exceptions, this crashes immediately
    on any error (unless continue_on_error is True).

    Args:
        *coros: Coroutines to run in parallel
        timeout: Optional timeout in seconds
        continue_on_error: If True, collect exceptions instead of crashing

    Returns:
        List of results in the same order as coros

    Raises:
        RuntimeError: If any coroutine fails and continue_on_error=False
        asyncio.TimeoutError: If timeout is exceeded
    """
    if continue_on_error:
        # Use return_exceptions=True and filter later
        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=True),
            timeout=timeout,
        )

        # Check for exceptions and log them
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            for exc in exceptions:
                logger.error(f"[PERFORMANCE] Coroutine failed: {exc}")

        return results

    # Crash immediately on any error
    try:
        return await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False),
            timeout=timeout,
        )
    except Exception as e:
        raise RuntimeError(
            f"Parallel operation failed: {e}\n"
            f"This is a CRITICAL error that prevents proper context gathering."
        ) from e


async def gather_with_retry(
    *coros,
    max_retries: int = 3,
    backoff: float = 1.0,
) -> List[Any]:
    """Gather coroutines with retry logic.

    Args:
        *coros: Coroutines to run
        max_retries: Maximum retry attempts
        backoff: Initial backoff delay (doubles each retry)

    Returns:
        List of results

    Raises:
        RuntimeError: If all retries exhausted
    """
    last_error = None
    current_backoff = backoff

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.gather(*coros, return_exceptions=False)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"[PERFORMANCE] Gather attempt {attempt + 1} failed, "
                    f"retrying in {current_backoff}s: {e}"
                )
                await asyncio.sleep(current_backoff)
                current_backoff *= 2
            else:
                raise RuntimeError(
                    f"Parallel operation failed after {max_retries} retries: {e}"
                ) from e


async def bounded_gather(
    *coros,
    semaphore: Optional[asyncio.Semaphore] = None,
    limit: int = 10,
) -> List[Any]:
    """Gather coroutines with concurrency limit.

    Args:
        *coros: Coroutines to run
        semaphore: Optional semaphore for limiting
        limit: Concurrency limit if semaphore not provided

    Returns:
        List of results
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)

    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *(run_with_semaphore(c) for c in coros),
        return_exceptions=False
    )


# =============================================================================
# CACHING LAYER
# =============================================================================

class CacheEntry:
    """A single cache entry with TTL."""

    def __init__(
        self,
        value: Any,
        ttl: Optional[float] = None,
    ):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def age(self) -> float:
        """Get the age of this entry in seconds."""
        return time.time() - self.created_at


class LRUCache:
    """Simple LRU cache with TTL support.

    CRASH-ONLY: Cache failures don't crash, but cache corruption is logged.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]

    def _evict_if_full(self) -> None:
        """Evict oldest entry if cache is full."""
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        self._cleanup_expired()

        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return default

        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return default

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        self._cleanup_expired()
        self._evict_if_full()

        if ttl is None:
            ttl = self.default_ttl

        self._cache[key] = CacheEntry(value, ttl)
        self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            pattern: Optional pattern to match (invalidates all if None)

        Returns:
            Number of entries invalidated
        """
        if pattern is None:
            count = len(self._cache)
            self.clear()
            return count

        # Pattern-based invalidation
        to_remove = [
            k for k in self._cache.keys()
            if pattern in k
        ]

        for key in to_remove:
            del self._cache[key]

        return len(to_remove)


# Global cache instances
_search_cache = LRUCache(max_size=500, default_ttl=300)  # 5 min TTL
_llm_cache = LRUCache(max_size=1000, default_ttl=1800)  # 30 min TTL


def get_search_cache() -> LRUCache:
    """Get the global search cache."""
    return _search_cache


def get_llm_cache() -> LRUCache:
    """Get the global LLM cache."""
    return _llm_cache


# =============================================================================
# CACHE DECORATORS
# =============================================================================

def cached(
    cache: Optional[LRUCache] = None,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None,
):
    """Decorator for caching async function results.

    Args:
        cache: Cache instance (uses search cache if None)
        key_fn: Function to generate cache key (uses args if None)
        ttl: TTL for cached results

    Returns:
        Decorated function
    """
    if cache is None:
        cache = _search_cache

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # Default key generation from args
                key_parts = [str(a) for a in args]
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = f"{func.__name__}:{hashlib.sha256(':'.join(key_parts).encode()).hexdigest()}"

            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                logger.debug(f"[CACHE] Hit for key: {key[:50]}...")
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl)
            logger.debug(f"[CACHE] Cached result for key: {key[:50]}...")

            return result

        return wrapper
    return decorator


def cached_llm(
    model: Optional[str] = None,
    ttl: Optional[float] = None,
):
    """Decorator for caching LLM results.

    Args:
        model: Model name for key prefix
        ttl: TTL for cached results

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Extract prompt from args (assumes first arg is prompt)
            prompt = args[0] if args else kwargs.get('prompt', '')

            # Generate cache key from prompt and model
            model_name = model or kwargs.get('model', 'default')
            key = f"llm:{model_name}:{hashlib.sha256(prompt.encode()).hexdigest()}"

            # Check cache
            cached_value = _llm_cache.get(key)
            if cached_value is not None:
                logger.debug(f"[LLM_CACHE] Hit for model={model_name}")
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            _llm_cache.set(key, result, ttl)
            logger.debug(f"[LLM_CACHE] Cached result for model={model_name}")

            return result

        return wrapper
    return decorator


# =============================================================================
# BULK OPERATIONS
# =============================================================================

async def bulk_store_artifacts(
    store: Any,
    artifacts: List[Any],
    session_id: str,
    batch_size: int = 100,
) -> int:
    """Store artifacts in batches for better performance.

    Args:
        store: Storage backend with add method
        artifacts: Artifacts to store
        session_id: Session ID
        batch_size: Number of artifacts per batch

    Returns:
        Total number of artifacts stored

    Raises:
        RuntimeError: If storage fails (CRASH-ONLY)
    """
    total_stored = 0

    for i in range(0, len(artifacts), batch_size):
        batch = artifacts[i:i + batch_size]

        try:
            await store.add(batch, session_id)
            total_stored += len(batch)
            logger.debug(
                f"[BULK] Stored batch {i // batch_size + 1}: "
                f"{len(batch)} artifacts (total: {total_stored})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Bulk storage failed at batch {i // batch_size + 1}: {e}"
            ) from e

    return total_stored


async def bulk_search(
    search_fn: Callable,
    queries: List[str],
    session_id: str,
    k: int = 10,
    concurrency: int = 5,
) -> List[List[Any]]:
    """Perform multiple searches in parallel with concurrency control.

    Args:
        search_fn: Search function (should be async)
        queries: Search queries
        session_id: Session ID
        k: Number of results per query
        concurrency: Max concurrent searches

    Returns:
        List of result lists (one per query)

    Raises:
        RuntimeError: If any search fails (CRASH-ONLY)
    """
    async def search_one(query: str) -> List[Any]:
        return await search_fn(query=query, session_id=session_id, k=k)

    return await bounded_gather(
        *(search_one(q) for q in queries),
        limit=concurrency,
    )


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> Self:
        """Start monitoring."""
        self._start_time = time.time()
        return self

    def stop(self) -> Self:
        """Stop monitoring."""
        self._end_time = time.time()
        return self

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    def log(self) -> None:
        """Log the performance metric."""
        logger.info(
            f"[PERF] {self.name}: {self.elapsed:.2f}s"
        )

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
        self.log()


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Decorated function
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> T:
        monitor = PerformanceMonitor(func.__name__)
        monitor.start()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            monitor.stop()
            monitor.log()

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> T:
        monitor = PerformanceMonitor(func.__name__)
        monitor.start()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            monitor.stop()
            monitor.log()

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
