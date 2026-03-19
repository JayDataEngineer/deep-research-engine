"""
Decorators for Deep Research Graph Nodes

Provides common decorators for logging, error handling, and timing
throughout the research system.
"""

import asyncio
import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger

from .constants import Phase, LogPrefix, format_phase_log, format_error_log
from .exceptions import DeepResearchError, ErrorHandler
from .helpers.config import get_run_logger_from_config


logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level

F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# Phase Logging Decorator
# ============================================================================

def log_phase(
    phase: Phase,
    get_logger_func: Optional[Callable] = None
):
    """
    Decorator to add phase start/end logging to node functions.

    Usage:
        @log_phase(Phase.WORKER)
        def worker_node(state, config):
            # Node implementation
            return result
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(state: Dict, **kwargs) -> Any:
            # Get config from kwargs (LangGraph passes config as keyword argument)
            config = kwargs.get("config", {})

            # Get run logger
            run_logger = None
            if get_logger_func:
                run_logger = get_logger_func(state, config)
            else:
                run_logger = get_run_logger_from_config(config)

            # Log phase start
            if run_logger:
                run_logger.log_phase_start(phase.value, details=config.get("metadata", {}))

            start_time = time.time()

            try:
                # Execute the actual function (await if it's async)
                result = func(state, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result

                # Log phase end
                elapsed = time.time() - start_time
                if run_logger:
                    run_logger.log_phase_end(phase.value, elapsed)

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                if run_logger:
                    run_logger.log_error(phase.value, e, context="phase_execution")

                logger.error(f"Error in {phase.value} phase: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(state: Dict, **kwargs) -> Any:
            # For non-async functions, call synchronously
            config = kwargs.get("config", {})

            run_logger = None
            if get_logger_func:
                run_logger = get_logger_func(state, config)
            else:
                run_logger = get_run_logger_from_config(config)

            if run_logger:
                run_logger.log_phase_start(phase.value, details=config.get("metadata", {}))

            start_time = time.time()

            try:
                result = func(state, **kwargs)

                elapsed = time.time() - start_time
                if run_logger:
                    run_logger.log_phase_end(phase.value, elapsed)

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                if run_logger:
                    run_logger.log_error(phase.value, e, context="phase_execution")

                logger.error(f"Error in {phase.value} phase: {e}")
                raise

        # Return async wrapper if function is async, else sync wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


# ============================================================================
# Timing Decorator
# ============================================================================

def timed(
    log_result: bool = True,
    get_logger_func: Optional[Callable] = None
):
    """
    Decorator to time function execution and optionally log results.

    Usage:
        @timed()
        def expensive_function(data):
            # Expensive computation
            return result
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            func_name = func.__name__

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                logger.debug(f"{func_name} completed in {elapsed:.2f}s")

                if log_result and get_logger_func:
                    run_logger = get_logger_func(*args, **kwargs) if get_logger_func else None
                    if run_logger:
                        run_logger._write_to_file(f"[TIMING] {func_name}: {elapsed:.2f}s")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper  # type: ignore
    return decorator


# ============================================================================
# Error Handling Decorator
# ============================================================================

def handle_phase_errors(
    phase: Phase,
    raise_on_error: bool = True,
    error_context: Optional[Dict] = None
):
    """
    Decorator to handle errors consistently across phases.

    Usage:
        @handle_phase_errors(Phase.WORKER, raise_on_error=False)
        def worker_node(state, config):
            # Node implementation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(state: Dict, **kwargs) -> Any:
            try:
                result = func(state, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except DeepResearchError:
                if raise_on_error:
                    raise
                return None
            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.handle(e, phase, error_context, raise_error=raise_on_error)

                if raise_on_error:
                    raise

                return None

        @functools.wraps(func)
        def sync_wrapper(state: Dict, **kwargs) -> Any:
            try:
                return func(state, **kwargs)
            except DeepResearchError:
                if raise_on_error:
                    raise
                return None
            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.handle(e, phase, error_context, raise_error=raise_on_error)

                if raise_on_error:
                    raise

                return None

        # Return async wrapper if function is async, else sync wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


# ============================================================================
# Retry Decorator
# ============================================================================

def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on failure with exponential backoff.

    Usage:
        @retry_on_error(max_attempts=3, exceptions=(ValueError, ConnectionError))
        def unreliable_call():
            # Might fail
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise

                    logger.warning(f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}, retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None  # Should never reach here

        return wrapper  # type: ignore
    return decorator


# ============================================================================
# Combined Decorator for Research Nodes
# ============================================================================

def research_node(
    phase: Phase,
    get_logger_func: Optional[Callable] = None
):
    """
    Combined decorator for research nodes that adds:
    - Phase start/end logging
    - Timing
    - Error handling

    Usage:
        @research_node(Phase.WORKER)
        def worker_node(state, config):
            # Node implementation - logging and error handling handled automatically
            return result
    """

    def decorator(func: F) -> F:
        # Apply decorators in order (both return async-aware wrappers)
        decorated = log_phase(phase, get_logger_func)(func)
        decorated = handle_phase_errors(phase, raise_on_error=True)(decorated)  # type: ignore

        @functools.wraps(decorated)
        async def async_final_wrapper(state: Dict, **kwargs) -> Any:
            result = decorated(state, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        @functools.wraps(decorated)
        def sync_final_wrapper(state: Dict, **kwargs) -> Any:
            return decorated(state, **kwargs)

        # Return async wrapper if function is async, else sync wrapper
        if asyncio.iscoroutinefunction(func):
            return async_final_wrapper  # type: ignore
        else:
            return sync_final_wrapper  # type: ignore

    return decorator


# ============================================================================
# Worker Execution Decorator
# ============================================================================

def log_worker_execution(
    worker_id: str,
    worker_type: str = "generic",
    get_logger_func: Optional[Callable] = None
):
    """
    Decorator for worker execution functions.

    Usage:
        @log_worker_execution(worker_id="worker_1", worker_type="researcher")
        def run_research_task(task, config):
            # Worker implementation
            return result
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(task: str, config: Dict, *args, **kwargs) -> Any:
            run_logger = None
            if get_logger_func:
                run_logger = get_logger_func(task, config, *args, **kwargs)

            if run_logger:
                run_logger.log_worker_start(worker_id, worker_type, task)

            start_time = time.time()

            try:
                result = func(task, config, *args, **kwargs)

                elapsed = time.time() - start_time

                if run_logger:
                    run_logger.log_worker_end(
                        worker_id=worker_id,
                        llm_calls=getattr(result, 'llm_calls', 0),
                        prompt_tokens=getattr(result, 'prompt_tokens', 0),
                        completion_tokens=getattr(result, 'completion_tokens', 0),
                        tool_calls=getattr(result, 'tool_calls', 0),
                        output_length=getattr(result, 'output_length', len(str(result))),
                        status="success"
                    )

                return result

            except Exception as e:
                elapsed = time.time() - start_time

                if run_logger:
                    run_logger.log_worker_end(
                        worker_id=worker_id,
                        status="failed",
                        error_message=str(e)
                    )

                logger.error(f"Worker {worker_id} failed: {e}")
                raise

        return wrapper  # type: ignore
    return decorator


# ============================================================================
# Debug Logging Decorator
# ============================================================================

def debug_log(
    get_logger_func: Optional[Callable] = None,
    log_args: bool = True,
    log_result: bool = True
):
    """
    Decorator for debug logging of function calls.

    Usage:
        @debug_log()
        def my_function(arg1, arg2):
            # Function implementation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__

            if log_args:
                logger.debug(f"[{func_name}] Called with args={args}, kwargs={kwargs}")

            result = func(*args, **kwargs)

            if log_result:
                result_str = str(result)[:100]
                logger.debug(f"[{func_name}] Returned: {result_str}...")

            return result

        return wrapper  # type: ignore
    return decorator


# ============================================================================
# Memoization Decorator
# ============================================================================

def memoize(max_size: int = 128):
    """
    Simple memoization decorator for caching function results.

    Usage:
        @memoize(max_size=64)
        def expensive_computation(input_data):
            # Expensive calculation
            return result
    """

    def decorator(func: F) -> F:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a cache key from args and kwargs
            key = str(args) + str(sorted(kwargs.items()))

            if key in cache:
                logger.debug(f"{func.__name__} cache hit for {key[:50]}...")
                return cache[key]

            result = func(*args, **kwargs)

            # Implement simple LRU by size
            if len(cache) >= max_size:
                # Remove first (oldest) item
                cache.pop(next(iter(cache)))

            cache[key] = result
            return result

        wrapper.cache_clear = lambda: cache.clear()  # type: ignore
        wrapper.cache_info = lambda: {"size": len(cache), "max_size": max_size}  # type: ignore

        return wrapper  # type: ignore

    return decorator


# Make it available for import
__all__ = [
    "log_phase",
    "timed",
    "handle_phase_errors",
    "retry_on_error",
    "research_node",
    "log_worker_execution",
    "debug_log",
    "memoize",
]
