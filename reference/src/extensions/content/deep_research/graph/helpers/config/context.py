"""Context variables for thread-safe, async-safe data passing.

This module provides context variables for passing data through the
LangGraph execution without modifying state or function signatures.
"""

import contextvars
from typing import Dict, Any, Optional


# =============================================================================
# CONTEXT VARIABLES (for thread-safe, async-safe data passing)
# =============================================================================

_run_logger_context: contextvars.ContextVar = contextvars.ContextVar('run_logger', default=None)
"""Context variable for passing RunLogger through the graph execution."""

_yaml_config_context: contextvars.ContextVar = contextvars.ContextVar('yaml_config', default=None)
"""Context variable for passing YAML config through the graph execution."""

_context_engine_context: contextvars.ContextVar = contextvars.ContextVar('context_engine', default=None)
"""Context variable for passing Context Engine through the graph execution."""


# =============================================================================
# RUN LOGGER CONTEXT FUNCTIONS
# =============================================================================

def get_run_logger_from_config(config: Dict[str, Any] = None):
    """Get run logger from config or context variable.

    First checks the config dict for run_logger (for LangGraph config propagation),
    then falls back to context variable (for thread-safe, async-safe access).

    Args:
        config: Config dictionary (may contain run_logger in configurable)

    Returns:
        RunLogger instance or None
    """
    # First, try to get from config (for LangGraph config propagation)
    if config is not None:
        configurable = config.get("configurable", {})
        if "run_logger" in configurable:
            return configurable["run_logger"]

    # Fall back to context variable (thread-safe, async-safe)
    return _run_logger_context.get(None)


def set_run_logger_context(run_logger):
    """Set run logger in context variable.

    Args:
        run_logger: RunLogger instance or None

    Returns:
        Context token (for resetting later)
    """
    # Set the run_logger in the context variable
    return _run_logger_context.set(run_logger)


def reset_run_logger_context(token):
    """Reset run logger context.

    Args:
        token: Context token from set_run_logger_context
    """
    # Reset the context variable to the previous value
    _run_logger_context.reset(token)


# =============================================================================
# YAML CONFIG CONTEXT FUNCTIONS
# =============================================================================

def set_yaml_config_context(yaml_config: Dict[str, Any]):
    """Set yaml config in context variable.

    Args:
        yaml_config: YAML config dict from _load_yaml_config()

    Returns:
        Context token (for resetting later)
    """
    return _yaml_config_context.set(yaml_config)


def get_yaml_config_from_context() -> Optional[Dict[str, Any]]:
    """Get yaml config from context variable.

    This is the standard way for LangGraph V1 nodes to access yaml config
    without receiving it as a function parameter.

    Returns:
        YAML config dict or None if not set

    Example usage in node:
        async def worker_node(state: GlobalState, config: RunnableConfig = None) -> Dict[str, Any]:
            yaml_config = get_yaml_config_from_context()
            run_logger = get_run_logger_from_config(config)  # Pass config for LangGraph propagation

            # Use yaml_config for accessing configuration values
            spawner_config = yaml_config.get("spawner_config", {})

            return {"research_brief": brief}
    """
    return _yaml_config_context.get(None)


def reset_yaml_config_context(token):
    """Reset yaml config context.

    Args:
        token: Context token from set_yaml_config_context
    """
    _yaml_config_context.reset(token)


# =============================================================================
# CONTEXT ENGINE CONTEXT FUNCTIONS
# =============================================================================

def set_context_engine_context(context_engine):
    """Set context engine in context variable.

    Args:
        context_engine: ContextEngineBackends instance

    Returns:
        Context token (for resetting later)
    """
    return _context_engine_context.set(context_engine)


def get_context_engine_from_context():
    """Get context engine from context variable.

    This is the standard way for LangGraph nodes to access the Context Engine
    without receiving it as a function parameter and without storing it in state
    (which would cause serialization issues).

    Returns:
        ContextEngineBackends instance or None if not set
    """
    return _context_engine_context.get(None)


def reset_context_engine_context(token):
    """Reset context engine context.

    Args:
        token: Context token from set_context_engine_context
    """
    _context_engine_context.reset(token)
