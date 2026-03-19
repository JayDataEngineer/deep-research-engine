"""Config package for deep research graph.

This package provides configuration loading, caching, and management utilities.
It is organized into modular components:

- constants: Environment variable caches
- context: Context variables for thread-safe data passing
- loader: YAML loading, caching, and merging
- modes: Mode transformations (overdrive mode)
- runtime: Runtime config utilities

All public APIs are re-exported from this module for backward compatibility.
"""

# Constants (env var caches)
from .constants import (
    _LLM_MODEL_NAME,
    _OPENAI_API_KEY,
    _LLM_API_KEY,
    _OPENAI_BASE_URL,
    _LLM_BASE_URL,
)

# Context functions
from .context import (
    get_run_logger_from_config,
    set_run_logger_context,
    reset_run_logger_context,
    set_yaml_config_context,
    get_yaml_config_from_context,
    reset_yaml_config_context,
    set_context_engine_context,
    get_context_engine_from_context,
    reset_context_engine_context,
)

# Loader functions
from .loader import (
    _load_yaml_config,
    _apply_config_overrides,
    _expand_env_vars,
    _validate_config,
    get_config_from_runnable_config,
)

# Mode transformations
from .modes import (
    _apply_overdrive_mode,
)

# Runtime utilities
from .runtime import (
    yaml_to_runtime_config,
    get_runtime_config,
)


__all__ = [
    # Constants
    "_LLM_MODEL_NAME",
    "_OPENAI_API_KEY",
    "_LLM_API_KEY",
    "_OPENAI_BASE_URL",
    "_LLM_BASE_URL",
    # Context functions
    "get_run_logger_from_config",
    "set_run_logger_context",
    "reset_run_logger_context",
    "set_yaml_config_context",
    "get_yaml_config_from_context",
    "reset_yaml_config_context",
    "set_context_engine_context",
    "get_context_engine_from_context",
    "reset_context_engine_context",
    # Loader functions
    "_load_yaml_config",
    "_apply_config_overrides",
    "_expand_env_vars",
    "_validate_config",
    "get_config_from_runnable_config",
    # Mode transformations
    "_apply_overdrive_mode",
    # Runtime utilities
    "yaml_to_runtime_config",
    "get_runtime_config",
]
