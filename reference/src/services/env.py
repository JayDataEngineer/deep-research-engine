"""Environment Variable Processing Utilities.

Centralized, type-safe environment variable access with:
- Type casting (int, float, bool, list)
- Default values
- Fallback chains
- Validation
- Caching

Usage:
    from src.services.env import get_env, get_env_int, get_env_bool, get_env_list

    # Simple string
    api_key = get_env("OPENAI_API_KEY")

    # With default
    host = get_env("HOST", "localhost")

    # Integer with default
    port = get_env_int("PORT", 8080)

    # Boolean
    debug = get_env_bool("DEBUG", default=False)

    # List (comma-separated)
    allowed_hosts = get_env_list("ALLOWED_HOSTS", ["localhost"])

    # Fallback chain (first non-empty wins)
    api_key = get_env_fallback("Z_AI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
"""

import os
from typing import Optional, List, Any, Callable, TypeVar, Union
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def get_env(
    key: str,
    default: Optional[str] = None,
    *,
    required: bool = False,
    strip: bool = True,
) -> Optional[str]:
    """Get environment variable as string.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set and no default
        strip: Strip whitespace from value

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required=True and variable not set

    Example:
        >>> host = get_env("HOST", "localhost")
        >>> api_key = get_env("API_KEY", required=True)
    """
    value = os.environ.get(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default

    return value.strip() if strip else value


def get_env_int(
    key: str,
    default: int = 0,
    *,
    required: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Get environment variable as integer.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set and no default
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        Integer value

    Raises:
        ValueError: If required, invalid format, or out of range

    Example:
        >>> port = get_env_int("PORT", 8080, min_value=1, max_value=65535)
    """
    value = os.environ.get(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default

    try:
        result = int(value.strip())
    except ValueError:
        raise ValueError(f"Environment variable '{key}' must be an integer, got: {value}")

    if min_value is not None and result < min_value:
        raise ValueError(f"Environment variable '{key}' must be >= {min_value}, got: {result}")

    if max_value is not None and result > max_value:
        raise ValueError(f"Environment variable '{key}' must be <= {max_value}, got: {result}")

    return result


def get_env_float(
    key: str,
    default: float = 0.0,
    *,
    required: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Get environment variable as float.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set and no default
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        Float value

    Example:
        >>> threshold = get_env_float("THRESHOLD", 0.5, min_value=0.0, max_value=1.0)
    """
    value = os.environ.get(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default

    try:
        result = float(value.strip())
    except ValueError:
        raise ValueError(f"Environment variable '{key}' must be a float, got: {value}")

    if min_value is not None and result < min_value:
        raise ValueError(f"Environment variable '{key}' must be >= {min_value}, got: {result}")

    if max_value is not None and result > max_value:
        raise ValueError(f"Environment variable '{key}' must be <= {max_value}, got: {result}")

    return result


def get_env_bool(
    key: str,
    default: bool = False,
    *,
    required: bool = False,
) -> bool:
    """Get environment variable as boolean.

    Truthy values: "true", "1", "yes", "on", "enabled" (case-insensitive)
    Falsy values: "false", "0", "no", "off", "disabled", "" (case-insensitive)

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set and no default

    Returns:
        Boolean value

    Example:
        >>> debug = get_env_bool("DEBUG", default=False)
        >>> enabled = get_env_bool("FEATURE_ENABLED")
    """
    value = os.environ.get(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default

    value_lower = value.strip().lower()

    if value_lower in ("true", "1", "yes", "on", "enabled"):
        return True
    elif value_lower in ("false", "0", "no", "off", "disabled", ""):
        return False
    else:
        raise ValueError(
            f"Environment variable '{key}' must be a boolean, got: {value}. "
            f"Expected: true/false, 1/0, yes/no, on/off, enabled/disabled"
        )


def get_env_list(
    key: str,
    default: Optional[List[str]] = None,
    *,
    separator: str = ",",
    required: bool = False,
    strip_items: bool = True,
    filter_empty: bool = True,
) -> List[str]:
    """Get environment variable as list of strings.

    Args:
        key: Environment variable name
        default: Default value if not set
        separator: String to split on (default: comma)
        required: Raise error if not set and no default
        strip_items: Strip whitespace from each item
        filter_empty: Remove empty strings from result

    Returns:
        List of strings

    Example:
        >>> hosts = get_env_list("ALLOWED_HOSTS", ["localhost"])
        >>> tags = get_env_list("TAGS", separator=";")
    """
    if default is None:
        default = []

    value = os.environ.get(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default

    items = value.split(separator)

    if strip_items:
        items = [item.strip() for item in items]

    if filter_empty:
        items = [item for item in items if item]

    return items


def get_env_fallback(*keys: str, default: Optional[str] = None) -> Optional[str]:
    """Get first non-empty environment variable from a list of keys.

    Args:
        *keys: Environment variable names (checked in order)
        default: Default value if none are set

    Returns:
        First non-empty value or default

    Example:
        >>> api_key = get_env_fallback("Z_AI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
    """
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()

    return default


def get_env_json(
    key: str,
    default: Any = None,
    *,
    required: bool = False,
) -> Any:
    """Get environment variable as parsed JSON.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set and no default

    Returns:
        Parsed JSON value

    Example:
        >>> config = get_env_json("CONFIG", {"enabled": True})
    """
    import json

    value = os.environ.get(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default

    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"Environment variable '{key}' must be valid JSON: {e}")


def require_env(key: str) -> str:
    """Get required environment variable (convenience function).

    Args:
        key: Environment variable name

    Returns:
        Environment variable value

    Raises:
        ValueError: If variable not set

    Example:
        >>> api_key = require_env("API_KEY")
    """
    return get_env(key, required=True)  # type: ignore


# =============================================================================
# Cached Configuration Helpers
# =============================================================================

@lru_cache(maxsize=1)
def get_cors_origins() -> List[str]:
    """Get CORS origins from environment (cached).

    Reads from CORS_ORIGINS env var (JSON array or comma-separated).

    Returns:
        List of allowed CORS origins
    """
    cors_env = os.environ.get("CORS_ORIGINS")

    if not cors_env:
        # Default origins
        return [
            "http://localhost:3001",
            "http://localhost:3002",
            "http://localhost:31245",
            "http://localhost:31246",
        ]

    # Try JSON first
    if cors_env.startswith("["):
        return get_env_json("CORS_ORIGINS")

    # Fall back to comma-separated
    return get_env_list("CORS_ORIGINS")


@lru_cache(maxsize=1)
def get_server_config() -> dict:
    """Get server configuration from environment (cached).

    Returns:
        Dict with host, port, and other server settings
    """
    return {
        "host": get_env("HOST", "0.0.0.0"),
        "port": get_env_int("PORT", 8787),
        "workers": get_env_int("WORKERS", 1),
        "debug": get_env_bool("DEBUG", default=False),
    }


@lru_cache(maxsize=1)
def get_llm_config() -> dict:
    """Get LLM configuration from environment (cached).

    Returns:
        Dict with LLM settings
    """
    return {
        "base_url": get_env_fallback("LLM_BASE_URL", "OPENAI_BASE_URL", default="http://localhost:18734/v1"),
        "api_key": get_env_fallback("LLM_API_KEY", "OPENAI_API_KEY", "Z_AI_API_KEY"),
        "model": get_env("LLM_MODEL", "qwen-omni-turbo"),
    }


def clear_env_cache() -> None:
    """Clear all cached environment configurations.

    Call this if environment variables change at runtime.
    """
    get_cors_origins.cache_clear()
    get_server_config.cache_clear()
    get_llm_config.cache_clear()


# =============================================================================
# Environment Detection
# =============================================================================

def is_testing() -> bool:
    """Check if running in test mode."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def is_development() -> bool:
    """Check if running in development mode."""
    return get_env("ENVIRONMENT", "development").lower() in ("development", "dev", "local")


def is_production() -> bool:
    """Check if running in production mode."""
    return get_env("ENVIRONMENT", "development").lower() in ("production", "prod")


def is_docker() -> bool:
    """Check if running inside Docker."""
    return os.path.exists("/.dockerenv") or bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


__all__ = [
    # Basic getters
    "get_env",
    "get_env_int",
    "get_env_float",
    "get_env_bool",
    "get_env_list",
    "get_env_json",
    "get_env_fallback",
    "require_env",
    # Cached helpers
    "get_cors_origins",
    "get_server_config",
    "get_llm_config",
    "clear_env_cache",
    # Environment detection
    "is_testing",
    "is_development",
    "is_production",
    "is_docker",
]
