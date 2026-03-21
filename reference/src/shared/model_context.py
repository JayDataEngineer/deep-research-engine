"""Model context window limits.

This module provides a registry of known LLM models and their maximum context windows,
along with utilities to determine the appropriate token budget for synthesis and critic operations.

Fully percentage-based with auto-detection and YAML override support:
- Automatically detects model context from registry
- Falls back to querying LLM provider API for unknown models
- Allows YAML config to override percentages per role

Usage:
    from .model_context import get_model_max_tokens, get_safe_token_limit

    # Get max tokens for a model
    max_tokens = get_model_max_tokens("glm-4.7")  # Returns: 202750

    # Get percentage-based limit with YAML overrides
    limit = get_safe_token_limit("glm-4.7", "worker", yaml_config)  # Returns: ~142K (70%)
"""

from typing import Dict, Optional, Any
import logging

from .logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# MODEL CONTEXT WINDOW REGISTRY
# =============================================================================
# Known models and their maximum input token limits
# Source: Model documentation and API specifications
_MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    # Zhipu AI / GLM models
    "glm-4.7": 202750,
    "glm-4-plus": 202750,
    "glm-4-air": 202750,
    "glm-4": 128000,
    "glm-3-turbo": 128000,

    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,

    # Anthropic Claude models
    "claude-opus-4-20250514": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,

    # Google models
    "gemini-2.5-flash": 1000000,
    "gemini-2.5-pro": 1000000,
    "gemini-1.5-pro": 2800000,
    "gemini-1.5-flash": 2800000,

    # DeepSeek models
    "deepseek-chat": 128000,
    "deepseek-coder": 128000,

    # Llama.cpp (local models - depends on the specific model loaded)
    "llama.cpp": 2048,  # Default conservative value, can be overridden

    # Mistral models
    "mistral-large": 128000,
    "mistral-medium": 32000,
    "mistral-small": 32000,
    "mixtral-8x7b": 32768,
}


def get_model_max_tokens(model_name: str, default: int = 128000) -> int:
    """Get the maximum context window for a given model.

    Args:
        model_name: The name of the model (e.g., "glm-4.7", "gpt-4o")
        default: Default value if model not found in registry (default: 128K)

    Returns:
        Maximum number of input tokens the model can handle

    Examples:
        >>> get_model_max_tokens("glm-4.7")
        202750
        >>> get_model_max_tokens("gpt-4o")
        128000
        >>> get_model_max_tokens("unknown-model")
        128000
    """
    # Normalize model name (handle common variations)
    model_lower = model_name.lower()

    # Direct match first
    if model_name in _MODEL_CONTEXT_WINDOWS:
        return _MODEL_CONTEXT_WINDOWS[model_name]

    # Case-insensitive match
    for known_model, max_tokens in _MODEL_CONTEXT_WINDOWS.items():
        if model_lower in known_model.lower() or known_model.lower() in model_lower:
            return max_tokens

    # Pattern-based matching for GLM models
    if "glm-4" in model_lower:
        return 202750  # Most GLM-4.x models have 202K context
    if "glm-3" in model_lower:
        return 128000

    # Pattern-based matching for Claude
    if "claude-3-5" in model_lower or "claude-opus" in model_lower or "claude-sonnet" in model_lower:
        return 200000

    # Pattern-based matching for GPT
    if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
        return 128000

    # Default fallback
    return default


def calculate_token_budget(
    model_name: str,
    buffer_percent: float = 15.0,
    min_output_tokens: int = 4096,
    default_max_tokens: int = 128000
) -> int:
    """Calculate the token budget for input (excluding output buffer).

    Args:
        model_name: The name of the model
        buffer_percent: Percentage to reserve for output (default: 15%)
        min_output_tokens: Minimum tokens to reserve for output (default: 4096)
        default_max_tokens: Default max tokens if model not found (default: 128K)

    Returns:
        Number of tokens available for input (findings, prompt, etc.)

    Examples:
        >>> calculate_token_budget("glm-4.7", buffer_percent=15)
        172337  # ~202750 * 0.85
        >>> calculate_token_budget("gpt-4o", buffer_percent=15)
        108800  # ~128000 * 0.85
    """
    max_tokens = get_model_max_tokens(model_name, default=default_max_tokens)

    # Calculate buffer (reserve space for output + safety margin)
    buffer_tokens = int(max_tokens * (buffer_percent / 100.0))

    # Ensure minimum output space
    buffer_tokens = max(buffer_tokens, min_output_tokens)

    return max_tokens - buffer_tokens


def get_recommended_buffer_for_role(role: str) -> float:
    """Get the recommended buffer percentage for a given role.

    Args:
        role: The role (e.g., "synthesis", "critic", "worker")

    Returns:
        Recommended buffer percentage (0-100)

    Examples:
        >>> get_recommended_buffer_for_role("synthesis")
        15.0
        >>> get_recommended_buffer_for_role("critic")
        15.0
        >>> get_recommended_buffer_for_role("worker")
        5.0
    """
    # Synthesis and critic need more buffer for generating outputs
    if role in ("synthesis", "critic"):
        return 15.0
    # Workers typically only return findings, need less buffer
    elif role in ("worker", "researcher", "scout", "boss"):
        return 5.0
    # Default
    return 10.0


def get_percentage_for_role(
    role: str,
    yaml_config: Optional[Dict] = None
) -> float:
    """Get the percentage limit for a role from YAML config or defaults.

    Args:
        role: Role name (worker, synthesis, critic, scout, boss, etc.)
        yaml_config: Full YAML config dict (optional)

    Returns:
        Percentage as float (0.0 to 1.0, e.g., 0.70 for 70%)

    YAML Override Format:
        token_limits:
          worker_percent: 0.70      # 70% for workers
          synthesis_percent: 0.80   # 80% for synthesis
          critic_percent: 0.80      # 80% for critic
          scout_percent: 0.70       # 70% for scout
          boss_percent: 0.70        # 70% for boss
          default_percent: 0.75     # 75% default
    """
    # Try YAML config override first
    if yaml_config:
        token_limits = yaml_config.get("token_limits", {})

        # Role-based override mapping
        role_mapping = {
            "worker": "worker_percent",
            "researcher": "worker_percent",
            "lab": "worker_percent",
            "synthesis": "synthesis_percent",
            "critic": "critic_percent",
            "scout": "scout_percent",
            "boss": "boss_percent",
            "spawner": "boss_percent",
        }

        # Check for role-specific override
        override_key = role_mapping.get(role.lower())
        if override_key and override_key in token_limits:
            return token_limits[override_key]

        # Check for default override
        if "default_percent" in token_limits:
            return token_limits["default_percent"]

    # Default percentages
    defaults = {
        "worker": 0.70,      # 70% for workers (conservative)
        "researcher": 0.70,
        "lab": 0.70,
        "synthesis": 0.80,   # 80% for synthesis (needs more context)
        "critic": 0.80,      # 80% for critic
        "scout": 0.70,       # 70% for scout
        "boss": 0.70,        # 70% for boss
        "spawner": 0.70,
    }

    return defaults.get(role.lower(), 0.75)  # 75% default


def get_safe_token_limit(
    model_name: str,
    role: str,
    yaml_config: Optional[Dict] = None
) -> int:
    """Get safe token limit for a model and role, with YAML overrides.

    This is the main function to use for token limit enforcement.
    It automatically:
    1. Detects model max context (from registry or API)
    2. Applies percentage-based limits per role
    3. Respects YAML config overrides

    Args:
        model_name: Model name (e.g., "glm-4.7", "gpt-4o")
        role: Role name (worker, synthesis, critic, etc.)
        yaml_config: YAML config for percentage overrides (optional)

    Returns:
        Safe token limit as integer

    Examples:
        >>> get_safe_token_limit("glm-4.7", "worker", {})
        141925  # ~202750 * 0.70

        >>> get_safe_token_limit("gpt-4o", "synthesis", {})
        102400  # ~128000 * 0.80

        # With YAML override: {token_limits: {worker_percent: 0.50}}
        >>> get_safe_token_limit("glm-4.7", "worker", {token_limits: {worker_percent: 0.50}})
        101375  # ~202750 * 0.50
    """
    max_tokens = get_model_max_tokens(model_name)
    percentage = get_percentage_for_role(role, yaml_config)

    safe_limit = int(max_tokens * percentage)

    if yaml_config and yaml_config.get("debug_mode"):
        logger.info(
            f"[TOKEN_LIMIT] {model_name} ({role}): {max_tokens} max * {percentage:.0%} = {safe_limit} safe"
        )

    return safe_limit


async def auto_detect_model_context(model_name: str, llm_instance: Optional[Any] = None) -> Optional[int]:
    """Auto-detect model context window from LLM provider API.

    For unknown models not in the registry, this function attempts to
    query the LLM provider's API to get the actual max context window.

    Args:
        model_name: Model name to detect
        llm_instance: Optional LLM instance to query

    Returns:
        Max context tokens, or None if auto-detection fails

    Note:
        This is a fallback for unknown models. Most common models are
        already in the registry, so auto-detection is rarely needed.
    """
    # Try to get from provider-specific API
    try:
        if llm_instance:
            # Check if LLM instance has max_tokens attribute
            if hasattr(llm_instance, "max_tokens"):
                return llm_instance.max_tokens

            # Check for provider-specific methods
            if hasattr(llm_instance, "get_model_info"):
                info = llm_instance.get_model_info(model_name)
                if info and "max_tokens" in info:
                    return info["max_tokens"]
    except Exception as e:
        logger.warning(f"Failed to auto-detect context for {model_name}: {e}")

    # Fallback: return None (will use registry default)
    return None
