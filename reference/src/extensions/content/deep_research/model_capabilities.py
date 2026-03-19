"""Model Capabilities Registry - Derive configuration from model characteristics.

This module provides model-specific capabilities that automatically derive
configuration values (like recursion limits, concurrency, and iteration counts)
from model characteristics instead of using hardcoded magic numbers.

Design Principles:
1. NO MAGIC NUMBERS - All derived from model capabilities
2. MODEL-SPECIFIC - Different models have different optimal settings
3. OVERRIDABLE - YAML config can override any derived value
4. PROVENANCE TRACKED - Each derived value logs its source for debugging

Usage:
    from src.extensions.content.deep_research.model_capabilities import (
        get_llm_concurrency,
        get_max_iterations,
        get_flash_recursion_limit,
    )

    # Auto-derived from model capabilities
    concurrency = get_llm_concurrency("glm-4.7")  # Returns: 5
    iterations = get_max_iterations("glm-4.7")  # Returns: 4
    recursion_limit = get_flash_recursion_limit("glm-4.7")  # Returns: 75
"""

from typing import Dict, Optional, Any
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from src.shared.model_context import get_model_max_tokens

from .logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])

# =============================================================================
# MODEL CAPABILITIES REGISTRY
# =============================================================================
# Each model has capabilities that determine optimal configuration values.
# These are derived from:
# - Context window size (affects recursion limit)
# - Reasoning capability (affects iteration count)
# - Rate limit handling (affects concurrency)
# - Speed/cost profile (affects recursion strategy)
# =============================================================================

class ModelCapabilities:
    """Model capabilities for deriving configuration values.

    Attributes:
        context_tier: Context window tier (tiny, small, medium, large, massive)
        reasoning_tier: Reasoning capability tier (basic, standard, advanced, expert)
        speed_tier: Speed tier (slow, medium, fast, instant)
        rate_limit_handling: Ability to handle concurrent requests (poor, fair, good, excellent)
        supports_thinking: Whether model supports thinking mode (for longer chains)
    """

    def __init__(
        self,
        context_tier: str = "medium",
        reasoning_tier: str = "standard",
        speed_tier: str = "medium",
        rate_limit_handling: str = "fair",
        supports_thinking: bool = False,
    ):
        self.context_tier = context_tier
        self.reasoning_tier = reasoning_tier
        self.speed_tier = speed_tier
        self.rate_limit_handling = rate_limit_handling
        self.supports_thinking = supports_thinking

    def derive_llm_concurrency(self) -> int:
        """Derive optimal LLM concurrency from rate limit handling capability.

        Models with better rate limit handling can sustain higher concurrency.
        """
        concurrency_map = {
            "poor": 1,
            "fair": 3,
            "good": 5,
            "excellent": 10,
        }
        return concurrency_map.get(self.rate_limit_handling, 3)

    def derive_max_iterations(self) -> int:
        """Derive optimal max iterations from reasoning capability.

        Models with stronger reasoning can benefit from more refinement iterations.
        Models with weaker reasoning should stop earlier to avoid degradation.
        """
        iterations_map = {
            "basic": 2,
            "standard": 3,
            "advanced": 4,
            "expert": 5,
        }
        base_iterations = iterations_map.get(self.reasoning_tier, 3)

        # Bonus for thinking-enabled models (can handle longer chains)
        if self.supports_thinking:
            base_iterations += 1

        return base_iterations

    def derive_flash_recursion_limit(self) -> int:
        """Derive optimal Flash Explorer recursion limit from context tier.

        Models with larger context windows can support deeper research chains.
        This is based on the formula:
        - Base: 20 LLM calls
        - Context multiplier: +0.1 per 100K tokens of context
        - Speed bonus: +10 for fast models
        """
        base_limit = 20

        # Context-based multiplier (per 100K tokens)
        context_multiplier_map = {
            "tiny": 1,      # < 10K tokens (old models)
            "small": 1.5,   # 10-32K tokens
            "medium": 2,    # 32-128K tokens
            "large": 3,     # 128-256K tokens
            "massive": 4,   # > 256K tokens (Gemini 2.5, etc.)
        }
        multiplier = context_multiplier_map.get(self.context_tier, 2)

        # Speed bonus (fast models can afford more exploration)
        speed_bonus_map = {
            "slow": 0,
            "medium": 5,
            "fast": 10,
            "instant": 15,
        }
        speed_bonus = speed_bonus_map.get(self.speed_tier, 5)

        return int(base_limit * multiplier) + speed_bonus


# Registry of known model capabilities
_MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    # =========================================================================
    # LiteLLM Model Names (PRIMARY - use these)
    # =========================================================================
    # ZAI Cloud models (via LiteLLM)
    "zai-glm-4-5-air": ModelCapabilities(
        context_tier="large",        # 128K tokens
        reasoning_tier="advanced",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=False,     # Air = fast, no thinking
    ),
    "zai-glm-4-7": ModelCapabilities(
        context_tier="large",        # ~202K tokens
        reasoning_tier="advanced",
        speed_tier="medium",
        rate_limit_handling="good",
        supports_thinking=True,
    ),
    "zai-glm-4-6": ModelCapabilities(
        context_tier="large",
        reasoning_tier="advanced",
        speed_tier="medium",
        rate_limit_handling="good",
        supports_thinking=True,
    ),
    "zai-glm-5": ModelCapabilities(
        context_tier="massive",      # 1M+ tokens
        reasoning_tier="expert",
        speed_tier="medium",
        rate_limit_handling="good",
        supports_thinking=True,
    ),

    # Local Qwen models (via llama.cpp)
    "qwen-35-27": ModelCapabilities(
        context_tier="large",        # 32K tokens
        reasoning_tier="advanced",
        speed_tier="medium",
        rate_limit_handling="good",  # Local, single-threaded
        supports_thinking=True,
    ),
    "qwen-35-27-fast": ModelCapabilities(
        context_tier="large",
        reasoning_tier="advanced",
        speed_tier="fast",           # No thinking = faster
        rate_limit_handling="good",
        supports_thinking=False,
    ),

    # =========================================================================
    # Legacy Model Names (for backward compatibility)
    # =========================================================================
    # Zhipu AI / GLM models
    "glm-4.7": ModelCapabilities(
        context_tier="large",       # ~202K tokens
        reasoning_tier="advanced",   # Strong reasoning
        speed_tier="medium",         # Local model, medium speed
        rate_limit_handling="good",  # Good rate limit handling
        supports_thinking=True,      # Supports thinking mode
    ),
    "glm-4-plus": ModelCapabilities(
        context_tier="large",
        reasoning_tier="expert",     # Best reasoning
        speed_tier="fast",           # Fast API
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "glm-4-air": ModelCapabilities(
        context_tier="large",
        reasoning_tier="advanced",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=False,
    ),
    "glm-4": ModelCapabilities(
        context_tier="medium",       # 128K tokens
        reasoning_tier="standard",
        speed_tier="medium",
        rate_limit_handling="good",
        supports_thinking=True,
    ),
    "glm-3-turbo": ModelCapabilities(
        context_tier="medium",
        reasoning_tier="standard",
        speed_tier="fast",
        rate_limit_handling="fair",
        supports_thinking=False,
    ),

    # OpenAI models
    "gpt-4o": ModelCapabilities(
        context_tier="medium",       # 128K tokens
        reasoning_tier="expert",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "gpt-4o-mini": ModelCapabilities(
        context_tier="medium",
        reasoning_tier="advanced",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=False,
    ),
    "gpt-4-turbo": ModelCapabilities(
        context_tier="medium",
        reasoning_tier="expert",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "gpt-4": ModelCapabilities(
        context_tier="tiny",         # 8K tokens
        reasoning_tier="expert",
        speed_tier="slow",
        rate_limit_handling="fair",
        supports_thinking=False,
    ),
    "gpt-3.5-turbo": ModelCapabilities(
        context_tier="small",        # 16K tokens
        reasoning_tier="standard",
        speed_tier="fast",
        rate_limit_handling="good",
        supports_thinking=False,
    ),

    # Anthropic Claude models
    "claude-opus-4-20250514": ModelCapabilities(
        context_tier="large",        # 200K tokens
        reasoning_tier="expert",
        speed_tier="medium",
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "claude-sonnet-4-20250514": ModelCapabilities(
        context_tier="large",
        reasoning_tier="expert",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "claude-3-5-sonnet-20241022": ModelCapabilities(
        context_tier="large",
        reasoning_tier="expert",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "claude-3-5-haiku-20241022": ModelCapabilities(
        context_tier="large",
        reasoning_tier="advanced",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=False,
    ),

    # Google models
    "gemini-2.5-flash": ModelCapabilities(
        context_tier="massive",      # 1M tokens!
        reasoning_tier="advanced",
        speed_tier="instant",        # Very fast
        rate_limit_handling="excellent",
        supports_thinking=False,
    ),
    "gemini-2.5-pro": ModelCapabilities(
        context_tier="massive",      # 1M tokens!
        reasoning_tier="expert",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "gemini-1.5-pro": ModelCapabilities(
        context_tier="massive",      # 2.8M tokens!
        reasoning_tier="expert",
        speed_tier="fast",
        rate_limit_handling="excellent",
        supports_thinking=True,
    ),
    "gemini-1.5-flash": ModelCapabilities(
        context_tier="massive",      # 2.8M tokens!
        reasoning_tier="advanced",
        speed_tier="instant",
        rate_limit_handling="excellent",
        supports_thinking=False,
    ),

    # DeepSeek models
    "deepseek-chat": ModelCapabilities(
        context_tier="medium",       # 128K tokens
        reasoning_tier="advanced",
        speed_tier="medium",
        rate_limit_handling="good",
        supports_thinking=True,
    ),
    "deepseek-coder": ModelCapabilities(
        context_tier="medium",
        reasoning_tier="advanced",
        speed_tier="medium",
        rate_limit_handling="good",
        supports_thinking=False,
    ),

    # Local llama.cpp models (conservative defaults)
    "llama.cpp": ModelCapabilities(
        context_tier="tiny",         # 2K tokens default
        reasoning_tier="basic",
        speed_tier="slow",
        rate_limit_handling="poor",  # Single-threaded
        supports_thinking=False,
    ),
}


def _get_model_capabilities(model_name: str) -> ModelCapabilities:
    """Get model capabilities from registry, with smart fallback.

    Args:
        model_name: The model name to look up

    Returns:
        ModelCapabilities object with derived settings
    """
    # Direct match
    if model_name in _MODEL_CAPABILITIES:
        return _MODEL_CAPABILITIES[model_name]

    # Case-insensitive match
    model_lower = model_name.lower()
    for known_model, capabilities in _MODEL_CAPABILITIES.items():
        if model_lower in known_model.lower() or known_model.lower() in model_lower:
            return capabilities

    # Fallback: Infer from context window
    max_tokens = get_model_max_tokens(model_name)

    # Infer tiers from context window
    if max_tokens >= 1000000:
        context_tier = "massive"
    elif max_tokens >= 200000:
        context_tier = "large"
    elif max_tokens >= 32000:
        context_tier = "medium"
    elif max_tokens >= 10000:
        context_tier = "small"
    else:
        context_tier = "tiny"

    # Default to standard capabilities for unknown models
    logger.warning(
        f"[MODEL_CAPABILITIES] Unknown model '{model_name}', inferring from context: {context_tier} ({max_tokens} tokens)"
    )

    return ModelCapabilities(
        context_tier=context_tier,
        reasoning_tier="standard",
        speed_tier="medium",
        rate_limit_handling="fair",
        supports_thinking=False,
    )


def get_llm_concurrency(
    model_name: str,
    yaml_config: Optional[Dict] = None,
) -> int:
    """Get optimal LLM concurrency for a model.

    This replaces the hardcoded `llm_concurrency: 3` magic number.

    Args:
        model_name: The model name (e.g., "glm-4.7", "gpt-4o")
        yaml_config: Optional YAML config for override

    Returns:
        Optimal concurrency value for the model

    Example:
        >>> get_llm_concurrency("glm-4.7")
        5
        >>> get_llm_concurrency("gpt-4o")
        10
    """
    # Check YAML override first
    if yaml_config:
        override = yaml_config.get("llm_concurrency")
        if override is not None:
            logger.info(f"[LLM_CONCURRENCY] Using YAML override: {override}")
            return override

    capabilities = _get_model_capabilities(model_name)
    concurrency = capabilities.derive_llm_concurrency()

    logger.info(
        f"[LLM_CONCURRENCY] Derived for {model_name}: {concurrency} "
        f"(rate_limit_handling: {capabilities.rate_limit_handling})"
    )

    return concurrency


def get_max_iterations(
    model_name: str,
    yaml_config: Optional[Dict] = None,
) -> int:
    """Get optimal max iterations for a model.

    This replaces the hardcoded `max_iterations: 3` magic number.

    Args:
        model_name: The model name (e.g., "glm-4.7", "gpt-4o")
        yaml_config: Optional YAML config for override

    Returns:
        Optimal max iterations value for the model

    Example:
        >>> get_max_iterations("glm-4.7")
        5  # Advanced reasoning + thinking support
        >>> get_max_iterations("gpt-4o")
        6  # Expert reasoning + thinking support
    """
    # Check YAML override first
    if yaml_config:
        override = yaml_config.get("max_iterations")
        if override is not None:
            logger.info(f"[MAX_ITERATIONS] Using YAML override: {override}")
            return override

    # Check spawner_config override
    if yaml_config:
        spawner_config = yaml_config.get("spawner_config", {})
        override = spawner_config.get("max_iterations")
        if override is not None:
            logger.info(f"[MAX_ITERATIONS] Using spawner_config override: {override}")
            return override

    capabilities = _get_model_capabilities(model_name)
    iterations = capabilities.derive_max_iterations()

    logger.info(
        f"[MAX_ITERATIONS] Derived for {model_name}: {iterations} "
        f"(reasoning: {capabilities.reasoning_tier}, thinking: {capabilities.supports_thinking})"
    )

    return iterations


def get_flash_recursion_limit(
    model_name: str,
    yaml_config: Optional[Dict] = None,
) -> int:
    """Get optimal Flash Explorer recursion limit for a model.

    This replaces the hardcoded `flash_recursion_limit: 50` magic number.

    Args:
        model_name: The model name (e.g., "glm-4.7", "gpt-4o")
        yaml_config: Optional YAML config for override

    Returns:
        Optimal recursion limit for the model

    Example:
        >>> get_flash_recursion_limit("glm-4.7")
        75  # Large context (202K) * multiplier + speed bonus
        >>> get_flash_recursion_limit("gemini-2.5-flash")
        95  # Massive context (1M) + instant speed
    """
    # Check YAML override first
    if yaml_config:
        override = yaml_config.get("flash_recursion_limit")
        if override is not None:
            logger.info(f"[FLASH_RECURSION] Using YAML override: {override}")
            return override

    # Check spawner_config override
    if yaml_config:
        spawner_config = yaml_config.get("spawner_config", {})
        override = spawner_config.get("flash_recursion_limit")
        if override is not None:
            logger.info(f"[FLASH_RECURSION] Using spawner_config override: {override}")
            return override

    capabilities = _get_model_capabilities(model_name)
    limit = capabilities.derive_flash_recursion_limit()

    logger.info(
        f"[FLASH_RECURSION] Derived for {model_name}: {limit} "
        f"(context: {capabilities.context_tier}, speed: {capabilities.speed_tier})"
    )

    return limit


def get_all_derived_config(
    model_name: str,
    yaml_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Get all derived configuration values for a model.

    This is useful for logging and debugging to see all derived values at once.

    Args:
        model_name: The model name (e.g., "glm-4.7", "gpt-4o")
        yaml_config: Optional YAML config for overrides

    Returns:
        Dictionary with all derived config values

    Example:
        >>> get_all_derived_config("glm-4.7")
        {
            "llm_concurrency": 5,
            "max_iterations": 5,
            "flash_recursion_limit": 75,
            "model_capabilities": {...}
        }
    """
    return {
        "llm_concurrency": get_llm_concurrency(model_name, yaml_config),
        "max_iterations": get_max_iterations(model_name, yaml_config),
        "flash_recursion_limit": get_flash_recursion_limit(model_name, yaml_config),
        "model_capabilities": _get_model_capabilities(model_name).__dict__,
    }
