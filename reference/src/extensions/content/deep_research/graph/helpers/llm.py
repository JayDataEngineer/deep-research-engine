"""LLM creation functions for deep research graph.

Simplified for LiteLLM:
- All models are LiteLLM model names (e.g., "zai-glm-4-5-air", "qwen-35-27")
- Thinking mode is handled by model name selection (qwen-35-27 vs qwen-35-27-fast)
- No provider parameter needed - LiteLLM handles routing
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=3)
from ...logging_utils import get_logger

from .config import _LLM_MODEL_NAME

logger = get_logger(__name__, __name__.split('.')[-1])


def _expand_env_vars(value: Any) -> Any:
    """Expand environment variables in configuration values.

    Supports ${VAR:-default} and ${VAR-default} syntax.
    """
    if isinstance(value, str):
        pattern = r'\$\{([^}:-]+)(?::-?|-)([^}]*)\}'

        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.getenv(var_name, default_value)

        return re.sub(pattern, replace_env_var, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _auto_route_role(role: str) -> str:
    """Auto-route role to appropriate model slot.

    Pro-Flash Hybrid Architecture:
    - Boss/Architect/Synthesis/Critic → MAIN model (with thinking)
    - Worker/Researcher → TOOL model (fast, no thinking)
    """
    if role in ("worker", "researcher"):
        return "tool"
    return "main"


def _get_default_thinking_for_slot(model_slot: str) -> bool:
    """Get default thinking mode based on model slot.

    - 'main' slot → thinking ON (complex reasoning)
    - 'tool' slot → thinking OFF (fast operations)
    """
    tool_slots = {"tool", "local", "flash"}
    return model_slot not in tool_slots


def _resolve_model_name(yaml_config: Dict[str, Any], role: str) -> str:
    """Resolve LiteLLM model name from config.

    Config format (simplified for LiteLLM):
        models:
          main: zai-glm-4-5-air      # Direct model name
          tool: qwen-35-27-fast      # Direct model name

    Legacy format (still supported):
        models:
          main:
            provider: zai
            model: glm-4.5

    Args:
        yaml_config: Full YAML config dict
        role: Role (spawner, researcher, critic, synthesis, worker)

    Returns:
        LiteLLM model name string
    """
    yaml_config = _expand_env_vars(yaml_config)
    models = yaml_config.get("models", {})
    model_configs = yaml_config.get("model_configs", {})

    # Determine model slot from role
    role_key = role
    if role == "worker" and role not in model_configs:
        role_key = "researcher"

    # Check if role has explicit model_slot
    model_slot = None
    if role_key in model_configs:
        role_config = model_configs[role_key]
        model_slot = role_config.get("model_slot")

    # Auto-route if no explicit slot
    if not model_slot:
        model_slot = _auto_route_role(role)

    # Get model from slot
    if model_slot not in models:
        # Try aliases
        slot_aliases = {"local": "tool", "tool": "local"}
        model_slot = slot_aliases.get(model_slot, model_slot)

    if model_slot not in models:
        raise ValueError(
            f"Model slot '{model_slot}' not found in config.\n"
            f"Available slots: {list(models.keys())}\n"
            f"Add to your config:\n"
            f"  models:\n"
            f"    {model_slot}: zai-glm-4-5-air"
        )

    slot_config = models[model_slot]

    # Handle string (direct model name) - new format
    if isinstance(slot_config, str):
        return slot_config

    # Handle dict (legacy format with provider/model) - DEPRECATED
    if isinstance(slot_config, dict):
        if "model" in slot_config:
            model_name = slot_config["model"]
            # Legacy format detected - warn user
            provider = slot_config.get("provider")
            if provider:
                logger.warning(
                    f"Legacy model config format detected: provider={provider}, model={model_name}. "
                    f"Please update to direct model name: models:\n  {model_slot}: zai-glm-4-5-air"
                )
                # Try to convert, but this is deprecated
                return _convert_to_litellm_name_legacy(model_name, provider)
            return model_name
        elif "model_name" in slot_config:
            # Very old format
            return slot_config["model_name"]

    raise ValueError(
        f"Invalid model slot config: {slot_config}\n"
        f"Expected string (model name) or dict with 'model' field."
    )


def _convert_to_litellm_name_legacy(model: str, provider: str) -> str:
    """DEPRECATED: Convert legacy model+provider to LiteLLM model name.

    This function is kept for backward compatibility only.
    New configs should use direct LiteLLM model names.

    Examples:
        glm-4.5 + zai -> zai-glm-4-5-air
        GLM-4.7-Flash + llamacpp -> qwen-35-27
    """
    # Normalize model name
    model_lower = model.lower().replace(".", "-")

    # Provider-specific mappings (deprecated)
    if provider == "zai":
        if "glm-4-5" in model_lower or "glm-4.5" in model_lower:
            return "zai-glm-4-5-air"
        if "glm-4-7" in model_lower or "glm-4.7" in model_lower:
            return "zai-glm-4-7"
        if "glm-5" in model_lower:
            return "zai-glm-5"
        return f"zai-{model_lower}"

    elif provider in ("llamacpp", "localhost", "local"):
        if "glm" in model_lower or "qwen" in model_lower:
            return "qwen-35-27"
        return f"qwen-35-27"  # Default local model

    # Unknown provider - return as-is
    return model_lower


def _resolve_model_from_slot(yaml_config: Dict[str, Any], model_slot: str, role: str) -> Dict[str, Any]:
    """Legacy function for backward compatibility.

    Returns dict with 'model' and 'provider' keys.
    """
    model_name = _resolve_model_name(yaml_config, role)
    return {"model": model_name, "provider": None}


# =============================================================================
# LLM CREATION
# =============================================================================

async def create_llm_from_config(
    yaml_config: Dict[str, Any],
    role: str = "spawner",
    enable_thinking: Optional[bool] = None,
):
    """Create LLM from YAML config using LiteLLM factory.

    All models are routed through LiteLLM proxy.
    Model names should be LiteLLM format (e.g., "zai-glm-4-5-air", "qwen-35-27").

    Args:
        yaml_config: Full YAML config dict
        role: Role (spawner, researcher, critic, synthesis, worker)
        enable_thinking: Override thinking mode (None = use slot-based default)

    Returns:
        BaseLLM instance from LiteLLM factory
    """
    from src.shared import get_llm

    # Resolve model name
    model_name = _resolve_model_name(yaml_config, role)

    # Get role-specific settings
    model_slot = _auto_route_role(role)
    model_configs = yaml_config.get("model_configs", {})

    # Map worker to researcher config if needed
    role_key = role
    if role in ("worker", "tool_binding") and role not in model_configs:
        role_key = "researcher"

    # Get settings from config or use defaults
    if role_key in model_configs:
        role_config = model_configs[role_key]
        temperature = role_config.get("temperature", 0.6)
        max_tokens = role_config.get("max_tokens")
        if enable_thinking is None:
            enable_thinking = role_config.get("enable_thinking", _get_default_thinking_for_slot(model_slot))
    else:
        temperature = 0.6
        max_tokens = None
        if enable_thinking is None:
            enable_thinking = _get_default_thinking_for_slot(model_slot)

    logger.info(
        f"[DEEP_RESEARCH_LLM] role={role}, slot={model_slot}, "
        f"model={model_name}, enable_thinking={enable_thinking}"
    )

    # Create LLM via LiteLLM factory
    llm = await get_llm(
        model=model_name,
        temperature=temperature,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
    )

    return llm
