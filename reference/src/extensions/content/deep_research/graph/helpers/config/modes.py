"""Mode transformation functions for deep research config.

This module provides transformations for special research modes
like overdrive mode (max quality, max context, deeper research).
"""

import copy
from typing import Dict


# =============================================================================
# OVERDRIVE MODE TRANSFORMATION
# =============================================================================

def _apply_overdrive_mode(config: Dict) -> Dict:
    """Apply overdrive mode - max quality, max context, deeper research.

    Overdrive mode overrides:
    - max_researchers: 5 (more parallel researchers)
    - max_iterations: 5 (deeper iteration)
    - quality_threshold: 0.9 (higher quality bar)
    - completeness_threshold: 0.95 (more thorough)
    - level_of_detail: 10 (exhaustive detail)
    - max_tokens: maximum for all models (longer outputs)
    - Adds overdrive prompt modifier to all phases

    Args:
        config: Base config from _load_yaml_config

    Returns:
        Config with overdrive overrides applied
    """
    overdrive_config = copy.deepcopy(config)

    # Spawner config overdrive settings
    if "spawner_config" not in overdrive_config:
        overdrive_config["spawner_config"] = {}
    spawner = overdrive_config["spawner_config"]

    spawner["max_researchers"] = 5
    spawner["max_iterations"] = 5
    spawner["quality_threshold"] = 0.9
    spawner["completeness_threshold"] = 0.95
    spawner["level_of_detail"] = 10

    # Max tokens for all model configs
    if "model_configs" in overdrive_config:
        for role in ["spawner", "researcher", "critic", "synthesis", "worker"]:
            if role in overdrive_config["model_configs"]:
                # Set max_tokens to maximum for the model
                role_config = overdrive_config["model_configs"][role]
                model_name = role_config.get("model_name", "glm-4.7")

                # Maximum output tokens by model
                MAX_OUTPUT_TOKENS = {
                    "glm-4.7": 128000,
                    "gpt-4o": 16384,
                    "gpt-4o-mini": 16384,
                    "gpt-4-turbo": 4096,
                    "gpt-4": 8192,
                    "gpt-3.5-turbo": 4096,
                    "claude-3-5-sonnet": 8192,
                    "claude-3-5-haiku": 8192,
                    "claude-3-opus": 4096,
                }
                role_config["max_tokens"] = MAX_OUTPUT_TOKENS.get(model_name, 128000)

    # Add overdrive prompt modifier from config
    overdrive_prompt = overdrive_config.get("internal_prompts", {}).get("overdrive_modifier", "")

    # Add overdrive modifier to all phase prompts
    if "prompts" in overdrive_config:
        for phase, prompt in overdrive_config["prompts"].items():
            if isinstance(prompt, str):
                overdrive_config["prompts"][phase] = prompt + "\n" + overdrive_prompt

    return overdrive_config
