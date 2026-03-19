"""Prompt management utilities for the deep research graph.

This module provides prompt utilities that delegate to the StrictPromptRegistry
in prompt_registry.py for multi-layer validation.

CRASH-ONLY PHILOSOPHY:
This module follows crash-only semantics - if a required prompt is missing,
the system crashes immediately with a clear error message. NO SILENT FALLBACKS.

All prompts must be defined in config files (prompts/base.yaml or use_case configs).
"""

import copy
import yaml
from pathlib import Path
from typing import Dict, Any

from .prompt_registry import get_prompt as registry_get_prompt


def _apply_prompt_extensions(base_prompts: Dict[str, Any], use_case_prompts: Dict[str, Any]) -> Dict[str, Any]:
    """Apply use_case prompt extensions to base prompts.

    This merges use_case prompts into base prompts, with use_case values
    overriding base values for matching keys.

    SPECIAL CASE: If base is a string (prompt template) and use_case is a dict (config),
    we KEEP the base template and store the config under a special key for later access.
    This allows use_case configs to provide synthesis_role, citation_instructions, etc.
    without replacing the actual prompt template.

    Args:
        base_prompts: Base prompt templates
        use_case_prompts: Use case specific prompt extensions

    Returns:
        Merged prompts dictionary
    """
    # Deep copy to avoid modifying original
    merged = copy.deepcopy(base_prompts)

    # Apply use_case extensions (overlay)
    for key, value in use_case_prompts.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Recursively merge nested dicts
            merged[key] = {**merged[key], **value}
        elif isinstance(value, dict) and key in merged and isinstance(merged[key], str):
            # SPECIAL CASE: base is a string template, use_case is a dict config
            # Keep the base template and store config under _{key}_config key
            # This allows configs to provide metadata without replacing the template
            merged[f"_{key}_config"] = value
            # Keep the original base template (don't override with dict)
        else:
            # Direct override
            merged[key] = value

    return merged


def get_prompt_from_config(config: Dict[str, Any], prompt_type: str, **kwargs) -> str:
    """Get a prompt from the configuration. CRASH if not found.

    Delegates to StrictPromptRegistry for multi-layer validation:
    - Existence check
    - Jinja2 syntax validation
    - Variable contract enforcement
    - Preflight rendering

    Args:
        config: Configuration dictionary containing prompts
        prompt_type: Type of prompt to retrieve (e.g., 'worker', 'boss.guidance')
        **kwargs: Variables to render into the prompt using Jinja2

    Returns:
        Rendered prompt string

    Raises:
        RuntimeError: If prompt is missing, contract violated, or render fails
    """
    return registry_get_prompt(config, prompt_type, **kwargs)


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    """Load prompts from a YAML file.

    Args:
        file_path: Path to the YAML file containing prompts

    Returns:
        Dictionary mapping prompt names to their templates
    """
    path = Path(file_path)
    if not path.exists():
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
            return data.get('prompts', {})
        except yaml.YAMLError:
            return {}
