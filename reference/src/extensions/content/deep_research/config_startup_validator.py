"""Startup Configuration Validator - Called at Extension Load.

CRASH-ONLY: This module validates the ENTIRE config at startup.
If anything is wrong, the extension will NOT load.

This catches configuration errors BEFORE any research runs.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Set

from .config_schema_strict import (
    ConfigValidationError,
    validate_config_crash,
    validate_all_prompts,
    validate_prompt_variable_mapping,
    REQUIRED_PROMPTS,
    OPTIONAL_TEMPLATE_VARS,
)
from .logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])


# =============================================================================
# CODE-TO-TEMPLATE VARIABLE MAPPING
# =============================================================================
# This maps the variable names passed by code to the expected template variables
# If there's a mismatch, we CRASH at startup (not at runtime)

# Maps template name -> {code_var: template_var}
CODE_TO_TEMPLATE_VAR_MAP: Dict[str, Dict[str, str]] = {
    # flash_explorer.planning: code passes 'research_avenue', template expects 'input_data'
    "flash_explorer.planning": {
        "research_avenue": "input_data",  # CRITICAL FIX: Map research_avenue -> input_data
    },
}

# Templates that have ALTERNATE variable names (code can pass either)
TEMPLATE_ALTERNATE_VARS: Dict[str, Dict[str, Set[str]]] = {
    "flash_explorer.planning": {
        "input_data": {"input_data", "research_avenue"},  # Accept either
    },
}


def validate_code_template_mapping() -> None:
    """Validate that all code-to-template variable mappings are correct.

    This catches the bug where:
    - Template expects {{ input_data }}
    - Code passes research_avenue=...
    """
    for template_name, code_var_map in CODE_TO_TEMPLATE_VAR_MAP.items():
        required_vars = REQUIRED_PROMPTS.get(template_name, set())
        optional_vars = OPTIONAL_TEMPLATE_VARS.get(template_name, set())
        all_allowed = required_vars | optional_vars

        # Check that mapped template vars exist in template
        for code_var, template_var in code_var_map.items():
            if template_var not in all_allowed:
                raise ConfigValidationError(
                    f"Code-to-template mapping error for '{template_name}':\n"
                    f"  Code variable '{code_var}' maps to '{template_var}'\n"
                    f"  But '{template_var}' is not a known template variable.\n"
                    f"  Known variables: {sorted(all_allowed)}",
                    f"mapping.{template_name}"
                )


def get_canonical_var_name(template_name: str, code_var: str) -> str:
    """Get the canonical template variable name for a code variable.

    This handles the case where code uses different names than templates.
    """
    if template_name in CODE_TO_TEMPLATE_VAR_MAP:
        if code_var in CODE_TO_TEMPLATE_VAR_MAP[template_name]:
            return CODE_TO_TEMPLATE_VAR_MAP[template_name][code_var]
    return code_var


# =============================================================================
# STARTUP VALIDATION
# =============================================================================

_validated_configs: Dict[str, Any] = {}


def validate_at_startup(
    config: Dict[str, Any],
    prompts: Dict[str, Any],
    config_name: str = "config.yaml",
) -> None:
    """Validate config and prompts at startup - CRASH if invalid.

    This should be called when the extension loads, BEFORE any research runs.

    Args:
        config: The merged config dictionary
        prompts: The prompts dictionary (from base.yaml + overrides)
        config_name: Name of the config for error messages

    Raises:
        ConfigValidationError: If validation fails (CRASH)
    """
    import traceback

    try:
        # 1. Validate main config structure
        validated = validate_config_crash(config)
        _validated_configs[config_name] = validated

        # 2. Merge prompts into config for validation
        merged_prompts = {**(config.get("prompts", {}) or {}), **prompts}

        # 3. Validate all prompts
        validate_all_prompts(merged_prompts)

        # 4. Validate code-to-template mappings
        validate_code_template_mapping()

        logger.info(f"[CONFIG] Validated {config_name} successfully")

    except ConfigValidationError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Wrap other errors
        raise ConfigValidationError(
            f"Unexpected validation error: {type(e).__name__}: {e}\n\n"
            f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}",
            config_name
        ) from e


def validate_prompt_call(
    template_name: str,
    passed_vars: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate variables passed to a prompt template at call time.

    This translates code variable names to template variable names
    and validates that all required variables are present.

    Args:
        template_name: Name of the template being called
        passed_vars: Variables passed by the code

    Returns:
        Translated variables dict for template rendering

    Raises:
        ConfigValidationError: If required variables are missing
    """
    required_vars = REQUIRED_PROMPTS.get(template_name, set())
    optional_vars = OPTIONAL_TEMPLATE_VARS.get(template_name, set())
    alternates = TEMPLATE_ALTERNATE_VARS.get(template_name, {})

    # Translate variable names
    translated = {}
    for code_var, value in passed_vars.items():
        canonical = get_canonical_var_name(template_name, code_var)
        translated[canonical] = value

        # Also set alternate names if applicable
        if canonical in alternates:
            for alt in alternates[canonical]:
                translated[alt] = value

    # Check for missing required variables
    missing = []
    for var in required_vars:
        if var not in translated:
            # Check alternates
            if var in alternates:
                if not any(alt in translated for alt in alternates[var]):
                    missing.append(var)
            else:
                missing.append(var)

    if missing:
        raise ConfigValidationError(
            f"Template '{template_name}' called without required variables: {missing}\n"
            f"Passed variables: {list(passed_vars.keys())}\n"
            f"Required: {sorted(required_vars)}\n"
            f"Optional: {sorted(optional_vars)}",
            f"prompts.{template_name}"
        )

    return translated


# =============================================================================
# DECORATOR FOR SAFE PROMPT CALLS
# =============================================================================

def validated_prompt_call(template_name: str):
    """Decorator that validates prompt template calls.

    Usage:
        @validated_prompt_call("flash_explorer.planning")
        def get_planning_prompt(research_avenue: str, **kwargs) -> str:
            # This will validate that research_avenue maps to input_data
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            translated = validate_prompt_call(template_name, kwargs)
            return func(*args, **{**kwargs, **translated})
        return wrapper
    return decorator


# =============================================================================
# INTEGRATION WITH EXISTING SYSTEM
# =============================================================================

def check_config_on_import() -> None:
    """Check config validity when this module is imported.

    This provides early detection of config issues.
    """
    # This will be called when the extension loads
    pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "validate_at_startup",
    "validate_prompt_call",
    "validate_code_template_mapping",
    "validated_prompt_call",
    "get_canonical_var_name",
    "CODE_TO_TEMPLATE_VAR_MAP",
    "TEMPLATE_ALTERNATE_VARS",
]
