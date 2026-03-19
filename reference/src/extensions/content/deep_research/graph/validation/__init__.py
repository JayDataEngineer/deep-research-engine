"""Validation utilities for deep research graph."""

from typing import Dict, Any
from .config_validation import validate_variant_config, validate_required_config_sections


def validate_all(config: Dict[str, Any], skip_deep_agent_check: bool = False) -> None:
    """Run all validation checks on a configuration.

    Args:
        config: The loaded YAML configuration dict
        skip_deep_agent_check: Whether to skip deep agent validation

    Raises:
        Various validation exceptions if configuration is invalid
    """
    # Validate required config sections exist
    validate_required_config_sections(config)

    # Validate variant configuration structure
    validate_variant_config(config)


__all__ = [
    "validate_all",
    "validate_variant_config",
    "validate_required_config_sections",
]
