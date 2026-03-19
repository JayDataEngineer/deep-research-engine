"""Extension Configuration Module

Provides schema validation for extension config.yaml files.
"""

from .schema import (
    ExtensionConfigSchema,
    ExtensionConfigModel,
    CapabilitiesConfig,
    UIConfig,
    DependenciesConfig,
    validate_config,
    get_missing_fields,
)

__all__ = [
    "ExtensionConfigSchema",
    "ExtensionConfigModel",
    "CapabilitiesConfig",
    "UIConfig",
    "DependenciesConfig",
    "validate_config",
    "get_missing_fields",
]
