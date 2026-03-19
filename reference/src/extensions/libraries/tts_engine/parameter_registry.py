"""
Parameter handling system for TTS extensions.

This module provides a standardized system for defining, validating,
and handling parameters across TTS extensions, inspired by ComfyUI's
type system.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.extensions.libraries.tts_engine.api.schemas.tts import ExtensionParameter

logger = logging.getLogger(__name__)


class ParameterTypes:
    """Standard parameter types with validation rules."""

    @staticmethod
    def text(
        name: str,
        default: str = "",
        multiline: bool = False,
        placeholder: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """Create a text parameter definition.

        Args:
            name: Parameter name
            default: Default value
            multiline: Whether to use multiline input
            placeholder: Placeholder text
            **kwargs: Additional properties

        Returns:
            Parameter definition dictionary
        """
        return {
            "type": "text",
            "default": default,
            "multiline": multiline,
            "placeholder": placeholder,
            **kwargs,
        }

    @staticmethod
    def number(
        name: str,
        default: int | float,
        min_val: int | float | None = None,
        max_val: int | float | None = None,
        step: int | float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a number parameter definition.

        Args:
            name: Parameter name
            default: Default value
            min_val: Minimum value
            max_val: Maximum value
            step: Step value for increment/decrement
            **kwargs: Additional properties

        Returns:
            Parameter definition dictionary
        """
        return {
            "type": "number",
            "default": default,
            "min_value": min_val,
            "max_value": max_val,
            "step": step,
            **kwargs,
        }

    @staticmethod
    def float(
        name: str,
        default: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.01,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a float parameter definition.

        Args:
            name: Parameter name
            default: Default value
            min_val: Minimum value
            max_val: Maximum value
            step: Step value
            **kwargs: Additional properties

        Returns:
            Parameter definition dictionary
        """
        return ParameterTypes.number(
            name=name,
            default=default,
            min_val=min_val,
            max_val=max_val,
            step=step,
            **kwargs,
        )

    @staticmethod
    def range(
        name: str,
        default: int | float,
        min_val: int | float,
        max_val: int | float,
        step: int | float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a range parameter definition (slider).

        Args:
            name: Parameter name
            default: Default value
            min_val: Minimum value
            max_val: Maximum value
            step: Step value
            **kwargs: Additional properties

        Returns:
            Parameter definition dictionary
        """
        return {
            "type": "range",
            "default": default,
            "min_value": min_val,
            "max_value": max_val,
            "step": step or ((max_val - min_val) / 100),
            **kwargs,
        }

    @staticmethod
    def boolean(name: str, default: bool = False, **kwargs) -> dict[str, Any]:
        """Create a boolean parameter definition.

        Args:
            name: Parameter name
            default: Default value
            **kwargs: Additional properties

        Returns:
            Parameter definition dictionary
        """
        return {"type": "boolean", "default": default, **kwargs}

    @staticmethod
    def select(
        name: str, options: list[str], default: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a select parameter definition.

        Args:
            name: Parameter name
            options: List of options
            default: Default value
            **kwargs: Additional properties

        Returns:
            Parameter definition dictionary
        """
        if not options:
            raise ValueError("Options list cannot be empty for select parameter")

        if default is None:
            default = options[0]
        elif default not in options:
            logger.warning(
                f"Default value '{default}' not in options, using first option"
            )
            default = options[0]

        return {"type": "select", "default": default, "options": options, **kwargs}

    @staticmethod
    def file(name: str, accept: str | None = None, **kwargs) -> dict[str, Any]:
        """Create a file parameter definition.

        Args:
            name: Parameter name
            accept: Accepted file types (e.g., "audio/*", ".wav,.mp3")
            **kwargs: Additional properties

        Returns:
            Parameter definition dictionary
        """
        return {"type": "file", "accept": accept, **kwargs}


class TTSParameter:
    """Enhanced parameter class with validation and conversion."""

    def __init__(
        self,
        name: str,
        param_type: str,
        label: str | None = None,
        description: str | None = None,
        required: bool = False,
        default: Any = None,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
        step: int | float | None = None,
        options: list[str] | None = None,
        visible_if: str | None = None,
        validator: Callable[[Any], bool] | None = None,
        converter: Callable[[Any], Any] | None = None,
        **properties,
    ):
        """Initialize TTS parameter.

        Args:
            name: Parameter name
            param_type: Parameter type
            label: Human-readable label
            description: Parameter description
            required: Whether parameter is required
            default: Default value
            min_value: Minimum value for numeric parameters
            max_value: Maximum value for numeric parameters
            step: Step value for numeric parameters
            options: Options for select parameters
            visible_if: Condition for visibility
            validator: Custom validation function
            converter: Value conversion function
            **properties: Additional properties
        """
        self.name = name
        self.type = param_type
        self.label = label or name.replace("_", " ").title()
        self.description = description
        self.required = required
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.options = options
        self.visible_if = visible_if
        self.validator = validator
        self.converter = converter
        self.properties = properties

    def validate(self, value: Any) -> bool:
        """Validate a parameter value.

        Args:
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required
            if self.required and (value is None or value == ""):
                return False

            # Skip validation if value is None and not required
            if value is None and not self.required:
                return True

            # Type-specific validation
            if self.type in ["number", "range"]:
                if not isinstance(value, (int, float)):
                    return False

                if self.min_value is not None and value < self.min_value:
                    return False

                if self.max_value is not None and value > self.max_value:
                    return False

            elif self.type == "select":
                if self.options and value not in self.options:
                    return False

            elif self.type == "boolean":
                if not isinstance(value, bool):
                    return False

            # Custom validator
            if self.validator:
                return self.validator(value)

            return True

        except Exception as e:
            logger.error(f"Error validating parameter '{self.name}': {str(e)}")
            return False

    def convert(self, value: Any) -> Any:
        """Convert a parameter value to the correct type.

        Args:
            value: Value to convert

        Returns:
            Converted value
        """
        try:
            # Custom converter
            if self.converter:
                return self.converter(value)

            # Type conversion
            if self.type in ["number", "range"]:
                if isinstance(value, str):
                    try:
                        return float(value) if "." in value else int(value)
                    except ValueError:
                        return self.default
                return value

            elif self.type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)

            return value

        except Exception as e:
            logger.error(f"Error converting parameter '{self.name}': {str(e)}")
            return self.default

    def to_dict(self) -> dict[str, Any]:
        """Convert parameter to dictionary.

        Returns:
            Parameter dictionary
        """
        result = {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }

        # Add type-specific properties
        if self.min_value is not None:
            result["min_value"] = self.min_value

        if self.max_value is not None:
            result["max_value"] = self.max_value

        if self.step is not None:
            result["step"] = self.step

        if self.options:
            result["options"] = self.options

        if self.visible_if:
            result["visible_if"] = self.visible_if

        # Add additional properties
        result.update(self.properties)

        return result

    def to_extension_parameter(self) -> ExtensionParameter:
        """Convert to ExtensionParameter object.

        Returns:
            ExtensionParameter instance
        """
        return ExtensionParameter(
            name=self.name,
            type=self.type,
            label=self.label,
            description=self.description,
            required=self.required,
            default=self.default,
            min_value=self.min_value,
            max_value=self.max_value,
            options=self.options,
        )


class ParameterRegistry:
    """Registry for managing parameter definitions and validation."""

    def __init__(self):
        """Initialize parameter registry."""
        self.parameters: dict[str, TTSParameter] = {}

    def register_parameter(self, param: TTSParameter) -> None:
        """Register a parameter.

        Args:
            param: Parameter to register
        """
        self.parameters[param.name] = param

    def register_parameters(self, params: list[TTSParameter]) -> None:
        """Register multiple parameters.

        Args:
            params: List of parameters to register
        """
        for param in params:
            self.register_parameter(param)

    def get_parameter(self, name: str) -> TTSParameter | None:
        """Get a parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter or None if not found
        """
        return self.parameters.get(name)

    def validate_parameters(self, values: dict[str, Any]) -> bool:
        """Validate a set of parameter values.

        Args:
            values: Parameter values to validate

        Returns:
            True if all valid, False otherwise
        """
        for name, value in values.items():
            param = self.get_parameter(name)
            if param and not param.validate(value):
                logger.error(f"Parameter '{name}' validation failed")
                return False
        return True

    def convert_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        """Convert parameter values to correct types.

        Args:
            values: Parameter values to convert

        Returns:
            Converted parameter values
        """
        converted = {}
        for name, value in values.items():
            param = self.get_parameter(name)
            if param:
                converted[name] = param.convert(value)
            else:
                converted[name] = value
        return converted

    def get_defaults(self) -> dict[str, Any]:
        """Get default values for all parameters.

        Returns:
            Dictionary of default values
        """
        return {name: param.default for name, param in self.parameters.items()}

    def to_extension_parameters(self) -> list[ExtensionParameter]:
        """Convert to ExtensionParameter objects.

        Returns:
            List of ExtensionParameter objects
        """
        return [param.to_extension_parameter() for param in self.parameters.values()]

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert to list of dictionaries.

        Returns:
            List of parameter dictionaries
        """
        return [param.to_dict() for param in self.parameters.values()]


# Utility function to create a parameter registry from definitions
def create_parameter_registry(definitions: list[dict[str, Any]]) -> ParameterRegistry:
    """Create a parameter registry from a list of definitions.

    Args:
        definitions: List of parameter definitions

    Returns:
        ParameterRegistry instance
    """
    registry = ParameterRegistry()

    for definition in definitions:
        param = TTSParameter(**definition)
        registry.register_parameter(param)

    return registry


# Common TTS parameters that can be reused across extensions
class CommonTTSParameters:
    """Common TTS parameter definitions for reuse across extensions."""

    @staticmethod
    def speed(
        default: float = 1.0,
        min_value: float = 0.25,
        max_value: float = 4.0,
        label: str = "Speech Speed",
        description: str = "Control the speed of speech synthesis",
    ) -> ExtensionParameter:
        """Create a standard speed parameter.

        Args:
            default: Default speed value
            min_value: Minimum speed
            max_value: Maximum speed
            label: Parameter label
            description: Parameter description

        Returns:
            ExtensionParameter for speed
        """
        return ExtensionParameter(
            name="speed",
            type="range",
            label=label,
            description=description,
            default=default,
            min_value=min_value,
            max_value=max_value,
        )

    @staticmethod
    def pitch(
        default: float = 1.0,
        min_value: float = 0.5,
        max_value: float = 2.0,
        label: str = "Pitch",
        description: str = "Adjust pitch of voice",
    ) -> ExtensionParameter:
        """Create a standard pitch parameter.

        Args:
            default: Default pitch value
            min_value: Minimum pitch
            max_value: Maximum pitch
            label: Parameter label
            description: Parameter description

        Returns:
            ExtensionParameter for pitch
        """
        return ExtensionParameter(
            name="pitch",
            type="range",
            label=label,
            description=description,
            default=default,
            min_value=min_value,
            max_value=max_value,
        )

    @staticmethod
    def format_selection(
        default: str = "wav",
        options: list[str] | None = None,
        label: str = "Audio Format",
        description: str = "Output audio format",
    ) -> ExtensionParameter:
        """Create a standard format selection parameter.

        Args:
            default: Default format
            options: Available formats
            label: Parameter label
            description: Parameter description

        Returns:
            ExtensionParameter for format selection
        """
        if options is None:
            options = ["wav", "mp3", "ogg"]

        return ExtensionParameter(
            name="format",
            type="select",
            label=label,
            description=description,
            default=default,
            options=options,
        )

    @staticmethod
    def use_gpu(
        default: bool = False,
        label: str = "Use GPU",
        description: str = "Use GPU acceleration if available",
    ) -> ExtensionParameter:
        """Create a standard GPU usage parameter.

        Args:
            default: Default value
            label: Parameter label
            description: Parameter description

        Returns:
            ExtensionParameter for GPU usage
        """
        return ExtensionParameter(
            name="use_gpu",
            type="boolean",
            label=label,
            description=description,
            default=default,
        )

    @staticmethod
    def device(
        default: str = "cpu",
        options: list[str] | None = None,
        label: str = "Device",
        description: str = "Device to run the model on",
    ) -> ExtensionParameter:
        """Create a standard device selection parameter.

        Args:
            default: Default device
            options: Available devices
            label: Parameter label
            description: Parameter description

        Returns:
            ExtensionParameter for device selection
        """
        if options is None:
            options = ["cuda", "cpu", "mps"]

        return ExtensionParameter(
            name="device",
            type="select",
            label=label,
            description=description,
            default=default,
            options=options,
        )
