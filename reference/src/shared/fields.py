"""Shared Pydantic field definitions.

This module contains reusable field factories for common Pydantic fields
to avoid duplication and ensure consistent validation across the codebase.
"""

from typing import Optional, Any
from pydantic import Field, confloat, conint

# =============================================================================
# LLM PARAMETER FIELDS
# =============================================================================


def temperature_field(
    default: float = 0.7,
    required: bool = False,
    description: str = "Sampling temperature (0.0=deterministic, 2.0=creative)"
) -> Any:
    """Create a temperature field with standard validation.

    Args:
        default: Default temperature value
        required: Whether the field is required (None if optional)
        description: Field description

    Returns:
        Configured temperature field
    """
    if required:
        return Field(
            default=default,
            ge=0.0,
            le=2.0,
            description=description
        )
    return Field(
        default=default,
        ge=0.0,
        le=2.0,
        description=description
    )


def optional_temperature_field(default: Optional[float] = None) -> Any:
    """Create an optional temperature field.

    Args:
        default: Default temperature value (None if not specified)

    Returns:
        Configured optional temperature field
    """
    return Field(
        default=default,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0=deterministic, 2.0=creative)"
    )


def max_tokens_field(
    default: Optional[int] = None,
    min_value: int = 1,
    max_value: int = 128000,
    description: str = "Maximum tokens to generate"
) -> Any:
    """Create a max_tokens field with standard validation.

    Args:
        default: Default max_tokens value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        description: Field description

    Returns:
        Configured max_tokens field
    """
    return Field(
        default=default,
        ge=min_value,
        le=max_value,
        description=description
    )


# =============================================================================
# STRING FIELDS
# =============================================================================


def string_field(
    min_length: int = 1,
    max_length: int = 100,
    description: str = "String value",
    default: Optional[str] = None,
) -> Any:
    """Create a constrained string field.

    Args:
        min_length: Minimum string length
        max_length: Maximum string length
        description: Field description
        default: Default value

    Returns:
        Configured string field
    """
    return Field(
        default=default,
        min_length=min_length,
        max_length=max_length,
        description=description
    )


def id_field(description: str = "Unique identifier") -> Any:
    """Create an ID field with standard validation.

    Args:
        description: Field description

    Returns:
        Configured ID field
    """
    return Field(
        ...,
        min_length=1,
        max_length=100,
        description=description
    )


# =============================================================================
# TIMESTAMP FIELDS
# =============================================================================


def created_at_field(description: str = "When this was created") -> Any:
    """Create a created_at timestamp field.

    Args:
        description: Field description

    Returns:
        Configured timestamp field
    """
    from datetime import datetime
    return Field(
        default_factory=datetime.utcnow,
        description=description
    )


# =============================================================================
# METADATA FIELDS
# =============================================================================


def metadata_field(description: str = "Additional metadata") -> Any:
    """Create a metadata dictionary field.

    Args:
        description: Field description

    Returns:
        Configured metadata field
    """
    return Field(
        default_factory=dict,
        description=description
    )


def extensions_field(description: str = "Custom metadata/extensions") -> Any:
    """Create an extensions dictionary field.

    Args:
        description: Field description

    Returns:
        Configured extensions field
    """
    return Field(
        default_factory=dict,
        description=description
    )


def variables_field(description: str = "Variable mappings") -> Any:
    """Create a variables dictionary field.

    Args:
        description: Field description

    Returns:
        Configured variables field
    """
    return Field(
        default_factory=dict,
        description=description
    )


# =============================================================================
# LIST FIELDS
# =============================================================================


def messages_field(min_length: int = 1, max_length: int = 100) -> Any:
    """Create a messages list field with standard validation.

    Args:
        min_length: Minimum number of messages
        max_length: Maximum number of messages

    Returns:
        Configured messages field
    """
    from pydantic import conlist
    return Field(
        ...,
        min_length=min_length,
        max_length=max_length,
        description=f"List of messages ({min_length}-{max_length})"
    )
