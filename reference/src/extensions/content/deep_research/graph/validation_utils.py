"""Validation Utilities - Centralized validation for common checks.

This module provides reusable validation functions to eliminate duplicate
validation code throughout the deep research system.

CRASH-ONLY PHILOSOPHY:
All validation failures raise exceptions immediately.
No fallbacks or silent returns.
"""

from typing import Any, Dict, Optional
from .error_utils import ValidationErrors, CrashErrors


# =============================================================================
# SESSION VALIDATION
# =============================================================================

def validate_session_id(state: Dict[str, Any], function_name: str = "Function") -> str:
    """Validate that session_id exists in state.

    CRASH-ONLY: Raises RuntimeError if session_id is missing.

    Args:
        state: Global state dict
        function_name: Name of calling function for error messages

    Returns:
        The session_id if valid

    Raises:
        RuntimeError: If session_id is not in state
    """
    session_id = state.get("session_id")
    if not session_id:
        raise ValidationErrors.missing_session_id(function_name)
    return session_id


def require_session_id(state: Dict[str, Any], function_name: str = "Function") -> str:
    """Alias for validate_session_id with clearer intent."""
    return validate_session_id(state, function_name)


# =============================================================================
# CONTEXT ENGINE VALIDATION
# =============================================================================

def validate_context_engine(context_engine: Any, function_name: str = "Function") -> Any:
    """Validate that context_engine is initialized.

    CRASH-ONLY: Raises RuntimeError if context_engine is None.

    Args:
        context_engine: The context engine instance to validate
        function_name: Name of calling function for error messages

    Returns:
        The context_engine if valid

    Raises:
        RuntimeError: If context_engine is None
    """
    if not context_engine:
        raise ValidationErrors.missing_context_engine(function_name)
    return context_engine


def require_context_engine(context_engine: Any, function_name: str = "Function") -> Any:
    """Alias for validate_context_engine with clearer intent."""
    return validate_context_engine(context_engine, function_name)


def validate_postgres_store(context_engine: Any, function_name: str = "Function") -> Any:
    """Validate that postgres_store exists on context_engine.

    CRASH-ONLY: Raises RuntimeError if postgres_store is None.

    Args:
        context_engine: The context engine instance
        function_name: Name of calling function for error messages

    Returns:
        The postgres_store if valid

    Raises:
        RuntimeError: If postgres_store is None
    """
    if not context_engine or not context_engine.postgres_store:
        raise ValidationErrors.missing_postgres_store()
    return context_engine.postgres_store


# =============================================================================
# STATE VALIDATION
# =============================================================================

def validate_state_field(state: Dict[str, Any], field: str, function_name: str = "Function") -> Any:
    """Validate that a required field exists in state.

    CRASH-ONLY: Raises RuntimeError if field is missing.

    Args:
        state: Global state dict
        field: Field name to check
        function_name: Name of calling function for error messages

    Returns:
        The field value if present

    Raises:
        RuntimeError: If field is not in state
    """
    if field not in state:
        raise RuntimeError(
            f"╔═══════════════════════════════════════════════════════════════════╗\n"
            f"║       {function_name} - MISSING REQUIRED FIELD                    ║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
            f"\n"
            f"Required field '{field}' not found in state.\n"
            f"This is a CRITICAL state corruption issue.\n"
        )
    return state[field]


def validate_state_fields(state: Dict[str, Any], fields: list[str], function_name: str = "Function") -> Dict[str, Any]:
    """Validate that multiple required fields exist in state.

    CRASH-ONLY: Raises RuntimeError if any field is missing.

    Args:
        state: Global state dict
        fields: List of field names to check
        function_name: Name of calling function for error messages

    Returns:
        Dict of field values if all present

    Raises:
        RuntimeError: If any field is not in state
    """
    return {field: validate_state_field(state, field, function_name) for field in fields}


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get config value with optional default, crash if missing and no default.

    Args:
        config: Config dict
        key: Key to look up
        default: Default value if key missing (crash if None and key missing)

    Returns:
        The config value

    Raises:
        RuntimeError: If key missing and no default provided
    """
    if key not in config:
        if default is not None:
            return default
        raise RuntimeError(
            f"╔═══════════════════════════════════════════════════════════════════╗\n"
            f"║       CONFIGURATION ERROR - MISSING REQUIRED KEY                  ║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
            f"\n"
            f"Required configuration key '{key}' not found.\n"
            f"Check your config.yaml file.\n"
        )
    return config[key]


# =============================================================================
# NEO4J VALIDATION
# =============================================================================

def validate_neo4j_available(context_engine: Any, function_name: str = "Function") -> bool:
    """Check if Neo4j is available on context_engine.

    Args:
        context_engine: The context engine instance
        function_name: Name of calling function for error messages

    Returns:
        True if Neo4j is available, False otherwise

    Note:
        This is a CHECK, not a crash. Returns boolean instead of raising.
    """
    return context_engine is not None and hasattr(context_engine, 'neo4j_store') and context_engine.neo4j_store is not None


# =============================================================================
# MINIO VALIDATION
# =============================================================================

def validate_minio_available(context_engine: Any, function_name: str = "Function") -> bool:
    """Check if MinIO/S3 is available on context_engine.

    Args:
        context_engine: The context engine instance
        function_name: Name of calling function for error messages

    Returns:
        True if MinIO/S3 is available, False otherwise

    Note:
        This is a CHECK, not a crash. Returns boolean instead of raising.
    """
    return context_engine is not None and hasattr(context_engine, 's3_backend') and context_engine.s3_backend is not None


# =============================================================================
# VALUE VALIDATION
# =============================================================================

def validate_positive(value: int, name: str = "value") -> int:
    """Validate that a value is positive.

    Args:
        value: Value to check
        name: Name of the value for error messages

    Returns:
        The value if positive

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative(value: int, name: str = "value") -> int:
    """Validate that a value is non-negative.

    Args:
        value: Value to check
        name: Name of the value for error messages

    Returns:
        The value if non-negative

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_in_range(value: int, min_val: int, max_val: int, name: str = "value") -> int:
    """Validate that a value is within a range.

    Args:
        value: Value to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the value for error messages

    Returns:
        The value if in range

    Raises:
        ValueError: If value is out of range
    """
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value


# =============================================================================
# TYPE VALIDATION
# =============================================================================

def validate_type(value: Any, expected_type: type, name: str = "value") -> Any:
    """Validate that a value is of the expected type.

    Args:
        value: Value to check
        expected_type: Expected type
        name: Name of the value for error messages

    Returns:
        The value if type matches

    Raises:
        TypeError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")
    return value


def validate_optional_type(value: Any, expected_type: type, name: str = "value") -> Any:
    """Validate that a value is None or of the expected type.

    Args:
        value: Value to check
        expected_type: Expected type if not None
        name: Name of the value for error messages

    Returns:
        The value if type matches or None

    Raises:
        TypeError: If value is not None and not of expected type
    """
    if value is not None and not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__} or None, got {type(value).__name__}")
    return value
