"""Common API helper functions to eliminate code duplication."""

import logging
from datetime import datetime
from typing import Optional, TypeVar, TypeVar, Callable, Any, Dict
from functools import wraps
from pathlib import Path

from fastapi import HTTPException
from src.config import get_output_dir, get_data_dir, get_inputs_dir

logger = logging.getLogger(__name__)

# Type vars for generic functions
T = TypeVar('T')


# =============================================================================
# ID Normalization
# =============================================================================

def normalize_id(name: str) -> str:
    """Convert a name to a URL-safe ID.

    Args:
        name: Name to normalize

    Returns:
        Lowercase, underscore-separated ID
    """
    return name.lower().replace(" ", "_")


# =============================================================================
# HTTP Exception Helpers
# =============================================================================

def or_404(obj: Optional[T], item_id: str, item_type: str = "Item") -> T:
    """Raise 404 if object is None.

    Args:
        obj: Object to check (can be None)
        item_id: ID of the item for error message
        item_type: Type name for error message

    Returns:
        The object if not None

    Raises:
        HTTPException: 404 if obj is None
    """
    if obj is None:
        raise HTTPException(
            status_code=404,
            detail=f"{item_type} '{item_id}' not found"
        )
    return obj


def or_409(condition: bool, message: str) -> None:
    """Raise 409 Conflict if condition is true.

    Args:
        condition: Condition to check
        message: Error message

    Raises:
        HTTPException: 409 if condition is True
    """
    if condition:
        raise HTTPException(status_code=409, detail=message)


def or_400(condition: bool, message: str) -> None:
    """Raise 400 Bad Request if condition is true.

    Args:
        condition: Condition to check
        message: Error message

    Raises:
        HTTPException: 400 if condition is True
    """
    if condition:
        raise HTTPException(status_code=400, detail=message)


# =============================================================================
# DateTime Serialization
# =============================================================================

def serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Safely serialize datetime to ISO format string.

    Args:
        dt: DateTime to serialize

    Returns:
        ISO format string or None
    """
    return dt.isoformat() if dt else None


def serialize_datetimes(obj: Any, datetime_fields: list[str]) -> dict:
    """Serialize multiple datetime fields in an object/dict.

    Args:
        obj: Object with datetime fields (can be dict or object)
        datetime_fields: List of field names to serialize

    Returns:
        Dict with serialized datetime fields
    """
    if isinstance(obj, dict):
        result = obj.copy()
    else:
        result = obj.__dict__.copy()

    for field in datetime_fields:
        if field in result and isinstance(result[field], datetime):
            result[field] = result[field].isoformat()
        elif field not in result or result[field] is None:
            result[field] = None

    return result


# =============================================================================
# Pydantic Response Helpers
# =============================================================================

def build_response(
    model_class: type[T],
    **fields
) -> T:
    """Build a Pydantic response model with automatic datetime serialization.

    Args:
        model_class: Response model class
        **fields: Field values for the model

    Returns:
        Instance of model_class with serialized fields
    """
    # Auto-serialize datetime fields
    processed = {}
    for key, value in fields.items():
        if isinstance(value, datetime):
            processed[key] = value.isoformat()
        elif value is None:
            processed[key] = None
        else:
            processed[key] = value

    return model_class(**processed)


# =============================================================================
# Registry Dependency Pattern
# =============================================================================

def with_registry(registry_getter: Callable[[], Any]):
    """Decorator to inject registry into endpoint functions.

    Usage:
        @with_registry(get_registry)
        async def my_endpoint(registry):
            return registry.list_all()

    Args:
        registry_getter: Function that returns the registry instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Inject registry as first argument if not already in kwargs
            if 'registry' not in kwargs:
                kwargs['registry'] = registry_getter()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# File Path Helpers
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_root_dir() -> Path:
    """Get the unified output directory (parent-level outputs/).

    Returns:
        Path to outputs/ at project root
    """
    from src.config import get_output_dir as _get_output_dir
    return _get_output_dir()


def get_data_dir() -> Path:
    """Get the unified data directory (redirects to outputs/).

    Note: This function is kept for backward compatibility.
    Use get_output_root_dir() directly for new code.
    """
    return get_output_root_dir()


# =============================================================================
# Input Directory Helpers
# =============================================================================

def get_inputs_root_dir() -> Path:
    """Get the unified input directory (parent-level inputs/).

    Returns:
        Path to inputs/ at project root
    """
    from src.config import get_inputs_dir as _get_inputs_dir
    return _get_inputs_dir()


def get_workflows_dir() -> Path:
    """Get the workflows directory.

    Returns:
        Path to inputs/workflows/
    """
    return ensure_dir(get_inputs_root_dir() / "workflows")


def get_input_characters_dir() -> Path:
    """Get the input characters directory.

    Returns:
        Path to inputs/characters/
    """
    return ensure_dir(get_inputs_root_dir() / "characters")


def get_input_data_dir() -> Path:
    """Get the input data sources directory.

    Returns:
        Path to inputs/data/
    """
    return ensure_dir(get_inputs_root_dir() / "data")


def get_input_lorebooks_dir() -> Path:
    """Get the input lorebooks directory.

    Returns:
        Path to inputs/lorebooks/
    """
    return ensure_dir(get_inputs_root_dir() / "lorebooks")


def get_input_prompts_dir() -> Path:
    """Get the input prompts directory.

    Returns:
        Path to inputs/prompts/
    """
    return ensure_dir(get_inputs_root_dir() / "prompts")


# =============================================================================
# Output Directory Helpers
# =============================================================================

def get_uploads_dir() -> Path:
    """Get the uploads directory.

    Returns:
        Path to outputs/uploads/
    """
    return ensure_dir(get_output_root_dir() / "uploads")


def get_lorebooks_dir() -> Path:
    """Get the lorebooks directory.

    Returns:
        Path to outputs/lorebooks/
    """
    return ensure_dir(get_output_root_dir() / "lorebooks")


def get_prompts_dir() -> Path:
    """Get the prompts directory.

    Returns:
        Path to outputs/prompts/
    """
    return ensure_dir(get_output_root_dir() / "prompts")


def get_output_dir() -> Path:
    """Get the generated content output directory.

    Returns:
        Path to outputs/generated/
    """
    return ensure_dir(get_output_root_dir() / "generated")


def get_generated_images_dir() -> Path:
    """Get the generated images directory.

    Returns:
        Path to outputs/generated_images/
    """
    return ensure_dir(get_output_root_dir() / "generated_images")


def get_roleplay_dir() -> Path:
    """Get the roleplay data directory.

    Returns:
        Path to outputs/roleplay/
    """
    return ensure_dir(get_output_root_dir() / "roleplay")


def get_characters_dir() -> Path:
    """Get the characters directory.

    Returns:
        Path to outputs/characters/
    """
    return ensure_dir(get_output_root_dir() / "characters")


def get_companion_output_dir() -> Path:
    """Get the companion output directory.

    Returns:
        Path to outputs/companion/
    """
    return ensure_dir(get_output_root_dir() / "companion")


def get_logs_dir() -> Path:
    """Get the logs directory.

    Returns:
        Path to outputs/logs/
    """
    return ensure_dir(get_output_root_dir() / "logs")


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory.

    Returns:
        Path to outputs/checkpoints/
    """
    return ensure_dir(get_output_root_dir() / "checkpoints")


def initialize_data_directories() -> None:
    """Initialize all output directories in one place.

    This is the single source of truth for directory creation.
    Call this during application startup.
    """
    output_dir = get_output_root_dir()
    ensure_dir(output_dir)

    # Core output subdirectories
    ensure_dir(output_dir / "uploads")
    ensure_dir(output_dir / "uploads" / "characters")
    ensure_dir(output_dir / "uploads" / "image")
    ensure_dir(output_dir / "lorebooks")
    ensure_dir(output_dir / "prompts")
    ensure_dir(output_dir / "generated")
    ensure_dir(output_dir / "generated_images")
    ensure_dir(output_dir / "roleplay")
    ensure_dir(output_dir / "characters")
    ensure_dir(output_dir / "companion")
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "checkpoints")

    logger.debug(f"Output directories initialized: {output_dir}")


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_id_exists(id_value: str, check_func: Callable[[str], bool], item_type: str = "Item") -> str:
    """Validate that an ID exists, raise 404 if not.

    Args:
        id_value: ID to check
        check_func: Function that returns True if ID exists
        item_type: Type name for error message

    Returns:
        The ID if valid

    Raises:
        HTTPException: 404 if ID doesn't exist
    """
    if not check_func(id_value):
        raise HTTPException(
            status_code=404,
            detail=f"{item_type} '{id_value}' not found"
        )
    return id_value


def validate_id_not_exists(id_value: str, check_func: Callable[[str], bool], item_type: str = "Item") -> str:
    """Validate that an ID doesn't exist, raise 409 if it does.

    Args:
        id_value: ID to check
        check_func: Function that returns True if ID exists
        item_type: Type name for error message

    Returns:
        The ID if valid

    Raises:
        HTTPException: 409 if ID already exists
    """
    if check_func(id_value):
        raise HTTPException(
            status_code=409,
            detail=f"{item_type} '{id_value}' already exists"
        )
    return id_value
