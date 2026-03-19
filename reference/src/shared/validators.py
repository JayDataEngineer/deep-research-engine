"""Shared Pydantic validators.

This module contains reusable validator functions and factory functions
to create validators for common patterns.
"""

from typing import Optional, Dict, Any, List
from pydantic import field_validator


# =============================================================================
# URL VALIDATION
# =============================================================================


def create_url_dict_validator(field_name: str):
    """Create a validator for URL dictionary fields.

    Validates that the field is a dict with a 'url' key containing a non-empty string.

    Args:
        field_name: Name of the field being validated (for error messages)

    Returns:
        A validator function that can be used with @field_validator decorator

    Example:
        class MyModel(BaseModel):
            image_url: Optional[dict] = None

            validate_image_url = create_url_dict_validator('image_url')
    """
    @field_validator(field_name)
    @classmethod
    def validate_url_dict(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate URL dictionary has required 'url' key."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError(f"{field_name} must be a dictionary with 'url' key")
            if "url" not in v:
                raise ValueError(f"{field_name} must contain 'url' key")
            if not v["url"]:
                raise ValueError(f"{field_name}.url cannot be empty")
        return v

    return validate_url_dict


# =============================================================================
# STRING VALIDATION
# =============================================================================


def create_not_empty_validator(field_name: str, allow_whitespace: bool = False):
    """Create a validator that ensures a string field is not empty.

    Args:
        field_name: Name of the field being validated (for error messages)
        allow_whitespace: Whether to allow whitespace-only strings

    Returns:
        A validator function

    Example:
        class MyModel(BaseModel):
            name: str

            validate_name = create_not_empty_validator('name')
    """
    @field_validator(field_name)
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that string is not empty."""
        if v is None:
            raise ValueError(f"{field_name} cannot be None")
        if not isinstance(v, str):
            raise ValueError(f"{field_name} must be a string")
        if allow_whitespace:
            if v == "":
                raise ValueError(f"{field_name} cannot be empty")
        else:
            if not v.strip():
                raise ValueError(f"{field_name} cannot be empty or whitespace-only")
        return v

    return validate_not_empty


def create_non_empty_list_validator(field_name: str, max_items: Optional[int] = None):
    """Create a validator that ensures a list field is not empty.

    Args:
        field_name: Name of the field being validated
        max_items: Optional maximum number of items

    Returns:
        A validator function
    """
    @field_validator(field_name)
    @classmethod
    def validate_not_empty(cls, v: List) -> List:
        """Validate that list is not empty."""
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError(f"{field_name} cannot be empty")
            if max_items and len(v) > max_items:
                raise ValueError(f"{field_name} cannot contain more than {max_items} items")
        return v

    return validate_not_empty


# =============================================================================
# URL STRING VALIDATION
# =============================================================================


def create_url_string_validator(field_name: str):
    """Create a validator for URL string fields.

    Validates that the field is a string starting with http:// or https://.

    Args:
        field_name: Name of the field being validated

    Returns:
        A validator function
    """
    @field_validator(field_name)
    @classmethod
    def validate_url_string(cls, v: str) -> str:
        """Validate that string is a valid URL."""
        if v is not None:
            if not isinstance(v, str):
                raise ValueError(f"{field_name} must be a string")
            if not v.startswith(("http://", "https://")):
                raise ValueError(f"{field_name} must start with http:// or https://")
        return v

    return validate_url_string


# =============================================================================
# CONTENT VALIDATION (for multimodal messages)
# =============================================================================


def create_content_validator(field_name: str = "content", max_items: int = 50):
    """Create a validator for message content fields.

    Handles both string content and list of content items.

    Args:
        field_name: Name of the field being validated
        max_items: Maximum number of content items in a list

    Returns:
        A validator function
    """
    @field_validator(field_name)
    @classmethod
    def validate_content(cls, v: Any) -> Any:
        """Validate that content is not empty."""
        from .types import is_empty_string

        if is_empty_string(v):
            raise ValueError(f"{field_name} cannot be empty or whitespace-only")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError(f"{field_name} list cannot be empty")
        if isinstance(v, list) and len(v) > max_items:
            raise ValueError(f"{field_name} cannot contain more than {max_items} items")
        return v

    return validate_content


# =============================================================================
# ROLE VALIDATION
# =============================================================================


def create_first_message_role_validator(field_name: str = "messages", allowed_first_roles: List[str] = None):
    """Create a validator that ensures the first message has an allowed role.

    Args:
        field_name: Name of the messages field
        allowed_first_roles: List of allowed roles for the first message

    Returns:
        A validator function
    """
    if allowed_first_roles is None:
        allowed_first_roles = ["system", "user"]

    @field_validator(field_name)
    @classmethod
    def validate_first_role(cls, v: List) -> List:
        """Validate that first message has an appropriate role."""
        if not v:
            raise ValueError(f"{field_name} list cannot be empty")
        first_role = getattr(v[0], "role", None)
        if first_role not in allowed_first_roles:
            raise ValueError(f"First message must have role in {allowed_first_roles}, got '{first_role}'")
        return v

    return validate_first_role
