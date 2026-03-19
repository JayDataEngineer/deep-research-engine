"""
Extension Manager Utilities

Shared utility functions for the extension system.
"""

from typing import Any
from ..base import BaseExtension
from .registry import ExtensionProxy


def unwrap_extension(ext: BaseExtension) -> BaseExtension:
    """
    Unwrap an ExtensionProxy to get the real extension instance.

    Args:
        ext: Extension (may be a proxy)

    Returns:
        The real extension instance, or the original if not a proxy
    """
    if isinstance(ext, ExtensionProxy):
        return ext._instance if ext._instance is not None else ext
    return ext


def is_extension_instance(ext: BaseExtension, cls: type) -> bool:
    """
    Check if extension is an instance of a class, unwrapping proxies.

    Args:
        ext: Extension to check (may be a proxy)
        cls: Class to check against

    Returns:
        True if ext (or its real instance) is an instance of cls
    """
    return isinstance(unwrap_extension(ext), cls)
