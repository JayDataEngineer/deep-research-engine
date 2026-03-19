"""Context Management Extension

Conversation context and token management.
Compatible with SillyTavern context format.
"""

from .extension import ContextExtension, register_extension

__all__ = [
    "ContextExtension",
    "register_extension",
]
