"""Inline Scripts Extension

Inline code/JavaScript execution for advanced customization.
Compatible with SillyTavern inline script format.
"""

from .extension import InlineScriptsExtension, register_extension

__all__ = [
    "InlineScriptsExtension",
    "register_extension",
]
