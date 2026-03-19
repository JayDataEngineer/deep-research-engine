"""Instruct Mode Extension

Instruct mode template management.
Compatible with SillyTavern instruct mode system.
"""

from .extension import InstructModeExtension, register_extension

__all__ = [
    "InstructModeExtension",
    "register_extension",
]
