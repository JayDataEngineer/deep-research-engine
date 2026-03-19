"""Lorebooks Extension

Lorebook (World Info) management CLI wrapper.
Uses the lorebook_engine library for core functionality.
"""

from .extension import LorebooksExtension, register_extension

__all__ = [
    "LorebooksExtension",
    "register_extension",
]
