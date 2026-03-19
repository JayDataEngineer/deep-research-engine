"""Slash Commands Extension

Comprehensive SillyTavern-compatible slash commands.
"""

from .extension import SlashCommandsExtension, register_extension

__all__ = [
    "SlashCommandsExtension",
    "register_extension",
]
