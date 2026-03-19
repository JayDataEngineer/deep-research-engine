"""Prompt Converter Extension

Convert prompts between different formats.
Supports SillyTavern, Claude, ChatGPT, and generic formats.
"""

from .extension import PromptConverterExtension, register_extension

__all__ = [
    "PromptConverterExtension",
    "register_extension",
]
