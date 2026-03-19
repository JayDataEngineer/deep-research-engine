"""Characters Extension

Character card management CLI wrapper.
Uses the character_engine library for core functionality.
"""

from .extension import CharactersExtension, register_extension

__all__ = [
    "CharactersExtension",
    "register_extension",
]
