"""Token Counter Extension

Real-time token counting and statistics.
Compatible with SillyTavern token counting system.
"""

from .extension import TokenCounterExtension, register_extension

__all__ = [
    "TokenCounterExtension",
    "register_extension",
]
