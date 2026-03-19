"""Probability Control Extension

Probability and random sampling management.
Compatible with SillyTavern probability control format.
"""

from .extension import ProbabilityExtension, register_extension

__all__ = [
    "ProbabilityExtension",
    "register_extension",
]
