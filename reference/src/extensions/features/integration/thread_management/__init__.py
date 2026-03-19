"""Thread Management Extension.

Manages conversation threads including listing, retrieving, deleting,
and managing thread metadata like titles and chain of thought data.
"""

from .extension import (
    ThreadManagementExtension,
    register_extension,
)

__all__ = [
    'ThreadManagementExtension',
    'register_extension',
]
