"""Message Branching Extension.

Manages message versioning, editing, regeneration, and deletion.
Provides support for multiple response versions per message.
"""

from .extension import (
    MessageBranchingExtension,
    register_extension,
)

__all__ = [
    'MessageBranchingExtension',
    'register_extension',
]
