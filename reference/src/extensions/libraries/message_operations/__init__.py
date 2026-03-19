"""Message Operations Library - Shared message manipulation logic.

This library provides shared functionality for message operations that are
used across multiple chat modes:
- Message editing (with history truncation)
- Message deletion (with cascading)
- Version switching

NOTE: Message regeneration is handled directly in the message_branching extension
at /api/v1/messages/regenerate, which reuses the extension's stream_chat() method.

This follows the DRY principle - all chat modes need these operations,
so they belong in a shared library rather than duplicated in each extension.
"""

from .edit import edit_message
from .delete import delete_message
from .version import switch_message_version

__all__ = [
    "edit_message",
    "delete_message",
    "switch_message_version",
]
