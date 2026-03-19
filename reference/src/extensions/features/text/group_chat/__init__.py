"""Group Chat Extension

Multi-character group conversations.
Compatible with SillyTavern group chat system.
"""

from .extension import GroupChatExtension, register_extension

__all__ = [
    "GroupChatExtension",
    "register_extension",
]
