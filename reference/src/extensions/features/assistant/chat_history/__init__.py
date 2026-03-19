"""Chat History Extension

Chat history management with JSONL import/export.
Compatible with SillyTavern chat format.
"""

from .extension import ChatHistoryExtension, register_extension
from .storage import ChatMessage, ChatMetadata, ChatHistoryStorage, get_chat_storage

__all__ = [
    "ChatHistoryExtension",
    "register_extension",
    "ChatMessage",
    "ChatMetadata",
    "ChatHistoryStorage",
    "get_chat_storage",
]
