"""Shared Checkpointer - Async PostgreSQL persistence.

CRASH-ONLY: AsyncPostgresSaver is REQUIRED. NO FALLBACKS.

Usage:
    from src.shared.checkpointer.factory import get_checkpointer, get_store

    checkpointer = await get_checkpointer()
    store = await get_store()
"""

# Re-export async factory functions
from src.shared.checkpointer.factory import (
    get_checkpointer,
    get_store,
    get_shared_checkpointer,
    CheckpointerError,
)

__all__ = [
    "get_checkpointer",
    "get_store",
    "get_shared_checkpointer",
    "CheckpointerError",
]
