"""Storage backends module."""

from .base import StorageBackend, StorageResult
from .registry import StorageRegistry, StorageManager
from .errors import StorageError, StorageInitializationError, StorageNotFoundError
from .backends.memory import InMemoryStorage

__all__ = [
    "StorageBackend",
    "StorageResult",
    "StorageRegistry",
    "StorageManager",
    "StorageError",
    "StorageInitializationError",
    "StorageNotFoundError",
    "InMemoryStorage",
]
