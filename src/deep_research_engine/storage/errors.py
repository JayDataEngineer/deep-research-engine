"""Storage errors."""


class StorageError(Exception):
    """Base exception for storage errors."""
    pass


class StorageInitializationError(StorageError):
    """Raised when storage initialization fails."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when storage backend not found."""
    pass
