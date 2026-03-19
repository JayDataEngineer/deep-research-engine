"""StorageBackend ABC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class StorageResult:
    """Result from a storage operation."""

    success: bool
    id: str | None = None
    metadata: dict[str, Any] = None
    error: str | None = None


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    backend_name: str = "base"
    description: str = "Base storage backend"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.initialized = False
        self.health_status = "unknown"

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize backend."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass

    @abstractmethod
    async def store(
        self,
        data: dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        """Store data in backend."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve data by ID."""
        pass

    @abstractmethod
    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        """Delete data by ID."""
        pass

    async def query(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Query data (optional)."""
        return []

    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List items in namespace (optional)."""
        return []

    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message."""
        from ..utils.logging import get_logger

        logger = get_logger(f"storage.{self.backend_name}")

        log_method = getattr(logger, level, logger.info)
        log_method(f"[{self.backend_name}] {message}", **kwargs)
