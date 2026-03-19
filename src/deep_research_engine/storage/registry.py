"""Storage registry and manager."""

from typing import TYPE_CHECKING, Any

from .errors import StorageNotFoundError

if TYPE_CHECKING:
    from .base import StorageBackend


class StorageRegistry:
    """Registry for storage backends."""

    _backends: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a backend."""

        def decorator(backend_class: type) -> type:
            cls._backends[name] = backend_class
            backend_class.backend_name = name
            return backend_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get backend class by name."""
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise StorageNotFoundError(f"Backend '{name}' not found. Available: {available}")
        return cls._backends[name]

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backends."""
        return list(cls._backends.keys())


class StorageManager:
    """Manage multiple storage backends."""

    def __init__(self):
        self.backends: dict[str, StorageBackend] = {}
        self.initialized = False

    async def initialize(
        self,
        backend_name: str,
        config: dict[str, Any],
    ) -> None:
        """Initialize a storage backend."""
        backend_class = StorageRegistry.get(backend_name)
        backend = backend_class(config)

        await backend.initialize()
        self.backends[backend_name] = backend

        self._log(f"Initialized backend: {backend_name}")

    async def cleanup(self) -> None:
        """Cleanup all backends."""
        for name, backend in self.backends.items():
            try:
                await backend.cleanup()
                self._log(f"Cleaned up backend: {name}")
            except Exception as e:
                self._log(f"Error cleaning up {name}: {e}", level="error")

    async def store(
        self,
        backend_name: str,
        data: dict[str, Any],
        namespace: str = "default",
    ) -> str | None:
        """Store data in specific backend."""
        if backend_name not in self.backends:
            self._log(f"Backend not initialized: {backend_name}", level="warning")
            return None

        backend = self.backends[backend_name]
        result = await backend.store(data, namespace)

        if result.success:
            self._log(f"Stored in {backend_name}: {result.id}")
            return result.id
        else:
            self._log(f"Store failed in {backend_name}: {result.error}", level="error")
            return None

    async def retrieve(
        self,
        backend_name: str,
        id: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve data from specific backend."""
        if backend_name not in self.backends:
            return None

        backend = self.backends[backend_name]
        return await backend.retrieve(id, namespace)

    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message."""
        from ..utils.logging import get_logger

        logger = get_logger("storage.manager")

        log_method = getattr(logger, level, logger.info)
        log_method(message, **kwargs)
