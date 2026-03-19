"""
Extension Context

Provides enhanced dependency injection for extensions.
The LLM Factory and other core services are accessed through this context.
"""

from typing import Optional, TYPE_CHECKING, Any, Dict, Callable, TypeVar, Type

if TYPE_CHECKING:
    from .manager import ExtensionManager
    from .base import BaseExtension

T = TypeVar('T')


class ExtensionContext:
    """
    Shared context passed to all extensions.

    This provides access to core services without requiring extensions
    to import them directly. The LLM factory stays in the core (kernel)
    and is injected via this context.

    Enhanced with dependency injection capabilities for better
    service management and testability.
    """

    def __init__(
        self,
        llm_factory=None,
        settings=None,
        vector_store=None,
        manager: Optional['ExtensionManager'] = None,
    ):
        """
        Initialize the extension context.

        Args:
            llm_factory: The core LLM factory (DOES NOT MOVE to extensions)
            settings: Application settings object
            vector_store: Optional vector store for memory extensions
            manager: The extension manager (for inter-extension communication)
        """
        self.llm = llm_factory
        self.settings = settings
        self.vector_store = vector_store
        self._manager = manager

        # Dependency injection container
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}

        # Lifecycle hooks
        self._on_shutdown_hooks: List[Callable[[], None]] = []

    # ==============================================================================
    # Extension Access
    # ==============================================================================

    def get_extension(self, name: str) -> Optional['BaseExtension']:
        """
        Get another extension by name.

        This enables inter-extension communication. For example,
        the emotion_tagger extension can get the index_tts extension
        to access its model paths.

        Args:
            name: The extension name (from config.yaml extension.name)

        Returns:
            The extension if found and active, None otherwise
        """
        if self._manager is None:
            return None
        return self._manager.get(name)

    def is_extension_active(self, name: str) -> bool:
        """
        Check if an extension is currently active.

        Args:
            name: The extension name

        Returns:
            True if the extension is loaded and active
        """
        if self._manager is None:
            return False
        return self._manager.is_active(name)

    # ==============================================================================
    # Dependency Injection
    # ==============================================================================

    def register_service(self, name: str, service: Any, singleton: bool = True) -> None:
        """
        Register a service in the DI container.

        Args:
            name: Service name/identifier
            service: The service instance or factory callable
            singleton: If True, service is treated as a singleton
        """
        if singleton:
            self._singletons[name] = service
        else:
            if callable(service):
                self._factories[name] = service
            else:
                self._services[name] = service

    def register_factory(self, name: str, factory: Callable[[], T]) -> None:
        """
        Register a factory function for lazy service instantiation.

        Args:
            name: Service name/identifier
            factory: Callable that creates the service instance
        """
        self._factories[name] = factory

    def get_service(self, name: str, default: Any = None) -> Any:
        """
        Get a service from the DI container.

        Lookup order:
        1. Singletons
        2. Regular services
        3. Factories (instantiates on first access)
        4. Default value

        Args:
            name: Service name/identifier
            default: Default value if service not found

        Returns:
            The service instance or default
        """
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]

        # Check regular services
        if name in self._services:
            return self._services[name]

        # Check factories (lazy instantiation)
        if name in self._factories:
            instance = self._factories[name]()
            # Cache as singleton for future access
            self._singletons[name] = instance
            del self._factories[name]
            return instance

        return default

    def get_service_or_fail(self, name: str) -> Any:
        """
        Get a service or raise an exception if not found.

        Args:
            name: Service name/identifier

        Returns:
            The service instance

        Raises:
            KeyError: If service is not found
        """
        service = self.get_service(name)
        if service is None:
            raise KeyError(f"Service '{name}' not found in extension context")
        return service

    def has_service(self, name: str) -> bool:
        """
        Check if a service is registered.

        Args:
            name: Service name/identifier

        Returns:
            True if service is available
        """
        return (
            name in self._singletons or
            name in self._services or
            name in self._factories
        )

    # ==============================================================================
    # Lifecycle Management
    # ==============================================================================

    def on_shutdown(self, hook: Callable[[], None]) -> None:
        """
        Register a shutdown hook.

        Hooks are called in reverse registration order during shutdown.

        Args:
            hook: Callable to execute on shutdown
        """
        self._on_shutdown_hooks.append(hook)

    async def shutdown(self) -> None:
        """
        Execute all shutdown hooks.

        Hooks are called in reverse order (LIFO).
        """
        for hook in reversed(self._on_shutdown_hooks):
            try:
                result = hook()
                if callable(result):  # It's a coroutine
                    await result()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"[Context] Error in shutdown hook: {e}")

        # Clear all services
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._on_shutdown_hooks.clear()

    # ==============================================================================
    # Configuration Helpers
    # ==============================================================================

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Setting key (supports dot notation)
            default: Default value if not found

        Returns:
            The setting value or default
        """
        if self.settings is None:
            return default

        # Support dot notation for nested settings
        value = self.settings
        for part in key.split('.'):
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    # ==============================================================================
    # Convenience Properties
    # ==============================================================================

    @property
    def llm_factory(self):
        """Get the LLM factory (alias for llm)."""
        return self.llm

    @property
    def manager(self) -> Optional['ExtensionManager']:
        """Get the extension manager."""
        return self._manager
