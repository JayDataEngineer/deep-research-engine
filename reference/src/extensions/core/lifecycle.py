"""
Extension Lifecycle Manager

Manages the activation/deactivation of heavy extensions to conserve resources.
Uses LRU (Least Recently Used) eviction for service extensions like TTS.
"""

import asyncio
import time
from typing import Dict, Optional

from .base import ServiceExtension, ExtensionType


class ExtensionLifecycle:
    """
    Manages resource usage for service extensions.

    Service extensions (like TTS daemons) can consume significant VRAM/RAM.
    This manager ensures only a limited number are active at once.
    """

    def __init__(self, max_heavy: int = 2):
        """
        Initialize the lifecycle manager.

        Args:
            max_heavy: Maximum number of heavy extensions to keep active
        """
        self.active_services: Dict[str, ServiceExtension] = {}
        self.max_heavy = max_heavy
        self.last_used: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def activate(
        self,
        name: str,
        extension: ServiceExtension
    ) -> ServiceExtension:
        """
        Activate a service extension.

        If the extension is already active, updates its last-used time.
        If at capacity, deactivates the oldest extension first.

        Args:
            name: The extension name
            extension: The service extension to activate

        Returns:
            The activated extension
        """
        async with self._lock:
            # Already active?
            if name in self.active_services:
                self.last_used[name] = time.time()
                return self.active_services[name]

            # Check capacity
            if len(self.active_services) >= self.max_heavy:
                await self._unload_oldest()

            # Start the service
            print(f"[Lifecycle] Starting service: {name}")
            await extension.start_service()
            self.active_services[name] = extension
            self.last_used[name] = time.time()

            return extension

    async def deactivate(self, name: str) -> None:
        """
        Deactivate a service extension to free resources.

        Args:
            name: The extension name
        """
        async with self._lock:
            if name in self.active_services:
                print(f"[Lifecycle] Stopping service: {name}")
                ext = self.active_services[name]
                await ext.stop_service()
                del self.active_services[name]
                del self.last_used[name]

    async def _unload_oldest(self) -> None:
        """Unload the least recently used service extension."""
        if not self.last_used:
            return

        # Find oldest
        oldest_name = min(self.last_used, key=self.last_used.get)
        print(f"[Lifecycle] Evicting oldest service: {oldest_name}")

        await self.deactivate(oldest_name)

    async def deactivate_all(self) -> None:
        """Deactivate all service extensions."""
        async with self._lock:
            names = list(self.active_services.keys())
            for name in names:
                await self.deactivate(name)

    def is_active(self, name: str) -> bool:
        """Check if a service is currently active."""
        return name in self.active_services

    def get_active_count(self) -> int:
        """Get the number of active services."""
        return len(self.active_services)
