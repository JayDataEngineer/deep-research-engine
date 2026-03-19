"""
Loading tracker service to manage and track the loading status of TTS extensions.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LoadingStatus(Enum):
    """Enum representing different loading states of extensions."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class LoadingState:
    """Data class to represent the loading state of an extension."""
    extension_name: str
    status: LoadingStatus = LoadingStatus.NOT_LOADED
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    details: dict[str, Any] = field(default_factory=dict)


class LoadingTracker:
    """Service to track and manage the loading states of TTS extensions.

    OPTIMIZED: Simplified to async-only pattern - removes threading complexity.
    """

    def __init__(self):
        """Initialize the loading tracker."""
        self._states: dict[str, LoadingState] = {}
        self._observers = []  # WebSocket connections or callbacks
        self._lock = asyncio.Lock()
        self._notify_pending = False  # Debounce flag for pending notifications

    def register_extension(self, extension_name: str) -> bool:
        """Register a new extension with the tracker."""
        if extension_name in self._states:
            logger.warning(f"Extension {extension_name} already registered in tracker")
            return False

        self._states[extension_name] = LoadingState(
            extension_name=extension_name,
            status=LoadingStatus.NOT_LOADED
        )
        logger.info(f"Registered extension {extension_name} in loading tracker")
        # OPTIMIZED: Lazy notification - observers will pick up on next status check
        return True

    def set_loading_status(self, extension_name: str, status: LoadingStatus,
                          progress: float = 0.0, message: str = "",
                          details: dict[str, Any] | None = None) -> bool:
        """Update the loading status of an extension."""
        if extension_name not in self._states:
            logger.warning(f"Extension {extension_name} not found in tracker")
            return False

        with __import__('contextlib').suppress(Exception):
            state = self._states[extension_name]
            state.status = status
            state.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
            state.message = message
            state.timestamp = __import__('time').time()
            if details:
                state.details.update(details)

            logger.info(f"Updated {extension_name} status to {status.value}, progress: {progress:.2f}, message: {message}")
            # OPTIMIZED: Just set flag - actual notification happens on async check
            self._notify_pending = True
            return True

    def get_loading_state(self, extension_name: str) -> LoadingState | None:
        """Get the loading state of an extension."""
        return self._states.get(extension_name)

    def get_all_loading_states(self) -> dict[str, LoadingState]:
        """Get the loading states of all extensions."""
        return self._states.copy()

    def get_loading_status(self, extension_name: str) -> LoadingStatus | None:
        """Get the loading status of an extension."""
        state = self._states.get(extension_name)
        return state.status if state else None

    def is_loaded(self, extension_name: str) -> bool:
        """Check if an extension is loaded."""
        state = self._states.get(extension_name)
        return state.status == LoadingStatus.LOADED if state else False

    def is_loading(self, extension_name: str) -> bool:
        """Check if an extension is currently loading."""
        state = self._states.get(extension_name)
        return state.status in [LoadingStatus.LOADING] if state else False

    def is_unloading(self, extension_name: str) -> bool:
        """Check if an extension is currently unloading."""
        state = self._states.get(extension_name)
        return state.status in [LoadingStatus.UNLOADING] if state else False

    def reset_extension(self, extension_name: str) -> bool:
        """Reset an extension's state to NOT_LOADED."""
        if extension_name not in self._states:
            return False

        self._states[extension_name] = LoadingState(
            extension_name=extension_name,
            status=LoadingStatus.NOT_LOADED
        )
        logger.info(f"Reset extension {extension_name} state to NOT_LOADED")
        # OPTIMIZED: Just set flag - observers poll for changes
        self._notify_pending = True
        return True

    def register_observer(self, callback: Callable):
        """Register a callback to be notified of state changes."""
        if callback not in self._observers:
            self._observers.append(callback)
            logger.debug("Registered observer for loading tracker")

    def unregister_observer(self, callback: Callable):
        """Unregister a callback."""
        if callback in self._observers:
            self._observers.remove(callback)
            logger.debug("Unregistered observer for loading tracker")

    async def notify_if_pending(self) -> int:
        """Notify observers if there are pending changes.

        OPTIMIZED: Call this periodically from async context instead of
        spawning threads. Returns number of observers notified.
        """
        if not self._notify_pending:
            return 0

        self._notify_pending = False
        notified = 0

        # Create a copy to avoid issues if observers modify the list
        observers_copy = self._observers[:]

        for observer in observers_copy:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer()
                else:
                    # Call the observer function, awaiting if it returns a coroutine
                    result = observer()
                    if asyncio.iscoroutine(result):
                        await result
                notified += 1
            except Exception as e:
                logger.error(f"Error notifying observer: {e}", exc_info=True)

        return notified

    async def _notify_observers(self):
        """Notify all registered observers of state changes.

        DEPRECATED: Use notify_if_pending() instead for better performance.
        """
        # Create a copy to avoid issues if observers modify the list
        observers_copy = self._observers[:]

        for observer in observers_copy:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer()
                else:
                    # Call the observer function, awaiting if it returns a coroutine
                    result = observer()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                logger.error(f"Error notifying observer: {e}", exc_info=True)


# Global instance of the loading tracker
loading_tracker = LoadingTracker()
