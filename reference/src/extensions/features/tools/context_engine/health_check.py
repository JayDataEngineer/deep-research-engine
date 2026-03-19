"""
Health Check System for Context Engine backends.

Provides runtime monitoring of backend availability with caching.
"""
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status of a backend."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


class HealthCheckRegistry:
    """Registry for backend health checks.

    Allows components to:
    - Register health checks for their backends
    - Query current health status
    - Get list of available backends
    - Conditionally enable features based on health
    """

    def __init__(self, cache_ttl: float = 5.0):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._cache: Dict[str, HealthCheckResult] = {}
        self._cache_ttl: float = cache_ttl
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult]
    ) -> None:
        """Register a health check function.

        Args:
            name: Unique name for this backend (e.g., "neo4j", "postgres", "minio")
            check_func: Function that returns HealthCheckResult
        """
        self._checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    async def check(
        self,
        name: str,
        use_cache: bool = True
    ) -> HealthCheckResult:
        """Check health of a specific backend.

        Args:
            name: Backend name to check
            use_cache: Use cached result if available and fresh

        Returns:
            HealthCheckResult for the backend
        """
        async with self._lock:
            # Check cache first
            if use_cache and name in self._cache:
                cached = self._cache[name]
                if time.time() - cached.timestamp < self._cache_ttl:
                    return cached

            # Run health check
            if name not in self._checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"No health check registered for '{name}'"
                )

            try:
                # Health checks can be sync or async
                result = self._checks[name]()
                if asyncio.iscoroutine(result):
                    result = await result
                self._cache[name] = result
                return result
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {str(e)}"
                )
                self._cache[name] = result
                return result

    async def check_all(
        self,
        use_cache: bool = True
    ) -> Dict[str, HealthCheckResult]:
        """Check health of all registered backends.

        Returns:
            Dict mapping backend names to HealthCheckResult
        """
        results = {}
        for name in self._checks:
            results[name] = await self.check(name, use_cache)
        return results

    def is_healthy(
        self,
        name: str,
        allow_degraded: bool = False
    ) -> bool:
        """Quick check if backend is healthy (synchronous, uses cache).

        Args:
            name: Backend name
            allow_degraded: Whether DEGRADED status counts as healthy

        Returns:
            True if backend is healthy (or degraded if allowed)
        """
        if name not in self._cache:
            return False

        status = self._cache[name].status
        if status == HealthStatus.HEALTHY:
            return True
        if allow_degraded and status == HealthStatus.DEGRADED:
            return True
        return False

    def get_available_backends(
        self,
        allow_degraded: bool = False
    ) -> List[str]:
        """Get list of available (healthy) backends.

        Args:
            allow_degraded: Whether to include degraded backends

        Returns:
            List of backend names that are healthy
        """
        return [
            name for name in self._checks.keys()
            if self.is_healthy(name, allow_degraded)
        ]

    async def invalidate_cache(self, name: Optional[str] = None) -> None:
        """Invalidate cached health check results.

        Args:
            name: Specific backend to invalidate, or None for all
        """
        async with self._lock:
            if name:
                self._cache.pop(name, None)
            else:
                self._cache.clear()


# Global health check registry
_global_registry: Optional[HealthCheckRegistry] = None


def get_health_registry() -> HealthCheckRegistry:
    """Get the global health check registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HealthCheckRegistry()
    return _global_registry
