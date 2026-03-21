"""Service Registry - Generic service URL resolution.

Provides a centralized way to resolve service URLs with support for:
1. Explicit override via environment variable
2. Remote mode via shared remote host (Tailscale)
3. Default fallback URL

Usage:
    from .services.registry import get_service_url, ServiceRegistry

    # Simple usage
    url = get_service_url("mcp_web")

    # Check service status
    registry = ServiceRegistry()
    status = await registry.check_service("neo4j")

    # Check all services in a group
    status = await registry.check_group("databases")
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ServiceMode(Enum):
    """How a service URL was resolved."""
    EXPLICIT = "explicit"  # Set via env_override
    REMOTE = "remote"      # Using remote_host_env
    LOCAL = "local"        # Using default_url


@dataclass
class ServiceConfig:
    """Configuration for a single service."""
    name: str
    description: str = ""
    default_url: str = ""
    env_override: Optional[str] = None
    remote_host_env: Optional[str] = None
    remote_port: Optional[int] = None
    health_endpoint: Optional[str] = None
    category: str = "general"

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ServiceConfig":
        return cls(
            name=name,
            description=data.get("description", ""),
            default_url=data.get("default_url", ""),
            env_override=data.get("env_override"),
            remote_host_env=data.get("remote_host_env"),
            remote_port=data.get("remote_port"),
            health_endpoint=data.get("health_endpoint"),
            category=data.get("category", "general"),
        )


@dataclass
class ServiceStatus:
    """Status of a service check."""
    name: str
    url: str
    mode: ServiceMode
    accessible: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class ServiceGroup:
    """A group of related services."""
    name: str
    services: list[str]
    description: str = ""


class ServiceRegistry:
    """Central registry for all external services.

    Loads service definitions from YAML and provides URL resolution
    with support for local/remote modes.
    """

    _instance: Optional["ServiceRegistry"] = None
    _config_path: Optional[Path] = None

    def __new__(cls) -> "ServiceRegistry":
        """Singleton pattern for consistent configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return

        self._services: dict[str, ServiceConfig] = {}
        self._groups: dict[str, ServiceGroup] = {}
        self._remote_hosts: dict[str, dict] = {}
        self._initialized = True
        self._load_config()

    def _load_config(self):
        """Load service configuration from YAML."""
        # Find config file
        if self._config_path:
            config_path = self._config_path
        else:
            # Default location - same directory as this module
            config_path = Path(__file__).parent / "services.yaml"

        if not config_path.exists():
            logger.warning(f"[SERVICE_REGISTRY] Config not found: {config_path}")
            self._load_defaults()
            return

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Load services
            for name, data in config.get("services", {}).items():
                self._services[name] = ServiceConfig.from_dict(name, data)

            # Load groups
            for name, data in config.get("groups", {}).items():
                self._groups[name] = ServiceGroup(
                    name=name,
                    services=data.get("services", []),
                    description=data.get("description", ""),
                )

            # Load remote hosts config
            self._remote_hosts = config.get("remote_hosts", {})

            logger.info(f"[SERVICE_REGISTRY] Loaded {len(self._services)} services, {len(self._groups)} groups")

        except Exception as e:
            logger.error(f"[SERVICE_REGISTRY] Failed to load config: {e}")
            self._load_defaults()

    def _load_defaults(self):
        """Load minimal defaults if config not found."""
        defaults = {
            "mcp_web": ServiceConfig(
                name="mcp_web",
                default_url="http://100.102.244.97:8000/mcp",  # Direct Tailscale IP
                env_override="MCP_WEB_SERVER_URL",
            ),
        }
        self._services.update(defaults)

    @classmethod
    def set_config_path(cls, path: Path):
        """Set custom config path (before first instantiation)."""
        cls._config_path = path

    def get_service(self, name: str) -> Optional[ServiceConfig]:
        """Get service configuration by name."""
        return self._services.get(name)

    def get_url(self, service_name: str) -> str:
        """Resolve service URL with priority: env_override > remote > local.

        Args:
            service_name: Name of the service (e.g., "mcp_web")

        Returns:
            Resolved URL for the service
        """
        service = self._services.get(service_name)
        if not service:
            logger.warning(f"[SERVICE_REGISTRY] Unknown service: {service_name}")
            return ""

        # Priority 1: Explicit environment override
        if service.env_override:
            url = os.getenv(service.env_override)
            if url:
                logger.debug(f"[SERVICE_REGISTRY] {service_name}: explicit ({service.env_override})")
                return url

        # Priority 2: Remote mode (if remote_host_env is set and has value)
        if service.remote_host_env and service.remote_port:
            remote_host = os.getenv(service.remote_host_env, "")
            if remote_host:
                url = f"http://{remote_host}:{service.remote_port}"
                logger.debug(f"[SERVICE_REGISTRY] {service_name}: remote ({remote_host}:{service.remote_port})")
                return url

        # Priority 3: Local mode (default)
        logger.debug(f"[SERVICE_REGISTRY] {service_name}: local")
        return service.default_url

    def get_mode(self, service_name: str) -> ServiceMode:
        """Determine how a service URL would be resolved."""
        service = self._services.get(service_name)
        if not service:
            return ServiceMode.LOCAL

        # Check explicit override
        if service.env_override and os.getenv(service.env_override):
            return ServiceMode.EXPLICIT

        # Check remote mode
        if service.remote_host_env and service.remote_port:
            if os.getenv(service.remote_host_env):
                return ServiceMode.REMOTE

        return ServiceMode.LOCAL

    def get_all_urls(self) -> dict[str, str]:
        """Get resolved URLs for all services."""
        return {name: self.get_url(name) for name in self._services}

    def get_group(self, group_name: str) -> Optional[ServiceGroup]:
        """Get a service group by name."""
        return self._groups.get(group_name)

    def get_group_services(self, group_name: str) -> list[str]:
        """Get list of service names in a group."""
        group = self._groups.get(group_name)
        return group.services if group else []

    async def check_service(self, service_name: str, timeout: float = 2.0) -> ServiceStatus:
        """Check if a service is accessible.

        Args:
            service_name: Name of the service
            timeout: Connection timeout in seconds

        Returns:
            ServiceStatus with accessibility info
        """
        import time

        service = self._services.get(service_name)
        if not service:
            return ServiceStatus(
                name=service_name,
                url="",
                mode=ServiceMode.LOCAL,
                accessible=False,
                error="Unknown service",
            )

        url = self.get_url(service_name)
        mode = self.get_mode(service_name)

        # Parse URL to get host/port
        try:
            # Simple URL parsing
            if url.startswith("http://") or url.startswith("https://"):
                protocol, rest = url.split("://", 1)
                # Remove path
                host_port = rest.split("/")[0]
                if ":" in host_port:
                    host, port_str = host_port.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = host_port
                    port = 443 if protocol == "https" else 80
            elif url.startswith("bolt://"):
                # Neo4j bolt protocol
                rest = url[7:]
                host_port = rest.split("/")[0]
                if ":" in host_port:
                    host, port_str = host_port.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = host_port
                    port = 7687
            elif url.startswith("postgresql://"):
                # PostgreSQL - extract host and port from connection string
                rest = url[13:]
                # Format: user:pass@host:port/db
                if "@" in rest:
                    _, host_part = rest.split("@", 1)
                else:
                    host_part = rest
                host_port = host_part.split("/")[0]
                if ":" in host_port:
                    host, port_str = host_port.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = host_port
                    port = 5432
            else:
                return ServiceStatus(
                    name=service_name,
                    url=url,
                    mode=mode,
                    accessible=False,
                    error=f"Unknown URL format: {url}",
                )
        except Exception as e:
            return ServiceStatus(
                name=service_name,
                url=url,
                mode=mode,
                accessible=False,
                error=f"URL parse error: {e}",
            )

        # Try to connect
        start_time = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            latency = (time.time() - start_time) * 1000

            accessible = result == 0
            error = None if accessible else f"Connection refused (error {result})"

            return ServiceStatus(
                name=service_name,
                url=url,
                mode=mode,
                accessible=accessible,
                error=error,
                latency_ms=round(latency, 1),
            )

        except socket.timeout:
            return ServiceStatus(
                name=service_name,
                url=url,
                mode=mode,
                accessible=False,
                error="Connection timeout",
                latency_ms=timeout * 1000,
            )
        except Exception as e:
            return ServiceStatus(
                name=service_name,
                url=url,
                mode=mode,
                accessible=False,
                error=str(e),
            )

    async def check_group(self, group_name: str, timeout: float = 2.0) -> dict[str, ServiceStatus]:
        """Check all services in a group.

        Args:
            group_name: Name of the group
            timeout: Connection timeout per service

        Returns:
            Dict mapping service names to their status
        """
        service_names = self.get_group_services(group_name)
        tasks = [self.check_service(name, timeout) for name in service_names]
        results = await asyncio.gather(*tasks)
        return {status.name: status for status in results}

    def is_remote_enabled(self, service_name: str) -> bool:
        """Check if a service is configured for remote mode."""
        return self.get_mode(service_name) == ServiceMode.REMOTE

    def get_remote_host(self, service_name: str) -> Optional[str]:
        """Get the remote host for a service (if configured)."""
        service = self._services.get(service_name)
        if service and service.remote_host_env:
            return os.getenv(service.remote_host_env)
        return None

    def list_services(self) -> list[str]:
        """List all registered service names."""
        return list(self._services.keys())

    def list_groups(self) -> list[str]:
        """List all service group names."""
        return list(self._groups.keys())


# Module-level convenience functions
_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """Get the singleton ServiceRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


def get_service_url(service_name: str) -> str:
    """Resolve a service URL.

    Args:
        service_name: Name of the service (e.g., "mcp_web", "neo4j")

    Returns:
        Resolved URL for the service
    """
    return get_registry().get_url(service_name)


def get_service_mode(service_name: str) -> ServiceMode:
    """Get the resolution mode for a service."""
    return get_registry().get_mode(service_name)


def is_remote_mode(service_name: str) -> bool:
    """Check if a service is using remote mode."""
    return get_registry().is_remote_enabled(service_name)


async def check_service(service_name: str, timeout: float = 2.0) -> ServiceStatus:
    """Check if a service is accessible."""
    return await get_registry().check_service(service_name, timeout)


async def check_service_group(group_name: str, timeout: float = 2.0) -> dict[str, ServiceStatus]:
    """Check all services in a group."""
    return await get_registry().check_group(group_name, timeout)
