"""
Extension Manager Subsystem

Provides modular extension management with separate concerns:
- ExtensionManager: Main orchestrator
- ExtensionRegistry: Extension storage and lookups
- ExtensionDiscoverer: Filesystem discovery
- ExtensionLoader: Instantiation and dependency checking
- CapabilityQuery: Frontend capability queries
- ExtensionOperations: Runtime operations (reload, health, etc.)
"""

from .manager import ExtensionManager
from .registry import ExtensionRegistry

__all__ = [
    "ExtensionManager",
    "ExtensionRegistry",
]
