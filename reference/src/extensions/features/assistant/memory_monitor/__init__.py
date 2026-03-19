"""Memory Monitor Extension.

Provides GPU and system memory monitoring capabilities.
This is an optional extension - if not loaded, memory endpoints will not be available.
"""

from .extension import MemoryMonitorExtension

__all__ = ["MemoryMonitorExtension"]
