"""
Extension System

Universal extension architecture for the Unified LLM Interface.
"""

from .core import (
    ExtensionType,
    ExtensionStatus,
    BaseExtension,
    LogicExtension,
    JobExtension,
    ServiceExtension,
    ExtensionContext,
    ExtensionManager,
    ExtensionLifecycle,
)

__all__ = [
    "ExtensionType",
    "ExtensionStatus",
    "BaseExtension",
    "LogicExtension",
    "JobExtension",
    "ServiceExtension",
    "ExtensionContext",
    "ExtensionManager",
    "ExtensionLifecycle",
]
