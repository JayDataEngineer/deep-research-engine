"""Base Features Infrastructure

Provides common base classes for CLI and Manager implementations.
"""

from .cli_base import BaseCLI, ResponseFormatter, create_cli_cli
from .manager_base import BaseManager, BaseItem, SimpleItem, ManagerWithBuiltinItems

__all__ = [
    "BaseCLI",
    "ResponseFormatter",
    "create_cli_cli",
    "BaseManager",
    "BaseItem",
    "SimpleItem",
    "ManagerWithBuiltinItems",
]
