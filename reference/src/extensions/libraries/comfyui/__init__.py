"""ComfyUI Library

Provides shared infrastructure for ComfyUI-based extensions.
Includes base classes, factory, and manager for workflow management.
"""

from .base import BaseComfyUIExtension
from .factory import ComfyUIExtensionFactory, create_extension_from_config
from .manager import ComfyUIManager

__all__ = [
    "BaseComfyUIExtension",
    "ComfyUIExtensionFactory",
    "create_extension_from_config",
    "ComfyUIManager",
]
