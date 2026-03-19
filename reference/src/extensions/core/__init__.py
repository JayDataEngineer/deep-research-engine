"""
Extension System Core

Provides the base classes and infrastructure for the universal extension architecture.
"""

from .base import (
    ExtensionType,
    ExtensionStatus,
    BaseExtension,
    LogicExtension,
    JobExtension,
    ServiceExtension,
    TTSExtensionInterface,
    ChatModeExtension,
    TextModeExtension,
    FailedExtension,
)
from .context import ExtensionContext
from .lifecycle import ExtensionLifecycle
from .manager import ExtensionManager
from .mixins import ChatHelperMixin
from .tts_base import (
    SubprocessServiceExtension,
    TTSExtensionBase,
    create_tts_extension_class,
)

__all__ = [
    "ExtensionType",
    "ExtensionStatus",
    "BaseExtension",
    "LogicExtension",
    "JobExtension",
    "ServiceExtension",
    "TTSExtensionInterface",
    "ChatModeExtension",
    "TextModeExtension",
    "FailedExtension",
    "ExtensionContext",
    "ExtensionManager",
    "ExtensionLifecycle",
    "ChatHelperMixin",
    "SubprocessServiceExtension",
    "TTSExtensionBase",
    "create_tts_extension_class",
]

# Note: ResearchExtension is not imported here to avoid loading deep_research dependencies
# Import it directly from src.extensions.core.base_research when needed
