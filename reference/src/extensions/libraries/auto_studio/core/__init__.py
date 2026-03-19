"""Auto Studio Core.

Exports core abstractions for composable functions and pipelines.
"""

from .composable import (
    BaseComposableFunction,
    Pipeline,
    FunctionResult,
    FunctionInput,
)
from .prompt_manager import ConfigPromptManager

__all__ = [
    "BaseComposableFunction",
    "Pipeline",
    "FunctionResult",
    "FunctionInput",
    "ConfigPromptManager",
]
