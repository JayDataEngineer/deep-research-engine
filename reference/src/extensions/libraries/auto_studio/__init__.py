"""Auto Studio - Composable Multi-Modal Generation.

Provides a scikit-learn-style pipeline system for building complex
AI media generation workflows.

Key Components:
- BaseComposableFunction: Base class for generation functions
- Pipeline: Chain functions together
- Functions: QwenImageEditFunction, LTX2VideoFunction
- DeepAgentPlanner: LangChain agent with structured output and built-in validation

Example:
    from extensions.libraries.auto_studio import (
        Pipeline,
        QwenImageEditFunction,
        LTX2VideoFunction,
    )

    config = {...}
    pipeline = Pipeline([
        ("qwen_edit", QwenImageEditFunction(config)),
        ("ltx2_video", LTX2VideoFunction(config)),
    ])

    result = await pipeline.run(input_data)
"""

from .core import (
    BaseComposableFunction,
    Pipeline,
    FunctionResult,
    FunctionInput,
    ConfigPromptManager,
)
from .functions import (
    QwenImageEditFunction,
    LTX2VideoFunction,
)

__all__ = [
    "BaseComposableFunction",
    "Pipeline",
    "FunctionResult",
    "FunctionInput",
    "ConfigPromptManager",
    "QwenImageEditFunction",
    "LTX2VideoFunction",
]
