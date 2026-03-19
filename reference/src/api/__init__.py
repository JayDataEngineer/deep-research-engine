"""API endpoints module."""

# SillyTavern-compatible Prompt Manager
# NOTE: Moved to src.extensions.libraries.prompt_engine
from src.extensions.libraries.prompt_engine.prompt_manager import (
    PromptManager,
    get_prompt_manager,
    create_prompt_manager,
)

# SillyTavern-compatible Prompt Models
from src.extensions.libraries.prompt_engine.prompt_models import (
    Prompt,
    PromptOrderEntry,
    PromptCollection,
    PromptExport,
    PromptExportData,
    CharacterPromptConfig,
    InjectionPosition,
    PromptRole,
    get_default_prompts,
    get_default_prompt_order,
)

# SillyTavern Prompt Manager Integration
from src.extensions.libraries.prompt_engine.prompt_integration import (
    PromptManagerIntegration,
    get_prompt_integration,
    inject_prompts_for_character,
    build_system_prompt,
)

__all__ = [
    # Prompt Manager
    "PromptManager",
    "get_prompt_manager",
    "create_prompt_manager",
    # Prompt Models
    "Prompt",
    "PromptOrderEntry",
    "PromptCollection",
    "PromptExport",
    "PromptExportData",
    "CharacterPromptConfig",
    "InjectionPosition",
    "PromptRole",
    "get_default_prompts",
    "get_default_prompt_order",
    # Integration
    "PromptManagerIntegration",
    "get_prompt_integration",
    "inject_prompts_for_character",
    "build_system_prompt",
]
