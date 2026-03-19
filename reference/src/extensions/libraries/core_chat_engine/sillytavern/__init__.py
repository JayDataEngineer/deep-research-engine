"""
SillyTavern Features for Companion Brain

This module contains SillyTavern-compatible features:
- Special commands (OOC, story mode, inline commands)
- Context templates (W++, STDP, Boostyle)
- Chat templates (Alpaca, ChatML, Vicuna)
- Group chats
- Macros
- Tokenizer integration
- Presets system
"""

from .special_commands import (
    SpecialCommandResult,
    detect_special_commands,
    strip_special_commands,
    process_ooc_content,
)
from .group_chats import (
    GroupChat,
    GroupCharacter,
    GroupTurn,
    TurnOrder,
    create_group_chat,
)
from .macros import (
    Macro,
    MacroManager,
    create_macro_manager,
)
from .tokenizer import (
    Tokenizer,
    TokenBudget,
    create_tokenizer,
    create_token_budget,
)
from .chat_templates import (
    ChatTemplate,
    ChatTemplateManager,
    create_chat_template_manager,
)
from .context_templates import (
    ContextTemplate,
    ContextTemplateManager,
    create_context_template_manager,
)
from .inline_commands import (
    InlineCommandType,
    InlineCommand,
    InlineCommandProcessor,
    create_inline_command_processor,
)
from .presets import (
    PresetCategory,
    GenerationPreset,
    FormatPreset,
    AuthorNotePreset,
    CharacterPreset,
    CompletePreset,
    PresetManager,
    create_preset_manager,
    create_generation_preset,
    create_format_preset,
)

__all__ = [
    # Special commands
    "SpecialCommandResult",
    "detect_special_commands",
    "strip_special_commands",
    "process_ooc_content",
    # Group chats
    "GroupChat",
    "GroupCharacter",
    "GroupTurn",
    "TurnOrder",
    "create_group_chat",
    # Macros
    "Macro",
    "MacroManager",
    "create_macro_manager",
    # Tokenizer
    "Tokenizer",
    "TokenBudget",
    "create_tokenizer",
    "create_token_budget",
    # Chat templates
    "ChatTemplate",
    "ChatTemplateManager",
    "create_chat_template_manager",
    # Context templates
    "ContextTemplate",
    "ContextTemplateManager",
    "create_context_template_manager",
    # Inline commands
    "InlineCommandType",
    "InlineCommand",
    "InlineCommandProcessor",
    "create_inline_command_processor",
    # Presets
    "PresetCategory",
    "GenerationPreset",
    "FormatPreset",
    "AuthorNotePreset",
    "CharacterPreset",
    "CompletePreset",
    "PresetManager",
    "create_preset_manager",
    "create_generation_preset",
    "create_format_preset",
]
