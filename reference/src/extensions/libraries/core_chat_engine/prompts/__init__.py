"""Prompts Module - Integration layer for prompt_engine library.

This module provides the connection between the prompt_engine library and
the chat modes. It replaces the old prompt_loader system.

Main entry point:
    from src.extensions.libraries.core_chat_engine.prompts import build_system_messages

    messages = build_system_messages(mode="roleplay", character=char)
"""

from .builder import (
    PromptBuilder,
    build_system_messages,
    get_system_prompt_string,
    get_prompt_builder,
)

__all__ = [
    "PromptBuilder",
    "build_system_messages",
    "get_system_prompt_string",
    "get_prompt_builder",
]
