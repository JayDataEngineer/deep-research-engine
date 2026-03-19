"""Image generation and editing tools.

This module exports LangChain @tool decorated functions for:
- generate_image: Generate images from text using ComfyUI
- edit_image: Edit images using Qwen GGUF model

The actual ComfyUI HTTP API logic is in logic/image/generation.py
"""

from .generation import generate_image, edit_image
from .edit import edit_character_image, set_character_reference, clear_character_reference

__all__ = [
    # New unified tools
    "generate_image",
    "edit_image",
    # Legacy tools (for backward compatibility)
    "edit_character_image",
    "set_character_reference",
    "clear_character_reference",
]
