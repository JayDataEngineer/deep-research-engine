"""Image generation and editing using ComfyUI.

This module provides image generation (z-image-turbo) and editing (Qwen GGUF)
functionality through ComfyUI's HTTP API.

Usage:
    from logic.image import get_image_service_http, edit_image_qwen, edit_character_image

    # Get service and generate
    service = get_image_service_http()
    image_bytes = await service.generate_image("a beautiful sunset")

    # Edit an existing image
    edited = await edit_image_qwen(image_bytes, "make it more vibrant")
"""

from .generation import (
    ImageGenConfig,
    ImageGenerationServiceHTTP,
    get_image_service_http,
    generate_image,
    generate_character_image,
)
from .edit import edit_image_qwen, edit_character_image

__all__ = [
    # Config
    "ImageGenConfig",
    # Service
    "ImageGenerationServiceHTTP",
    "get_image_service_http",
    # Generation
    "generate_image",
    "generate_character_image",
    # Editing
    "edit_image_qwen",
    "edit_character_image",
]
