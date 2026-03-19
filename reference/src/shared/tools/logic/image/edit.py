"""Image editing using ComfyUI with Qwen GGUF model.

This module provides image editing functionality through the Qwen Image Edit model.
Import edit functions directly from here for convenience.
"""

from .generation import ImageGenerationServiceHTTP, get_image_service_http

# Re-export the editing service and function
__all__ = [
    "ImageGenerationServiceHTTP",
    "get_image_service_http",
    "edit_image_qwen",
    "edit_character_image",
]


# Convenience function that uses the singleton
async def edit_image_qwen(
    image_data: bytes,
    prompt: str,
    strength: float = 1.0,
    negative_prompt: str = "",
) -> bytes:
    """Edit an image using the Qwen Image Edit GGUF model.

    This is a convenience wrapper that uses the singleton service.

    Args:
        image_data: Source image bytes to edit
        prompt: Edit instruction (e.g., "make her look happy and smiling")
        strength: Edit strength (0-1). For Qwen, this maps to denoise.
                  1.0 = full edit, lower = more faithful to original.
        negative_prompt: Things to avoid in the edit

    Returns:
        Edited image bytes (PNG format)

    Example:
        from logic.image.edit import edit_image_qwen

        with open("input.png", "rb") as f:
            image_data = f.read()

        edited = await edit_image_qwen(
            image_data=image_data,
            prompt="make the person look happy and smiling",
            strength=0.7
        )
    """
    service = get_image_service_http()
    return await service.edit_image_qwen(
        image_data=image_data,
        prompt=prompt,
        strength=strength,
        negative_prompt=negative_prompt,
    )


async def edit_character_image(
    session_id: str,
    prompt: str,
    strength: float = 0.5,
    negative_prompt: str = "",
) -> bytes:
    """Edit a character image using img2img.

    This is a convenience wrapper that uses the singleton service.
    Requires a character reference to be set for the session first.

    Args:
        session_id: Session ID with stored character reference
        prompt: Edit instruction (e.g., "make her look happy and smiling")
        strength: Edit strength (0-1). Lower = more faithful to original.
                  0.2-0.4: Subtle changes
                  0.5-0.7: Moderate changes
                  0.8+: Heavy edits
        negative_prompt: Things to avoid in the edit

    Returns:
        Edited image bytes (PNG format)

    Example:
        from logic.image.edit import edit_character_image
        from logic.image.generation import get_image_service_http

        # First, set a character reference
        service = get_image_service_http()
        with open("character.png", "rb") as f:
            service.set_character_reference("chat_123", f.read())

        # Then edit it
        edited = await edit_character_image(
            session_id="chat_123",
            prompt="make the person look surprised",
            strength=0.5
        )
    """
    service = get_image_service_http()
    return await service.edit_character_image(
        session_id=session_id,
        prompt=prompt,
        strength=strength,
        negative_prompt=negative_prompt,
    )
