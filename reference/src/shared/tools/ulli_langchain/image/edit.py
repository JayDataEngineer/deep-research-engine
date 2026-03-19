"""Qwen Image Edit tool for character-consistent image modification.

This tool allows the AI to edit character images while preserving
character appearance and style, using img2img techniques.
"""

import json
from src.shared.logging_config import get_logger
import os
import base64
from pathlib import Path
from typing import Optional
from datetime import datetime

from langchain_core.tools import tool

# Add src to path - logic is in sibling directory
src_path = Path(__file__).parent.parent.parent  # Go up to tools/ to include logic/
if str(src_path) not in os.sys.path:
    os.sys.path.insert(0, str(src_path))

from logic.image.generation import get_image_service_http, ImageGenConfig

logger = get_logger(__name__)


# Global session registry for tracking character references
# In production, this would be stored in Redis or a database
_character_references: dict[str, bytes] = {}


@tool
async def edit_character_image(
    prompt: str,
    image_base64: str = "",
    strength: float = 0.5,
    negative_prompt: str = "",
    session_id: str = "",
) -> str:
    """Edit an uploaded image while preserving the subject's appearance.

    Use this to change expressions (happy, sad, surprised), modify poses (sitting, standing),
    or change settings (backgrounds, lighting).

    IMPORTANT: The user must upload an image first using the image upload button.
    Once an image is uploaded, this tool will automatically use it.

    Args:
        prompt: Edit instruction describing the desired changes. Be specific about what you want.
               Examples: 'make the person smile', 'change expression to look surprised',
               'make her look happy and warm', 'change the background to a sunset beach'.
        image_base64: Base64 encoded image to edit (with or without data:image prefix). If not provided,
                       uses the most recently uploaded image for this session.
        strength: How much to change the image (0-1). Lower values stay closer to original appearance.
                 0.2-0.4: Subtle changes (small smile, minor adjustment)
                 0.5-0.7: Moderate edits (noticeable expression change, different pose)
                 0.8+: Heavy modifications (significant changes, more artistic interpretation)
        negative_prompt: Things to avoid in the edited image. Examples: 'blurry, ugly, distorted'
        session_id: Session ID (auto-injected) to find the uploaded image. Used to locate the
                     most recent image for this chat session.

    Returns:
        JSON string containing the edited image data (image_path, image_base64) that the
        frontend will display.

    Example:
        After user uploads a photo:
        edit_character_image(prompt="make the person look happy with a warm smile", strength=0.4)
    """
    try:
        # Get image service
        service = get_image_service_http()

        logger.info(f"[IMAGE_EDIT] Editing image: {prompt[:50]}...")

        # Get image data - either from base64 or from session storage
        if not image_base64:
            # Try to get image from session storage
            from src.api.routes import get_session_image
            session_image = get_session_image(session_id)
            if session_image:
                image_data = session_image
                logger.info(f"[IMAGE_EDIT] Using session image for {session_id}: {len(image_data)} bytes")
            else:
                return (
                    "Error: No image provided. "
                    "Please upload an image first using the image upload button, "
                    "then ask me to edit it (e.g., 'make the person smile')."
                )
        else:
            # Convert base64 to image bytes
            image_data = service.base64_to_image(image_base64)

        # Perform the edit
        edited_image_data = await service.edit_image_qwen(
            image_data=image_data,
            prompt=prompt,
            strength=strength,
            negative_prompt=negative_prompt,
        )

        # Save the edited image
        output_dir = Path(service.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"edit_qwen_{timestamp}.png"
        image_path = output_dir / filename
        image_path.write_bytes(edited_image_data)

        # Convert to base64 for frontend
        edited_image_base64 = base64.b64encode(edited_image_data).decode("utf-8")

        logger.info(f"[IMAGE_EDIT] Saved edited image: {image_path}")

        # Return structured result
        result = {
            "success": True,
            "image_path": str(image_path),
            "image_base64": edited_image_base64,
            "edit_prompt": prompt,
            "strength": strength,
        }
        return json.dumps(result)

    except ValueError as e:
        logger.error(f"[IMAGE_EDIT] Value error: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"[IMAGE_EDIT] Failed: {e}")
        return f"Failed to edit character image: {str(e)}"


@tool
async def set_character_reference(
    session_id: str,
    image_base64: str,
) -> str:
    """Set a character reference image for future editing.

    This stores a character image that will be used as the base for
    future edit_character_image calls. This is automatically called
    after generate_image, but can be used manually for custom references.

    Args:
        session_id: Session ID (chat ID) to associate with this character reference.
                    Used to link the reference to a specific conversation.
        image_base64: Base64 encoded image data (with or without data:image/png;base64 prefix).
                     The image data of the character to use as reference for edits.

    Returns:
        Success message confirming the character reference was stored.
    """
    try:
        service = get_image_service_http()
        image_data = service.base64_to_image(image_base64)
        service.set_character_reference(session_id, image_data)
        logger.info(f"[IMAGE_EDIT] Set character reference for session: {session_id}")
        return f"Character reference set for session: {session_id}"
    except Exception as e:
        logger.error(f"[IMAGE_EDIT] Failed to set reference: {e}")
        return f"Failed to set character reference: {str(e)}"


@tool
def clear_character_reference(
    session_id: str,
) -> str:
    """Clear the character reference for a session.

    Use this to forget the current character and start fresh with a new reference.

    Args:
        session_id: Session ID (chat ID) to clear the character reference from.

    Returns:
        Success message confirming the character reference was cleared.
    """
    try:
        service = get_image_service_http()
        service.clear_character_reference(session_id)
        logger.info(f"[IMAGE_EDIT] Cleared character reference for session: {session_id}")
        return f"Character reference cleared for session: {session_id}"
    except Exception as e:
        logger.error(f"[IMAGE_EDIT] Failed to clear reference: {e}")
        return f"Failed to clear character reference: {str(e)}"
