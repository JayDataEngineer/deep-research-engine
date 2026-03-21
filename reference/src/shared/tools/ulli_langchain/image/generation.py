"""Image Generation Tools - LangChain tool wrappers for ComfyUI.

This module provides @tool decorated functions that wrap the logic layer.
The logic layer (logic/image/generation.py) handles ComfyUI HTTP API communication.

Architecture:
    @tool functions (this file) → logic layer → ComfyUI HTTP API
"""

import json
from .logging_config import get_logger
import os
import base64
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import tool

# Add src to path - logic is in sibling directory
src_path = Path(__file__).parent.parent.parent  # Go up to tools/ to include logic/
if str(src_path) not in os.sys.path:
    os.sys.path.insert(0, str(src_path))

from logic.image.generation import get_image_service_http, ImageGenConfig

logger = get_logger(__name__)


# =============================================================================
# Pydantic Schemas for Input Validation
# =============================================================================

class GenerateImageInput(BaseModel):
    """Pydantic schema for generate_image tool input validation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The image description to generate. Be descriptive and specific."
    )
    negative_prompt: str = Field(
        default="ugly, blurry, low quality, distorted, disfigured",
        max_length=500,
        description="Things to avoid in the generated image"
    )
    width: int = Field(
        default=1024,
        ge=512,
        le=2048,
        description="Image width in pixels (512-2048)"
    )
    height: int = Field(
        default=1024,
        ge=512,
        le=2048,
        description="Image height in pixels (512-2048)"
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        le=2**32 - 1,
        description="Random seed (-1 for random)"
    )
    session_id: str = Field(
        default="",
        max_length=100,
        description="Optional session ID for organizing output files"
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty or whitespace only."""
        if not v or v.strip() == "":
            raise ValueError("prompt cannot be empty or whitespace only")
        return v.strip()


class EditImageInput(BaseModel):
    """Pydantic schema for edit_image tool input validation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Edit instruction describing the desired changes"
    )
    image_base64: str = Field(
        default="",
        description="Base64 encoded image to edit (uses session image if not provided)"
    )
    strength: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Edit strength (0.1-1.0). Lower = more faithful to original"
    )
    negative_prompt: str = Field(
        default="",
        max_length=500,
        description="Things to avoid in the edited image"
    )
    session_id: str = Field(
        default="",
        max_length=100,
        description="Session ID to find uploaded image (auto-injected)"
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty or whitespace only."""
        if not v or v.strip() == "":
            raise ValueError("prompt cannot be empty or whitespace only")
        return v.strip()


# =============================================================================
# LangChain Tools
# =============================================================================

@tool(args_schema=GenerateImageInput)
async def generate_image(
    prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, distorted, disfigured",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    session_id: str = "",
) -> str:
    """Generate an image from text description using ComfyUI.

    Uses z-image-turbo model for fast, high-quality image generation.

    Args:
        prompt: Image description - be descriptive and specific. Examples:
                - "A serene landscape with rolling hills at sunset, warm golden lighting"
                - "A futuristic cyberpunk city with neon lights and flying cars"
                - "A cute cat wearing a tiny hat, sitting on a windowsill"
        negative_prompt: Things to avoid (default: "ugly, blurry, low quality")
        width: Image width in pixels (512-2048, default: 1024)
        height: Image height in pixels (512-2048, default: 1024)
        seed: Random seed (-1 for random, or specify for reproducibility)
        session_id: Optional session ID for organizing output files

    Returns:
        JSON string containing:
        - success: bool
        - image_path: str (path to generated image)
        - image_base64: str (base64 encoded image for display)
        - prompt: str (the prompt used)

    Examples:
        generate_image(prompt="A beautiful sunset over mountains")
        generate_image(prompt="A cat wearing a hat", width=768, height=768)
    """
    try:
        service = get_image_service_http()

        logger.info(f"[IMAGE_GEN] Generating: {prompt[:80]}...")

        # Generate image
        image_data = await service.generate_character_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
        )

        # Save organized by session if provided
        output_dir = Path(service.config.output_dir)

        if session_id:
            session_dir = output_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_{seed}.png"
            image_path = session_dir / filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_{seed}.png"
            image_path = output_dir / filename

        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_data)

        # Convert to base64 for frontend
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        logger.info(f"[IMAGE_GEN] Saved: {image_path}")

        # Store as character reference for this session (so it can be edited)
        if session_id:
            service.set_character_reference(session_id, image_data)
            logger.info(f"[IMAGE_GEN] Set character reference for session: {session_id}")

        # Return structured result
        result = {
            "success": True,
            "image_path": str(image_path),
            "image_base64": image_base64,
            "prompt": prompt,
        }
        return json.dumps(result)

    except FileNotFoundError as e:
        logger.error(f"[IMAGE_GEN] Workflow not found: {e}")
        return json.dumps({
            "success": False,
            "error": "ComfyUI workflow not found. Make sure ComfyUI is running."
        })
    except Exception as e:
        logger.error(f"[IMAGE_GEN] Failed: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to generate image: {str(e)}"
        })


@tool(args_schema=EditImageInput)
async def edit_image(
    prompt: str,
    image_base64: str = "",
    strength: float = 0.5,
    negative_prompt: str = "",
    session_id: str = "",
) -> str:
    """Edit an image while preserving the subject's appearance.

    Use this to change expressions, modify poses, or change settings.
    The user must upload an image first, or provide image_base64.

    Args:
        prompt: Edit instruction - be specific about desired changes. Examples:
                - "make the person smile warmly"
                - "change expression to look surprised"
                - "make the background a sunset beach"
                - "make her look happy and cheerful"
        image_base64: Base64 encoded image (uses session image if not provided)
        strength: Edit strength (0.1-1.0):
                 - 0.2-0.4: Subtle changes (small smile, minor adjustment)
                 - 0.5-0.7: Moderate edits (noticeable expression change)
                 - 0.8+: Heavy modifications (significant changes)
        negative_prompt: Things to avoid
        session_id: Session ID to find uploaded image (auto-injected)

    Returns:
        JSON string with edited image data (image_path, image_base64)

    Examples:
        edit_image(prompt="make the person smile", strength=0.4)
        edit_image(prompt="change background to sunset", strength=0.6)
    """
    try:
        service = get_image_service_http()

        logger.info(f"[IMAGE_EDIT] Editing: {prompt[:50]}...")

        # Get image data - either from base64 or from session storage
        if not image_base64:
            # Try to get image from session storage
            from src.api.routes import get_session_image
            session_image = get_session_image(session_id)
            if session_image:
                image_data = session_image
                logger.info(f"[IMAGE_EDIT] Using session image: {len(image_data)} bytes")
            else:
                return json.dumps({
                    "success": False,
                    "error": "No image provided. Upload an image first, then ask to edit it."
                })
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
        filename = f"edit_{timestamp}.png"
        image_path = output_dir / filename
        image_path.write_bytes(edited_image_data)

        # Convert to base64 for frontend
        edited_image_base64 = base64.b64encode(edited_image_data).decode("utf-8")

        logger.info(f"[IMAGE_EDIT] Saved: {image_path}")

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
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"[IMAGE_EDIT] Failed: {e}")
        return json.dumps({"success": False, "error": f"Failed to edit image: {str(e)}"})


# =============================================================================
# Backward Compatibility - Old tool names
# =============================================================================

# Alias for edit_character_image (existing tool that uses session reference)
# The edit_image tool above is more flexible (accepts image_base64 directly)
# edit_character_image is kept for backward compatibility

__all__ = [
    "generate_image",
    "edit_image",
]
