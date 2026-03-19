"""Image Captioning Tool

Generates descriptive captions for images using vision models.
Dedicated tool for image captioning (vs general image analysis).
"""

import base64
import httpx
from src.shared.logging_config import get_logger
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from src.constants import GLM_DEFAULT_BASE_URL

logger = get_logger(__name__)

# API Configuration
API_URL = f"{GLM_DEFAULT_BASE_URL}/chat/completions"
MODEL = "glm-4v"

_API_KEY: Optional[str] = (
    os.environ.get("Z_AI_API_KEY")
    or os.environ.get("LLM_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)


def _encode_image(image_source: str) -> str:
    """Encode image to base64.

    Args:
        image_source: URL or file path

    Returns:
        Base64 encoded image
    """
    if image_source.startswith("http"):
        # It's a URL, return as-is
        return image_source

    # It's a file path
    image_path = Path(image_source)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_source}")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@tool
async def image_caption(
    image_source: str,
    detail_level: str = "standard",
    include_emotions: bool = False,
) -> str:
    """Generate a descriptive caption for an image.

    This tool is specifically designed for creating image captions:
    - Generates concise, descriptive captions
    - Focuses on the main subject and action
    - Optionally includes emotional context
    - Works with PNG, JPG, JPEG formats

    Use this when you need to:
    - Generate alt text for images
    - Create captions for image galleries
    - Describe image content for accessibility
    - Index images for search

    Args:
        image_source: URL or file path to the image (PNG, JPG, JPEG)
        detail_level: Level of detail (brief, standard, detailed)
        include_emotions: Whether to include emotional analysis in caption

    Returns:
        Generated image caption as text
    """
    if not _API_KEY:
        return "Error: No API key found. Set Z_AI_API_KEY or LLM_API_KEY environment variable."

    try:
        # Prepare the image
        encoded_image = _encode_image(image_source)

        # Build prompt based on options
        prompt_parts = [
            "Generate a descriptive caption for this image.",
        ]

        if detail_level == "brief":
            prompt_parts.append("Keep it concise (1 sentence).")
        elif detail_level == "detailed":
            prompt_parts.append("Provide a comprehensive description (3-5 sentences).")
        else:  # standard
            prompt_parts.append("Provide a standard description (1-2 sentences).")

        if include_emotions:
            prompt_parts.append("Also note any emotions or mood conveyed.")

        prompt = " ".join(prompt_parts)

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image if encoded_image.startswith("http") else f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                    ],
                }
            ],
            "max_tokens": 500,
        }

        # Make API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        # Extract caption
        caption = result["choices"][0]["message"]["content"].strip()

        # Format output
        output = [f"Caption: {caption}"]
        if detail_level != "brief":
            output.append(f"Detail Level: {detail_level}")
        if include_emotions:
            output.append("(including emotional analysis)")

        return "\n".join(output)

    except httpx.HTTPStatusError as e:
        logger.error(f"Vision API error: {e.response.status_code} - {e.response.text}")
        return f"Error: Vision API returned {e.response.status_code}"
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return f"Error: {str(e)}"


@tool
async def batch_caption(
    image_sources: list[str],
    detail_level: str = "standard",
) -> str:
    """Generate captions for multiple images at once.

    Efficiently processes multiple images and returns their captions.
    Useful for gallery indexing or bulk captioning.

    Args:
        image_sources: List of image URLs or file paths
        detail_level: Level of detail (brief, standard, detailed)

    Returns:
        Formatted captions for all images
    """
    if not image_sources:
        return "Error: No images provided"

    captions = []
    for i, image_source in enumerate(image_sources, 1):
        try:
            caption = await image_caption(image_source, detail_level=detail_level)
            captions.append(f"[{i}] {caption}")
        except Exception as e:
            captions.append(f"[{i}] Error processing image: {str(e)}")

    return "\n\n".join(captions)


# Register tools
__all__ = ["image_caption", "batch_caption"]
