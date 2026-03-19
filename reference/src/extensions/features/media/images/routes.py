"""Image Generation API Routes.

Provides image generation and character editing endpoints via ComfyUI.
"""

import base64
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from src.shared.tools.logic.image.generation import get_image_service_http as get_image_service
from src.services.session_images import get_image_store


router = APIRouter(prefix="/api/v1/images", tags=["images"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SetReferenceRequest(BaseModel):
    """Request to set character reference image."""
    session_id: str = Field(..., description="Unique session identifier")
    image_data: str = Field(..., description="Base64 encoded image data")


class EditCharacterRequest(BaseModel):
    """Request to edit character image."""
    session_id: str = Field(..., description="Session with character reference")
    prompt: str = Field(..., description="Edit instruction")
    strength: float = Field(default=0.5, ge=0, le=1, description="Edit strength (0-1)")


class GenerateCharacterRequest(BaseModel):
    """Request to generate new character image."""
    prompt: str = Field(..., description="Character description")
    negative_prompt: str = Field(default="", description="What to avoid")
    width: int = Field(default=512, ge=16, le=8192, description="Image width (16-8192)")
    height: int = Field(default=512, ge=16, le=8192, description="Image height (16-8192)")
    steps: int = Field(default=4, ge=1, le=50, description="Sampling steps")


class SetReferenceResponse(BaseModel):
    """Response to set reference request."""
    success: bool
    message: str


class ImageResponse(BaseModel):
    """Response with generated image."""
    image: str  # Base64 data URL
    prompt: str
    message: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/reference/set", response_model=SetReferenceResponse)
async def set_character_reference(request: SetReferenceRequest):
    """Set or update the character reference image for a session.

    This image will be used as the base for all Qwen Image Edit operations,
    ensuring character consistency across edits.
    """
    try:
        service = get_image_service()

        # Decode base64 image
        if "," in request.image_data:
            image_data = request.image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)

        # Store reference
        service.set_character_reference(request.session_id, image_bytes)

        return SetReferenceResponse(
            success=True,
            message="Character reference image saved",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reference/{session_id}")
async def clear_character_reference(session_id: str):
    """Remove the character reference image for a session."""
    service = get_image_service()
    service.clear_character_reference(session_id)
    return {"success": True, "message": "Character reference cleared"}


@router.post("/edit", response_model=ImageResponse)
async def edit_character_image(request: EditCharacterRequest):
    """Edit a character image while preserving appearance.

    Uses Qwen Image Edit to modify the character's expression, pose,
    or setting while maintaining consistent appearance.

    This runs ASYNCHRONOUSLY - may take several seconds.
    """
    try:
        service = get_image_service()

        # Generate edited image
        image_bytes = await service.edit_character_image(
            session_id=request.session_id,
            prompt=request.prompt,
            strength=request.strength,
        )

        # Convert to base64
        image_base64 = service.image_to_base64(image_bytes)

        return ImageResponse(
            image=image_base64,
            prompt=request.prompt,
            message="Character image edited successfully",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=ImageResponse)
async def generate_character_image(request: GenerateCharacterRequest):
    """Generate a new character image from scratch.

    Uses Z-Image Turbo for fast generation.

    This runs ASYNCHRONOUSLY - may take several seconds.
    """
    try:
        service = get_image_service()

        # Generate image
        image_bytes = await service.generate_character_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            sample_steps=request.steps,
        )

        # Convert to base64
        image_base64 = service.image_to_base64(image_bytes)

        return ImageResponse(
            image=image_base64,
            prompt=request.prompt,
            message="Character image generated successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_reference_image(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload and set character reference image.

    Convenience endpoint that handles file upload and reference setting in one call.
    """
    try:
        service = get_image_service()

        # Read file
        image_bytes = await file.read()

        # Store reference
        service.set_character_reference(session_id, image_bytes)

        return SetReferenceResponse(
            success=True,
            message=f"Reference image saved: {file.filename}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reference/{session_id}")
async def get_character_reference(session_id: str):
    """Get the character reference image for a session.

    Returns the base64-encoded reference image or 404 if not set.
    """
    service = get_image_service()
    ref_image = service.get_character_reference(session_id)

    if ref_image is None:
        raise HTTPException(status_code=404, detail="No character reference set for this session")

    image_base64 = service.image_to_base64(ref_image)
    return {"image": image_base64, "session_id": session_id}


# =============================================================================
# Session Last Image Management
# =============================================================================

@router.get("/last/{session_id}")
async def get_last_generated_image(session_id: str):
    """Get the last generated image for a session.

    Returns the base64-encoded last image with its prompt, or 404 if not set.
    This allows users to edit the last generated image without re-uploading.
    """
    image_store = get_image_store()
    image_data = image_store.get_image(session_id)

    if image_data is None:
        raise HTTPException(
            status_code=404,
            detail="No generated image found for this session. Image may have expired."
        )

    return {
        "image": f"data:image/png;base64,{image_data['image_base64']}",
        "prompt": image_data.get("prompt", ""),
        "image_path": image_data.get("image_path", ""),
        "session_id": session_id,
        "timestamp": image_data["timestamp"].isoformat(),
    }


class EditLastImageRequest(BaseModel):
    """Request to edit the last generated image."""
    session_id: str = Field(..., description="Session with the last generated image")
    edit_prompt: str = Field(..., description="Edit instruction (e.g., 'make her smile', 'change pose')")
    strength: float = Field(default=0.5, ge=0, le=1, description="Edit strength (0-1)")


@router.post("/last/edit", response_model=ImageResponse)
async def edit_last_generated_image(request: EditLastImageRequest):
    """Edit the last generated image for a session.

    Uses the last generated image as the base for Qwen Image Edit,
    allowing users to iteratively refine images without re-uploading.

    Example edits:
    - "make her smile"
    - "change to a sad expression"
    - "make her look at the camera"
    - "change the background to a sunset"
    """
    image_store = get_image_store()
    image_data = image_store.get_image(request.session_id)

    if image_data is None:
        raise HTTPException(
            status_code=404,
            detail="No generated image found for this session. Generate an image first."
        )

    try:
        service = get_image_service()

        # Decode the stored image
        image_bytes = base64.b64decode(image_data["image_base64"])

        # Use as character reference for editing
        temp_session_id = f"edit_temp_{request.session_id}"
        service.set_character_reference(temp_session_id, image_bytes)

        # Generate edited image
        edited_bytes = await service.edit_character_image(
            session_id=temp_session_id,
            prompt=request.edit_prompt,
            strength=request.strength,
        )

        # Update store with new image
        edited_base64 = service.image_to_base64(edited_bytes)
        image_store.store_image(
            request.session_id,
            edited_base64,
            prompt=f"{image_data.get('prompt', '')} -> {request.edit_prompt}",
        )

        # Clean up temp session
        service.clear_character_reference(temp_session_id)

        return ImageResponse(
            image=edited_base64,
            prompt=request.edit_prompt,
            message="Image edited successfully. You can continue editing with more changes.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/last/{session_id}")
async def clear_last_generated_image(session_id: str):
    """Clear the last generated image for a session."""
    image_store = get_image_store()
    cleared = image_store.clear_image(session_id)
    return {
        "success": cleared,
        "message": "Last generated image cleared" if cleared else "No image to clear"
    }
