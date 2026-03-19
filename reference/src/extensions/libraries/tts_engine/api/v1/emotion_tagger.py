"""
Emotion Tagger API endpoints.

Provides AI-powered emotion detection and tagging for expressive TTS.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.extensions.libraries.tts_engine.emotion_tagger import (
    EmotionTag,
    get_emotion_tagger,
)

# Get emotion_tagger instance (lazy initialization to avoid recursion during import)
_emotion_tagger = None

def emotion_tagger():
    global _emotion_tagger
    if _emotion_tagger is None:
        _emotion_tagger = get_emotion_tagger()
    return _emotion_tagger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion", tags=["emotion-tagger"])


class EmotionTagRequest(BaseModel):
    """Request for emotion tagging."""
    text: str = Field(..., description="Text to analyze for emotions", min_length=1)
    max_tags: int = Field(3, description="Maximum number of emotion tags to suggest", ge=1, le=10)


class EmotionTagResponse(BaseModel):
    """Response for emotion tagging."""
    original: str
    tagged: str
    tags: list[dict]
    available_emotions: dict


class EmotionSuggestionRequest(BaseModel):
    """Request for emotion suggestions."""
    text: str = Field(..., description="Text to analyze", min_length=1)
    max_suggestions: int = Field(5, description="Maximum number of suggestions", ge=1, le=10)


class ApplyTagsRequest(BaseModel):
    """Request to apply specific emotion tags."""
    text: str = Field(..., description="Original text", min_length=1)
    tags: list[dict] = Field(..., description="Emotion tags to apply")


class RemoveTagsRequest(BaseModel):
    """Request to remove emotion tags."""
    text: str = Field(..., description="Text with emotion tags to remove", min_length=1)


@router.post("/suggest", response_model=dict)
async def suggest_emotions(request: EmotionSuggestionRequest):
    """
    Suggest emotion tags for the given text using GLM-4.7.

    Analyzes text and returns emotion suggestions with positions and intensities.

    Args:
        request: EmotionSuggestionRequest with text and max_suggestions

    Returns:
        Dictionary with suggested emotion tags
    """
    try:
        tags = await emotion_tagger().suggest_emotions(request.text, request.max_suggestions)

        return {
            "text": request.text,
            "suggestions": [
                {
                    "emotion": tag.emotion,
                    "position": tag.position,
                    "intensity": tag.intensity,
                    "reason": tag.reason,
                }
                for tag in tags
            ],
            "count": len(tags),
        }

    except Exception as e:
        logger.error(f"Error suggesting emotions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to suggest emotions: {str(e)}",
        ) from e


@router.post("/auto-tag", response_model=EmotionTagResponse)
async def auto_tag_text(request: EmotionTagRequest):
    """
    Automatically tag text with suggested emotions.

    Applies AI-detected emotion tags directly to the text.

    Args:
        request: EmotionTagRequest with text and max_tags

    Returns:
        TaggedText with original and tagged versions
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty",
            )

        result = emotion_tagger().auto_tag_text(request.text, request.max_tags)

        return EmotionTagResponse(
            original=result.original,
            tagged=result.tagged,
            tags=[
                {
                    "emotion": tag.emotion,
                    "position": tag.position,
                    "intensity": tag.intensity,
                    "reason": tag.reason,
                }
                for tag in result.tags
            ],
            available_emotions=emotion_tagger().get_available_emotions(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error auto-tagging text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to auto-tag text: {str(e)}",
        ) from e


@router.post("/apply-tags", response_model=EmotionTagResponse)
async def apply_tags(request: ApplyTagsRequest):
    """
    Apply specific emotion tags to text.

    Args:
        request: ApplyTagsRequest with text and tags

    Returns:
        TaggedText with applied tags
    """
    try:
        # Convert dict tags to EmotionTag objects
        tags = []
        for tag_dict in request.tags:
            tags.append(
                EmotionTag(
                    emotion=tag_dict.get("emotion", "neutral"),
                    position=tag_dict.get("position", 0),
                    intensity=tag_dict.get("intensity", 1.0),
                )
            )

        result = emotion_tagger().apply_tags(request.text, tags)

        return EmotionTagResponse(
            original=result.original,
            tagged=result.tagged,
            tags=[
                {
                    "emotion": tag.emotion,
                    "position": tag.position,
                    "intensity": tag.intensity,
                    "reason": tag.reason,
                }
                for tag in result.tags
            ],
            available_emotions=emotion_tagger().get_available_emotions(),
        )

    except Exception as e:
        logger.error(f"Error applying tags: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply tags: {str(e)}",
        ) from e


@router.post("/remove-tags")
async def remove_tags(request: RemoveTagsRequest):
    """
    Remove all emotion tags from text.

    Args:
        request: RemoveTagsRequest with tagged text

    Returns:
        Clean text without emotion tags
    """
    try:
        cleaned = emotion_tagger().remove_tags(request.text)

        return {
            "original": request.text,
            "cleaned": cleaned,
        }

    except Exception as e:
        logger.error(f"Error removing tags: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove tags: {str(e)}",
        ) from e


@router.get("/emotions")
async def get_available_emotions():
    """
    Get list of available emotions.

    Returns:
        Dictionary of available emotions with descriptions
    """
    try:
        emotions = emotion_tagger().get_available_emotions()

        return {
            "emotions": emotions,
            "count": len(emotions),
        }

    except Exception as e:
        logger.error(f"Error getting available emotions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available emotions: {str(e)}",
        ) from e


@router.get("/status")
async def get_tagger_status():
    """
    Get the status of the emotion tagger.

    Returns:
        Status information including API availability
    """
    import os
    from src.constants import ZAI_DEFAULT_BASE_URL

    has_api_key = bool(os.getenv("ZAI_API_KEY"))
    base_url = os.getenv("ZAI_BASE_URL", ZAI_DEFAULT_BASE_URL)

    return {
        "available": emotion_tagger().client is not None,
        "mode": "ai" if has_api_key else "rule-based-fallback",
        "api_configured": has_api_key,
        "base_url": base_url if has_api_key else None,
    }
