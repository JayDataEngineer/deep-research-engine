"""
Speaker Management API for IndexTTS2 External API.

Provides endpoints to manage speaker-to-reference-audio mappings.
Supports both file upload and manual file path configuration.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.extensions.libraries.tts_engine.speakers import SpeakerMapping, get_speaker_manager

router = APIRouter(tags=["External API - IndexTTS Speakers"])
logger = __import__('logging').getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateSpeakerRequest(BaseModel):
    """Request to create a new speaker with manual file path."""

    speaker_id: str = Field(..., description="Unique speaker ID (alphanumeric, underscore, hyphen only)")
    name: str = Field(..., description="Display name for the speaker")
    audio_path: str = Field(..., description="Absolute path to reference audio file")
    description: str | None = Field(None, description="Optional description")
    gender: str | None = Field(None, description="Speaker gender (male/female/neutral)")
    language: str | None = Field(None, description="Speaker language")


class UpdateSpeakerRequest(BaseModel):
    """Request to update a speaker."""

    name: str | None = Field(None, description="New display name")
    audio_path: str | None = Field(None, description="New audio file path")
    description: str | None = Field(None, description="New description")
    gender: str | None = Field(None, description="New gender")
    language: str | None = Field(None, description="New language")
    is_active: bool | None = Field(None, description="New active status")


class SpeakerResponse(BaseModel):
    """Response with speaker details."""

    speaker_id: str
    name: str
    audio_path: str
    description: str | None
    gender: str | None
    language: str | None
    is_active: bool
    audio_exists: bool = Field(False, description="Whether the audio file exists")


class SpeakersListResponse(BaseModel):
    """Response listing all speakers."""

    speakers: list[SpeakerResponse]
    total_count: int
    active_count: int
    storage_dir: str
    audio_storage_dir: str


class SpeakerValidationResponse(BaseModel):
    """Response for speaker validation check."""

    speaker_id: str
    exists: bool
    is_active: bool
    audio_path: str | None
    audio_exists: bool


# ============================================================================
# Helper Functions
# ============================================================================

def _to_response(mapping: SpeakerMapping) -> SpeakerResponse:
    """Convert SpeakerMapping to SpeakerResponse with existence check."""
    return SpeakerResponse(
        **mapping.dict(),
        audio_exists=Path(mapping.audio_path).exists()
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/indextts/speakers", response_model=SpeakerResponse, status_code=status.HTTP_201_CREATED)
async def create_speaker(request: CreateSpeakerRequest) -> SpeakerMapping:
    """
    Create a new speaker mapping with manual file path.

    Use this endpoint when you have the audio file stored externally
    and want to reference it by path.

    **Request Body:**
    ```json
    {
        "speaker_id": "narrator",
        "name": "Narrator Voice",
        "audio_path": "/absolute/path/to/reference.wav",
        "description": "Calm narrator voice",
        "gender": "neutral",
        "language": "en"
    }
    ```

    **Validation:**
    - `speaker_id` must be unique and contain only alphanumeric, underscore, or hyphen
    - `audio_path` must point to an existing file
    - File must be a valid audio format (.wav, .mp3, .flac, .ogg, .m4a)

    **Response:** Returns the created speaker mapping

    **Error Codes:**
    - 409: Speaker ID already exists
    - 404: Audio file not found
    - 400: Invalid speaker ID format or audio file format
    """
    speaker_manager = get_speaker_manager()

    mapping = speaker_manager.add_speaker(
        speaker_id=request.speaker_id,
        name=request.name,
        audio_path=request.audio_path,
        description=request.description,
        gender=request.gender,
        language=request.language,
    )

    logger.info(f"Created speaker '{request.speaker_id}': {request.name}")
    return mapping


@router.post("/indextts/speakers/upload", response_model=SpeakerResponse, status_code=status.HTTP_201_CREATED)
async def upload_speaker_audio(
    speaker_id: str = Form(..., description="Unique speaker ID"),
    name: str = Form(..., description="Display name"),
    description: str | None = Form(None, description="Optional description"),
    gender: str | None = Form(None, description="Speaker gender"),
    language: str | None = Form(None, description="Speaker language"),
    audio_file: UploadFile = File(..., description="Reference audio file to upload"),
) -> SpeakerMapping:
    """
    Create a new speaker by uploading the reference audio file.

    The audio file will be saved to `backend/storage/reference_audio/`
    and automatically referenced by the speaker mapping.

    **Form Fields:**
    - `speaker_id`: Unique speaker ID (alphanumeric, underscore, hyphen only)
    - `name`: Display name
    - `description`: Optional description
    - `gender`: Optional gender (male/female/neutral)
    - `language`: Optional language
    - `audio_file`: Audio file to upload (.wav, .mp3, .flac, .ogg, .m4a)

    **Response:** Returns the created speaker mapping with the saved audio path

    **Error Codes:**
    - 409: Speaker ID already exists
    - 400: Invalid file format or speaker ID format
    """
    speaker_manager = get_speaker_manager()

    # Validate speaker ID format
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', speaker_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Speaker ID can only contain letters, numbers, underscore, and hyphen"
        )

    # Check speaker doesn't exist
    if speaker_manager.speaker_exists(speaker_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Speaker '{speaker_id}' already exists"
        )

    # Read and save uploaded file
    content = await audio_file.read()
    audio_path = speaker_manager.save_uploaded_audio(audio_file.filename, content)

    # Create speaker mapping
    mapping = speaker_manager.add_speaker(
        speaker_id=speaker_id,
        name=name,
        audio_path=audio_path,
        description=description,
        gender=gender,
        language=language,
    )

    logger.info(f"Created speaker '{speaker_id}' with uploaded audio: {audio_path}")
    return mapping


@router.get("/indextts/speakers", response_model=SpeakersListResponse)
async def list_speakers(active_only: bool = False) -> SpeakersListResponse:
    """
    List all speaker mappings.

    **Query Parameters:**
    - `active_only`: If true, only return active speakers (default: false)

    **Response:** Returns list of all speakers with metadata

    **Response Fields:**
    - `speakers`: List of speaker mappings
    - `total_count`: Total number of speakers
    - `active_count`: Number of active speakers
    - `storage_dir`: Path to storage directory
    - `audio_storage_dir`: Path to audio storage directory
    """
    speaker_manager = get_speaker_manager()
    speakers = speaker_manager.list_speakers(active_only=active_only)

    return SpeakersListResponse(
        speakers=[_to_response(s) for s in speakers],
        total_count=len(speakers),
        active_count=sum(1 for s in speakers if s.is_active),
        storage_dir=str(speaker_manager.get_storage_dir()),
        audio_storage_dir=str(speaker_manager.get_audio_storage_dir()),
    )


@router.get("/indextts/speakers/{speaker_id}", response_model=SpeakerResponse)
async def get_speaker(speaker_id: str) -> SpeakerMapping:
    """
    Get details for a specific speaker.

    **Path Parameters:**
    - `speaker_id`: Speaker ID to retrieve

    **Response:** Returns speaker details with audio file existence check

    **Error Codes:**
    - 404: Speaker not found
    """
    speaker_manager = get_speaker_manager()
    mapping = speaker_manager.get_speaker(speaker_id)

    logger.debug(f"Retrieved speaker '{speaker_id}'")
    return mapping


@router.get("/indextts/speakers/{speaker_id}/validate", response_model=SpeakerValidationResponse)
async def validate_speaker(speaker_id: str) -> SpeakerValidationResponse:
    """
    Check if a speaker exists and is valid for synthesis.

    This endpoint validates:
    - Speaker ID exists
    - Speaker is active
    - Audio file exists

    **Path Parameters:**
    - `speaker_id`: Speaker ID to validate

    **Response:** Returns validation status

    **Use Case:** Call this before synthesis to verify the speaker is ready
    """
    speaker_manager = get_speaker_manager()

    exists = speaker_manager.speaker_exists(speaker_id)
    audio_path = None
    audio_exists = False
    is_active = False

    if exists:
        audio_path = speaker_manager.get_audio_path(speaker_id)
        audio_exists = Path(audio_path).exists()
        mapping = speaker_manager.get_speaker(speaker_id)
        is_active = mapping.is_active

    response = SpeakerValidationResponse(
        speaker_id=speaker_id,
        exists=exists,
        is_active=is_active,
        audio_path=audio_path,
        audio_exists=audio_exists,
    )

    logger.debug(f"Validated speaker '{speaker_id}': exists={exists}, audio_exists={audio_exists}")
    return response


@router.put("/indextts/speakers/{speaker_id}", response_model=SpeakerResponse)
async def update_speaker(speaker_id: str, request: UpdateSpeakerRequest) -> SpeakerMapping:
    """
    Update an existing speaker mapping.

    All fields are optional. Only provided fields will be updated.

    **Path Parameters:**
    - `speaker_id`: Speaker ID to update

    **Request Body:** (all optional)
    ```json
    {
        "name": "New name",
        "audio_path": "/new/path/to/audio.wav",
        "description": "Updated description",
        "gender": "female",
        "language": "en",
        "is_active": false
    }
    ```

    **Response:** Returns updated speaker mapping

    **Error Codes:**
    - 404: Speaker not found
    - 404: New audio file not found (if updating audio_path)
    """
    speaker_manager = get_speaker_manager()

    mapping = speaker_manager.update_speaker(
        speaker_id=speaker_id,
        name=request.name,
        audio_path=request.audio_path,
        description=request.description,
        gender=request.gender,
        language=request.language,
        is_active=request.is_active,
    )

    logger.info(f"Updated speaker '{speaker_id}'")
    return mapping


@router.delete("/indextts/speakers/{speaker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_speaker(speaker_id: str) -> None:
    """
    Delete a speaker mapping.

    **Note:** This only removes the mapping. The audio file is NOT deleted
    automatically. You may need to clean up the audio file manually if it's
    in the storage directory.

    **Path Parameters:**
    - `speaker_id`: Speaker ID to delete

    **Response:** 204 No Content on success

    **Error Codes:**
    - 404: Speaker not found
    """
    speaker_manager = get_speaker_manager()
    speaker_manager.remove_speaker(speaker_id)

    logger.info(f"Deleted speaker '{speaker_id}'")


@router.get("/indextts/speakers/{speaker_id}/audio")
async def get_speaker_audio_info(speaker_id: str) -> dict[str, Any]:
    """
    Get detailed information about a speaker's audio file.

    **Path Parameters:**
    - `speaker_id`: Speaker ID

    **Response:** Returns audio file information

    **Response Fields:**
    - `exists`: Whether audio file exists
    - `path`: Full path to audio file
    - `size_bytes`: File size in bytes
    - `filename`: Just the filename
    - `format`: File extension/format

    **Error Codes:**
    - 404: Speaker not found
    """
    speaker_manager = get_speaker_manager()
    audio_path = speaker_manager.get_audio_path(speaker_id)
    path = Path(audio_path)

    if not path.exists():
        return {
            "exists": False,
            "path": audio_path,
            "size_bytes": 0,
            "filename": path.name,
            "format": path.suffix,
        }

    return {
        "exists": True,
        "path": audio_path,
        "size_bytes": path.stat().st_size,
        "filename": path.name,
        "format": path.suffix,
    }
