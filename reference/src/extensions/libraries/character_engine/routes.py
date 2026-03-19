"""Character Engine API Routes

FastAPI routes for character management.
Compatible with SillyTavern API endpoints where possible.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .crud import get_character_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/characters", tags=["characters"])

# Redirect /api/characters to /api/characters/ for compatibility
@router.get("", include_in_schema=False)
async def redirect_characters():
    """Redirect /api/characters to /api/characters/ for compatibility."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/characters/")


# =============================================================================
# Request/Response Models
# =============================================================================

class CharacterCreate(BaseModel):
    """Request model for creating a character."""
    name: str
    description: str
    personality: str = ""
    scenario: str = ""
    mes_example: str = ""
    first_mes: str = ""
    creator: str = ""
    character_version: str = "1.0"
    tags: List[str] = []
    creator_notes: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    alternate_greetings: List[str] = []
    personality_summary: str = ""
    character_book: Dict[str, Any] = {}
    extensions: Dict[str, Any] = {}
    avatar: str = ""
    # Voice settings for TTS
    voice_id: Optional[str] = None
    voice_extension: str = "kokoro"
    voice_speed: float = 1.0
    voice_pitch: float = 1.0


class CharacterUpdate(BaseModel):
    """Request model for updating a character."""
    description: Optional[str] = None
    personality: Optional[str] = None
    scenario: Optional[str] = None
    mes_example: Optional[str] = None
    first_mes: Optional[str] = None
    creator: Optional[str] = None
    tags: Optional[List[str]] = None
    creator_notes: Optional[str] = None
    system_prompt: Optional[str] = None
    post_history_instructions: Optional[str] = None
    alternate_greetings: Optional[List[str]] = None
    personality_summary: Optional[str] = None
    # Voice settings for TTS
    voice_id: Optional[str] = None
    voice_extension: Optional[str] = None
    voice_speed: Optional[float] = None
    voice_pitch: Optional[float] = None


class CharacterVoiceUpdate(BaseModel):
    """Request model for updating character voice settings."""
    voice_id: str
    voice_extension: str = "kokoro"
    voice_speed: float = 1.0
    voice_pitch: float = 1.0


class CharacterResponse(BaseModel):
    """Response model for character data."""
    name: str
    description: str
    personality: str = ""
    scenario: str = ""
    mes_example: str = ""
    first_mes: str = ""
    creator: str = ""
    character_version: str = "1.0"
    tags: List[str] = []
    extensions: Dict[str, Any] = {}


# =============================================================================
# Character Endpoints
# =============================================================================

@router.get("/", response_model=List[Dict[str, str]])
async def list_characters():
    """List all available characters.

    Compatible with SillyTavern GET /api/characters endpoint.
    """
    storage = get_character_storage()
    characters = storage.list_characters()
    return characters


@router.get("/{name}")
async def get_character(name: str):
    """Get a specific character by name.

    Compatible with SillyTavern GET /api/characters/{name} endpoint.
    """
    storage = get_character_storage()
    character = storage.load_character(name)

    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

    return JSONResponse(content=character)


@router.post("/")
async def create_character(character: CharacterCreate):
    """Create a new character.

    Compatible with SillyTavern POST /api/characters endpoint.
    """
    storage = get_character_storage()

    # Check if character already exists
    if storage.character_exists(character.name):
        raise HTTPException(status_code=400, detail=f"Character '{character.name}' already exists")

    # Save character
    storage.save_character(character.dict(), format="json")
    return {"name": character.name, "message": "Character created successfully"}


@router.put("/{name}")
async def update_character(name: str, character: CharacterUpdate):
    """Update an existing character.

    Compatible with SillyTavern PUT /api/characters/{name} endpoint.
    """
    storage = get_character_storage()

    # Load existing character
    existing = storage.load_character(name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

    # Update with new values
    update_data = character.dict(exclude_unset=True)
    existing.update(update_data)

    # Save updated character
    storage.save_character(existing, format="json")
    return {"name": name, "message": "Character updated successfully"}


@router.delete("/{name}")
async def delete_character(name: str):
    """Delete a character.

    Compatible with SillyTavern DELETE /api/characters/{name} endpoint.
    """
    storage = get_character_storage()

    if not storage.character_exists(name):
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

    storage.delete_character(name)
    return {"name": name, "message": "Character deleted successfully"}


@router.get("/{name}/download")
async def download_character(name: str, format: str = Query("json", enum=["json", "yaml", "png"])):
    """Download a character card file.

    Args:
        name: Character name
        format: Export format (json, yaml, or png)

    Compatible with SillyTavern GET /api/characters/{name}/download endpoint.
    """
    from .parser import CharacterCard

    storage = get_character_storage()
    character = storage.load_character(name)

    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

    # Convert to CharacterCard model
    card = CharacterCard.from_dict(character)

    if format == "json":
        content = card.to_json()
        filename = f"{name}.json"
        media_type = "application/json"
    elif format == "yaml":
        content = card.to_yaml()
        filename = f"{name}.yaml"
        media_type = "text/yaml"
    elif format == "png":
        content = card.to_png()
        filename = f"{name}.png"
        media_type = "image/png"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    return FileResponse(
        path=content,
        filename=filename,
        media_type=media_type
    )


@router.post("/upload")
async def upload_character(
    file: UploadFile = File(...),
    format: str = Query("json", enum=["json", "yaml", "png"])
):
    """Upload and import a character card.

    Args:
        file: Uploaded character file
        format: File format (json, yaml, or png)

    Compatible with SillyTavern POST /api/characters/upload endpoint.
    """
    from .parser import parse_character_card

    storage = get_character_storage()

    # Read file content
    content = await file.read()

    # Parse character card
    try:
        if format == "png":
            card = parse_character_card(content, format="png")
        elif format == "json":
            card = parse_character_card(content.decode(), format="json")
        elif format == "yaml":
            card = parse_character_card(content.decode(), format="yaml")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse character card: {e}")

    # Check if character already exists
    if storage.character_exists(card.name):
        raise HTTPException(status_code=400, detail=f"Character '{card.name}' already exists")

    # Save character
    storage.save_character(card.to_dict(), format="json")
    return {"name": card.name, "message": "Character imported successfully"}


# =============================================================================
# Voice Association Endpoints
# =============================================================================

@router.get("/{name}/voice")
async def get_character_voice(name: str):
    """Get the voice settings for a character.

    Args:
        name: Character name

    Returns:
        Character voice settings or empty dict if not set
    """
    storage = get_character_storage()
    character = storage.load_character(name)

    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

    voice_settings = {
        "voice_id": character.get("voice_id"),
        "voice_extension": character.get("voice_extension", "kokoro"),
        "voice_speed": character.get("voice_speed", 1.0),
        "voice_pitch": character.get("voice_pitch", 1.0),
    }

    # Only return if voice_id is set
    if not voice_settings["voice_id"]:
        return {
            "name": name,
            "voice": None,
            "message": "No voice configured for this character"
        }

    return {
        "name": name,
        "voice": voice_settings
    }


@router.put("/{name}/voice")
async def set_character_voice(name: str, voice: CharacterVoiceUpdate):
    """Set the voice for a character.

    Args:
        name: Character name
        voice: Voice settings

    Returns:
        Success response
    """
    storage = get_character_storage()
    character = storage.load_character(name)

    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

    # Update character with voice settings
    character["voice_id"] = voice.voice_id
    character["voice_extension"] = voice.voice_extension
    character["voice_speed"] = voice.voice_speed
    character["voice_pitch"] = voice.voice_pitch

    # Save character
    storage.save_character(character, format="json")

    logger.info(f"Set voice '{voice.voice_id}' for character '{name}'")

    return {
        "name": name,
        "voice": {
            "voice_id": voice.voice_id,
            "voice_extension": voice.voice_extension,
            "voice_speed": voice.voice_speed,
            "voice_pitch": voice.voice_pitch,
        },
        "message": "Voice updated successfully"
    }


@router.delete("/{name}/voice")
async def remove_character_voice(name: str):
    """Remove the voice association from a character.

    Args:
        name: Character name

    Returns:
        Success response
    """
    storage = get_character_storage()
    character = storage.load_character(name)

    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")

    # Remove voice settings
    character.pop("voice_id", None)
    character.pop("voice_extension", None)
    character.pop("voice_speed", None)
    character.pop("voice_pitch", None)

    # Save character
    storage.save_character(character, format="json")

    logger.info(f"Removed voice from character '{name}'")

    return {
        "name": name,
        "message": "Voice removed successfully"
    }
