"""
Audio file serving API endpoints.
"""

import os

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.get("/audio/{audio_id_with_format}")
async def get_audio(audio_id_with_format: str):
    """Serve generated audio by ID with format."""
    from fastapi.responses import FileResponse

    from src.extensions.libraries.tts_engine.api.utils.storage import get_audio_by_id

    # Get the full path to the audio file
    full_path = get_audio_by_id(audio_id_with_format)

    if full_path is None or not os.path.exists(full_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found"
        )

    _, ext = os.path.splitext(audio_id_with_format)
    media_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg"
    }.get(ext.lower(), "audio/wav")

    # Log the file access
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Serving audio file: {full_path}")

    return FileResponse(
        full_path,
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename={audio_id_with_format}"},
    )
