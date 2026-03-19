"""
Health check API endpoint.
"""

import logging
from datetime import datetime

from fastapi import APIRouter

from src.config import settings
from src.extensions.libraries.tts_engine.api.schemas.tts import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service is running."""
    try:
        # In a real implementation, you might check:
        # - Database connectivity
        # - External service availability
        # - Model loading status
        # - Memory usage, etc.

        timestamp = datetime.utcnow().isoformat() + "Z"

        logger.debug("Health check performed")

        return HealthResponse(
            status="healthy", version=settings.app_version, timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
