"""System Router - Health checks, configuration, and system-level endpoints.

These endpoints provide system information and diagnostics.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.config import settings
from src.constants import ROOT_EXTENSIONS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["system"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    extensions_dir: str | None = None
    extensions_count: int = 0


class ConfigResponse(BaseModel):
    """Public configuration response."""
    app_name: str
    version: str
    llm_model: str
    llm_base_url: str | None = None
    features: list[str]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns system health status and basic information.
    """
    # Check if extensions directory exists
    extensions_dir = None
    extensions_count = 0

    if ROOT_EXTENSIONS_DIR.exists():
        extensions_dir = str(ROOT_EXTENSIONS_DIR)
        # Count extension folders
        if ROOT_EXTENSIONS_DIR.is_dir():
            extensions_count = len([
                d for d in ROOT_EXTENSIONS_DIR.iterdir()
                if d.is_dir() and not d.name.startswith("_")
            ])

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        extensions_dir=extensions_dir,
        extensions_count=extensions_count,
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get public configuration.

    Returns non-sensitive configuration information for the frontend.
    """
    # Detect available features based on extensions
    features = ["chat"]  # Always available

    # Check for TTS
    if os.environ.get("TTS_AVAILABLE") == "true":
        features.append("tts")

    # Check for deep research
    deep_research_path = ROOT_EXTENSIONS_DIR / "deep_research"
    if deep_research_path.exists():
        features.append("deep_research")

    # Check for vision
    features.append("vision")  # Assuming vision is always available

    return ConfigResponse(
        app_name="Unified LLM Interface",
        version="0.1.0",
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url if settings.show_llm_config else None,
        features=features,
    )


@router.get("/extensions")
async def list_extensions(request: Request):
    """List all loaded extensions.

    Returns information about extensions loaded by the ExtensionManager.
    The response includes both 'extensions' (full info) and 'available_extensions'
    (simple list for frontend compatibility).
    """
    extension_manager = getattr(request.app.state, "extension_manager", None)

    if not extension_manager:
        return {
            "extensions": [],
            "available_extensions": [],
            "message": "Extension manager not initialized"
        }

    extensions = []
    available_extensions = []
    for ext_id, ext in extension_manager.extensions.items():
        ext_info = {
            "id": ext_id,
            "name": getattr(ext, "name", ext_id),
            "type": type(ext).__name__,
            "active": getattr(ext, "active", True),
        }
        extensions.append(ext_info)
        available_extensions.append(ext_id)

    return {
        "extensions": extensions,
        "available_extensions": available_extensions,
    }


@router.get("/capabilities")
async def get_capabilities(request: Request):
    """Get extension capabilities for the frontend.

    Returns which extensions are available and what they provide.
    This endpoint is also available at /api/extensions/capabilities via redirect.
    """
    extension_manager = getattr(request.app.state, "extension_manager", None)

    if not extension_manager:
        return {
            "modes": [],
            "tools": {},
            "tts": {},
            "features": [],
            "extensions": [],
        }

    return await extension_manager.get_capabilities()
