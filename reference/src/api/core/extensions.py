"""
Extension API Routes

Provides endpoints for the extension system:
- GET /api/extensions/capabilities - Query available extensions
- POST /api/extensions/{name}/activate - Activate a service extension
- POST /api/extensions/{name}/deactivate - Deactivate a service extension
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from src.extensions.core.manager import ExtensionManager

router = APIRouter(prefix="/api/extensions", tags=["extensions"])

# Global manager instance - will be set by main.py
extension_manager: Optional[ExtensionManager] = None


def set_extension_manager(manager: ExtensionManager, lifecycle=None):
    """
    Set the global extension manager instance.

    Note: lifecycle parameter is kept for backward compatibility but is ignored.
    The manager uses its own built-in lifecycle.
    """
    global extension_manager
    extension_manager = manager


def get_extension_manager() -> Optional[ExtensionManager]:
    """
    Get the global extension manager instance.

    Returns:
        The ExtensionManager instance, or None if not initialized.
    """
    return extension_manager


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """
    Get extension capabilities for the frontend.

    Returns which extensions are available and what they provide.
    """
    if extension_manager is None:
        return {
            "modes": [],
            "tools": {},
            "tts": {},
            "features": [],
            "extensions": [],
        }

    return await extension_manager.get_capabilities()


@router.post("/{name}/activate")
async def activate_extension(name: str) -> Dict[str, Any]:
    """
    Activate a service extension.

    For heavy extensions like TTS, this starts the service
    and manages resource usage via LRU eviction.
    """
    if extension_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extension manager not initialized"
        )

    # Use the manager's built-in activate_service method
    ext = await extension_manager.activate_service(name)

    if ext is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extension '{name}' not found or is not a service extension"
        )

    return {
        "name": name,
        "status": "active",
        "message": f"Extension '{name}' activated"
    }


@router.post("/{name}/deactivate")
async def deactivate_extension(name: str) -> Dict[str, Any]:
    """
    Deactivate a service extension to free resources.
    """
    if extension_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extension manager not initialized"
        )

    await extension_manager.deactivate_service(name)

    return {
        "name": name,
        "status": "inactive",
        "message": f"Extension '{name}' deactivated"
    }


@router.get("/{name}/status")
async def get_extension_status(name: str) -> Dict[str, Any]:
    """
    Get the status of a specific extension.
    """
    if extension_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extension manager not initialized"
        )

    ext = extension_manager.get(name)

    if ext is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extension '{name}' not found"
        )

    from src.extensions.core.base import ServiceExtension

    is_service_active = False
    if isinstance(ext, ServiceExtension):
        is_service_active = extension_manager.lifecycle.is_active(name)

    return {
        "name": name,
        "status": ext.status.value,
        "category": ext.category,
        "is_active": extension_manager.is_active(name),
        "is_service_active": is_service_active,
        "display_name": ext.config.get('extension', {}).get('display_name', name),
    }
