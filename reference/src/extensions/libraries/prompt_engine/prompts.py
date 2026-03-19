"""
SillyTavern-compatible Prompt Manager API Endpoints

This module provides API endpoints for:
- Exporting prompts in SillyTavern format
- Importing prompts from SillyTavern
- Managing prompt order per character
- Building injection contexts
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse

from .prompt_models import (
    Prompt,
    PromptOrderEntry,
    PromptExport,
    PromptExportData,
    CharacterPromptConfig,
    PromptRole,
    InjectionPosition,
)
from .prompt_manager import get_prompt_manager, create_prompt_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


# =============================================================================
# PROMPT LISTING
# =============================================================================

@router.get("")
async def list_prompts(
    include_system: bool = Query(True, description="Include system prompts"),
    include_markers: bool = Query(True, description="Include marker prompts"),
):
    """List all available prompts."""
    try:
        manager = get_prompt_manager()
        prompts = manager.list_prompts(
            include_system=include_system,
            include_markers=include_markers,
        )

        return {
            "prompts": [p.model_dump() for p in prompts],
            "total_count": len(prompts),
        }
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {str(e)}")


@router.get("/{identifier}")
async def get_prompt(identifier: str):
    """Get a specific prompt by identifier."""
    try:
        manager = get_prompt_manager()
        prompt = manager.get_prompt(identifier)

        if not prompt:
            raise HTTPException(status_code=404, detail=f"Prompt '{identifier}' not found")

        return prompt.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prompt: {str(e)}")


# =============================================================================
# PROMPT ORDER MANAGEMENT
# =============================================================================

@router.get("/order/{character_id}")
async def get_prompt_order(character_id: str):
    """Get prompt order for a character."""
    try:
        manager = get_prompt_manager()
        order = manager.get_prompt_order(character_id)

        # Enrich with prompt details
        enriched_order = []
        for entry in order:
            prompt = manager.get_prompt(entry.identifier)
            if prompt:
                enriched_order.append({
                    "identifier": entry.identifier,
                    "enabled": entry.enabled,
                    "name": prompt.name,
                    "role": prompt.role,
                    "marker": prompt.marker,
                    "system_prompt": prompt.system_prompt,
                })

        return {
            "character_id": character_id,
            "prompt_order": enriched_order,
        }
    except Exception as e:
        logger.error(f"Error getting prompt order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prompt order: {str(e)}")


@router.post("/order/{character_id}")
async def set_prompt_order(character_id: str, order_data: Dict[str, Any]):
    """Set prompt order for a character."""
    try:
        # Convert to PromptOrderEntry objects
        order_entries = [
            PromptOrderEntry(
                identifier=e.get("identifier"),
                enabled=e.get("enabled", True),
            )
            for e in order_data.get("prompt_order", [])
        ]

        manager = get_prompt_manager()
        success = manager.set_prompt_order(character_id, order_entries)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to set prompt order - unknown prompt identifier")

        return {
            "success": True,
            "character_id": character_id,
            "entry_count": len(order_entries),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting prompt order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set prompt order: {str(e)}")


@router.patch("/order/{character_id}/{identifier}")
async def update_prompt_order_entry(
    character_id: str,
    identifier: str,
    enabled: Optional[bool] = None,
):
    """Update a single entry in the prompt order."""
    try:
        manager = get_prompt_manager()
        success = manager.update_prompt_order_entry(
            character_id,
            identifier,
            enabled=enabled,
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"Prompt '{identifier}' not found")

        return {
            "success": True,
            "character_id": character_id,
            "identifier": identifier,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt order entry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update entry: {str(e)}")


@router.post("/order/{character_id}/append/{identifier}")
async def append_prompt_to_order(
    character_id: str,
    identifier: str,
    enabled: bool = Query(True, description="Enable the appended prompt"),
):
    """Append a prompt to the character's order."""
    try:
        manager = get_prompt_manager()
        success = manager.append_prompt_to_order(character_id, identifier, enabled)

        if not success:
            raise HTTPException(status_code=404, detail=f"Prompt '{identifier}' not found")

        return {
            "success": True,
            "character_id": character_id,
            "identifier": identifier,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error appending prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to append prompt: {str(e)}")


@router.delete("/order/{character_id}/{identifier}")
async def detach_prompt_from_order(character_id: str, identifier: str):
    """Remove a prompt from the character's order."""
    try:
        manager = get_prompt_manager()
        success = manager.detach_prompt_from_order(character_id, identifier)

        if not success:
            raise HTTPException(status_code=404, detail=f"Prompt '{identifier}' not in order")

        return {
            "success": True,
            "character_id": character_id,
            "identifier": identifier,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detaching prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detach prompt: {str(e)}")


@router.post("/order/{character_id}/reset")
async def reset_prompt_order(character_id: str):
    """Reset prompt order to defaults."""
    try:
        manager = get_prompt_manager()
        success = manager.reset_prompt_order(character_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset prompt order")

        return {
            "success": True,
            "character_id": character_id,
            "message": "Prompt order reset to defaults",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting prompt order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")


# =============================================================================
# PROMPT COLLECTION / INJECTION CONTEXT
# =============================================================================

@router.get("/collection/{character_id}")
async def get_prompt_collection(
    character_id: str,
    generation_type: str = Query("normal", description="Generation type: normal, continue, quiet"),
    only_enabled: bool = Query(True, description="Only include enabled prompts"),
):
    """Get prompt collection for a character (filtered, ordered)."""
    try:
        manager = get_prompt_manager()
        collection = manager.get_prompt_collection(
            character_id,
            generation_type=generation_type,
            only_enabled=only_enabled,
        )

        return {
            "character_id": character_id,
            "generation_type": generation_type,
            "prompts": [p.model_dump() for p in collection.prompts],
            "count": len(collection.prompts),
            "overridden": collection.overridden,
        }
    except Exception as e:
        logger.error(f"Error getting prompt collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection: {str(e)}")


@router.post("/injection/{character_id}")
async def build_injection_context(
    character_id: str,
    request_data: Dict[str, Any],
):
    """
    Build injection context for LLM.

    Resolves markers, substitutes variables, and orders by depth.
    Returns prompts ready for LLM injection.
    """
    try:
        manager = get_prompt_manager()

        character_data = request_data.get("character_data")
        persona_data = request_data.get("persona_data")
        generation_type = request_data.get("generation_type", "normal")

        injection_context = manager.build_injection_context(
            character_id=character_id,
            character_data=character_data,
            persona_data=persona_data,
            generation_type=generation_type,
        )

        return {
            "character_id": character_id,
            "generation_type": generation_type,
            "injection_context": injection_context,
            "count": len(injection_context),
        }
    except Exception as e:
        logger.error(f"Error building injection context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build context: {str(e)}")


# =============================================================================
# EXPORT/IMPORT (SillyTavern-compatible)
# =============================================================================

@router.get("/export/full")
async def export_full_prompts():
    """Export all prompts in SillyTavern format."""
    try:
        manager = get_prompt_manager()
        export = manager.export_full()

        return JSONResponse(
            content=export.model_dump(),
            headers={
                "Content-Disposition": "attachment; filename=prompts-full.json",
            },
        )
    except Exception as e:
        logger.error(f"Error exporting prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export: {str(e)}")


@router.get("/export/character/{character_id}")
async def export_character_prompts(character_id: str):
    """Export prompts for a specific character in SillyTavern format."""
    try:
        manager = get_prompt_manager()

        # Use character_id as name if no separate name provided
        character_name = character_id

        export = manager.export_character(character_id, character_name)

        filename = f"{character_name}-prompts.json"

        return JSONResponse(
            content=export.model_dump(),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )
    except Exception as e:
        logger.error(f"Error exporting character prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export: {str(e)}")


@router.post("/import")
async def import_prompts(export_data: Dict[str, Any]):
    """
    Import prompts from SillyTavern export format.

    Accepts JSON matching ST export format:
    {
        "version": 1,
        "type": "full" | "character",
        "data": {
            "prompts": [...],
            "prompt_order": [...]
        }
    }
    """
    try:
        manager = get_prompt_manager()
        success = manager.import_export(export_data)

        if not success:
            raise HTTPException(status_code=400, detail="Invalid export format")

        return {
            "success": True,
            "imported_type": export_data.get("type", "unknown"),
            "prompt_count": len(export_data.get("data", {}).get("prompts", [])),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import: {str(e)}")


# =============================================================================
# PROMPT CRUD
# =============================================================================

@router.post("")
async def create_prompt(prompt_data: Dict[str, Any]):
    """Create a new prompt."""
    try:
        prompt = Prompt(**prompt_data)

        manager = get_prompt_manager()
        success = manager.add_prompt(prompt)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to add prompt")

        return {
            "success": True,
            "identifier": prompt.identifier,
            "prompt": prompt.model_dump(),
        }
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create: {str(e)}")


@router.put("/{identifier}")
async def update_prompt(identifier: str, prompt_data: Dict[str, Any]):
    """Update an existing prompt."""
    try:
        manager = get_prompt_manager()
        existing = manager.get_prompt(identifier)

        if not existing:
            raise HTTPException(status_code=404, detail=f"Prompt '{identifier}' not found")

        # Update fields
        update_data = existing.model_dump()
        update_data.update(prompt_data)

        updated_prompt = Prompt(**update_data)
        manager.add_prompt(updated_prompt)

        return {
            "success": True,
            "identifier": identifier,
            "prompt": updated_prompt.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update: {str(e)}")


@router.delete("/{identifier}")
async def delete_prompt(identifier: str):
    """Delete a prompt (only non-system prompts)."""
    try:
        manager = get_prompt_manager()
        success = manager.remove_prompt(identifier)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot delete system prompt or prompt not found")

        return {
            "success": True,
            "identifier": identifier,
            "message": f"Prompt '{identifier}' deleted",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


# =============================================================================
# PERSISTENCE
# =============================================================================

@router.post("/save")
async def save_prompts():
    """Save prompts to file."""
    try:
        manager = get_prompt_manager()
        success = manager.save_to_file()

        if not success:
            raise HTTPException(status_code=500, detail="No storage directory configured")

        return {
            "success": True,
            "message": "Prompts saved",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")


@router.post("/load")
async def load_prompts(path: Optional[str] = None):
    """Load prompts from file."""
    try:
        if not path:
            raise HTTPException(status_code=400, detail="Path is required")

        manager = get_prompt_manager()
        success = manager.load_from_file(Path(path))

        if not success:
            raise HTTPException(status_code=404, detail=f"File not found or invalid format")

        return {
            "success": True,
            "path": path,
            "message": "Prompts loaded",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load: {str(e)}")
