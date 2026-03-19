"""Lorebooks API Routes

FastAPI routes for lorebook (World Info) management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lorebooks", tags=["lorebooks"])


# =============================================================================
# Request/Response Models
# =============================================================================

class LorebookEntryCreate(BaseModel):
    """Request model for creating a lorebook entry."""
    keywords: List[str] = []
    content: str
    enabled: bool = True
    order: int = 100
    position: int = 0


class LorebookEntryUpdate(BaseModel):
    """Request model for updating a lorebook entry."""
    keywords: Optional[List[str]] = None
    content: Optional[str] = None
    enabled: Optional[bool] = None
    order: Optional[int] = None
    position: Optional[int] = None


class LorebookCreate(BaseModel):
    """Request model for creating a lorebook."""
    name: str
    description: str = ""
    entries: List[LorebookEntryCreate] = []


class LorebookUpdate(BaseModel):
    """Request model for updating a lorebook."""
    description: Optional[str] = None
    entries: Optional[List[LorebookEntryCreate]] = None


# =============================================================================
# Storage helpers
# =============================================================================

def get_storage_dir() -> Path:
    """Get the lorebooks storage directory."""
    try:
        from src.shared.utils.api_helpers import get_inputs_root_dir
        return get_inputs_root_dir() / "lorebooks"
    except (ImportError, AttributeError):
        # Fallback: go up to project root (backend/ -> project root)
        return Path(__file__).parent.parent.parent.parent.parent.parent.parent / "data" / "inputs" / "lorebooks"


def get_lorebook_path(name: str) -> Path:
    """Get the path to a lorebook file."""
    return get_storage_dir() / f"{name}.json"


# =============================================================================
# Lorebook Endpoints
# =============================================================================

@router.get("/", response_model=List[str])
async def list_lorebooks():
    """List all available lorebooks."""
    storage_dir = get_storage_dir()
    if not storage_dir.exists():
        return []

    lorebooks = []
    for file_path in storage_dir.glob("*.json"):
        lorebooks.append(file_path.stem)
    return lorebooks


@router.get("/{name}")
async def get_lorebook(name: str):
    """Get a specific lorebook by name."""
    lorebook_path = get_lorebook_path(name)

    if not lorebook_path.exists():
        raise HTTPException(status_code=404, detail=f"Lorebook '{name}' not found")

    import json
    with open(lorebook_path) as f:
        data = json.load(f)

    return data


@router.post("/")
async def create_lorebook(lorebook: LorebookCreate):
    """Create a new lorebook."""
    lorebook_path = get_lorebook_path(lorebook.name)

    if lorebook_path.exists():
        raise HTTPException(status_code=400, detail=f"Lorebook '{lorebook.name}' already exists")

    storage_dir = get_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)

    import json
    data = {
        "name": lorebook.name,
        "description": lorebook.description,
        "entries": [entry.dict() for entry in lorebook.entries]
    }

    with open(lorebook_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"name": lorebook.name, "message": "Lorebook created successfully"}


@router.put("/{name}")
async def update_lorebook(name: str, lorebook: LorebookUpdate):
    """Update an existing lorebook."""
    lorebook_path = get_lorebook_path(name)

    if not lorebook_path.exists():
        raise HTTPException(status_code=404, detail=f"Lorebook '{name}' not found")

    import json
    with open(lorebook_path) as f:
        data = json.load(f)

    if lorebook.description is not None:
        data["description"] = lorebook.description

    if lorebook.entries is not None:
        data["entries"] = [entry.dict() for entry in lorebook.entries]

    with open(lorebook_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"name": name, "message": "Lorebook updated successfully"}


@router.delete("/{name}")
async def delete_lorebook(name: str):
    """Delete a lorebook."""
    lorebook_path = get_lorebook_path(name)

    if not lorebook_path.exists():
        raise HTTPException(status_code=404, detail=f"Lorebook '{name}' not found")

    lorebook_path.unlink()
    return {"name": name, "message": "Lorebook deleted successfully"}


@router.post("/{name}/entries")
async def add_entry(name: str, entry: LorebookEntryCreate):
    """Add an entry to a lorebook."""
    lorebook_path = get_lorebook_path(name)

    if not lorebook_path.exists():
        raise HTTPException(status_code=404, detail=f"Lorebook '{name}' not found")

    import json
    with open(lorebook_path) as f:
        data = json.load(f)

    if "entries" not in data:
        data["entries"] = []

    data["entries"].append(entry.dict())

    with open(lorebook_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"message": "Entry added successfully"}


@router.delete("/{name}/entries/{index}")
async def remove_entry(name: str, index: int):
    """Remove an entry from a lorebook."""
    lorebook_path = get_lorebook_path(name)

    if not lorebook_path.exists():
        raise HTTPException(status_code=404, detail=f"Lorebook '{name}' not found")

    import json
    with open(lorebook_path) as f:
        data = json.load(f)

    if "entries" not in data or index < 0 or index >= len(data["entries"]):
        raise HTTPException(status_code=400, detail="Invalid entry index")

    data["entries"].pop(index)

    with open(lorebook_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"message": "Entry removed successfully"}
