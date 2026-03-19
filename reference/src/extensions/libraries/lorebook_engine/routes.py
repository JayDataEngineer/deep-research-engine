"""Lorebooks API endpoints.

REST API for managing SillyTavern-compatible lorebooks with keyword-activated context.
"""

import logging
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Literal

from fastapi import APIRouter, UploadFile, File, Form, Query, Depends
from pydantic import BaseModel

from src.extensions.libraries.lorebook_engine import (
    Lorebook,
    LorebookEntry,
    LorebookRegistry,
    DEFAULT_LOREBOOKS,
    ActivationResult,
)
from src.extensions.libraries.lorebook_engine.trigger import LorebookTrigger
from src.shared.utils.api_helpers import (
    normalize_id,
    or_404,
    or_409,
    or_400,
    serialize_datetime,
    get_lorebooks_dir,
    ensure_dir,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/lorebooks", tags=["lorebooks"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateLorebookRequest(BaseModel):
    """Request to create a new lorebook."""
    name: str
    description: Optional[str] = None
    category: Optional[str] = "custom"
    scan_depth: Optional[int] = 10
    token_budget: Optional[int] = 2000


class UpdateLorebookRequest(BaseModel):
    """Request to update a lorebook."""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    scan_depth: Optional[int] = None
    token_budget: Optional[int] = None


class CreateEntryRequest(BaseModel):
    """Request to create a lorebook entry - SillyTavern compatible."""
    id: Optional[int] = None  # Numeric ID like SillyTavern
    name: Optional[str] = None
    comment: Optional[str] = None
    content: str
    keys: Optional[List[str]] = []  # SillyTavern field name
    secondary_keys: Optional[List[str]] = []  # SillyTavern field name
    enabled: Optional[bool] = True
    constant: Optional[bool] = False
    position: Optional[str] = "after_char"
    case_sensitive: Optional[bool] = False
    selective: Optional[bool] = False
    use_regex: Optional[bool] = False  # New SillyTavern field
    insertion_order: Optional[int] = 0
    extensions: Optional[dict] = None  # Contains depth, probability


class UpdateEntryRequest(BaseModel):
    """Request to update a lorebook entry - SillyTavern compatible."""
    content: Optional[str] = None
    keys: Optional[List[str]] = None  # SillyTavern field name
    secondary_keys: Optional[List[str]] = None  # SillyTavern field name
    enabled: Optional[bool] = None
    extensions: Optional[dict] = None  # Contains depth, probability


class ActivateRequest(BaseModel):
    """Request to activate lorebooks for a conversation."""
    messages: List[Dict]
    linked_lorebooks: Optional[List[str]] = None
    scan_depth: Optional[int] = None
    token_budget: Optional[int] = None


class LorebookResponse(BaseModel):
    """Response containing lorebook details."""
    id: str
    name: str
    description: Optional[str]
    category: str
    scan_depth: int
    token_budget: int
    entry_count: int
    created_at: Optional[str]
    updated_at: Optional[str]


class EntryResponse(BaseModel):
    """Response containing entry details - SillyTavern compatible."""
    id: int  # Numeric ID like SillyTavern
    name: Optional[str]
    comment: Optional[str]
    content: str
    keys: List[str]  # SillyTavern field name
    secondary_keys: List[str]  # SillyTavern field name
    enabled: bool
    constant: bool
    position: str
    selective: bool
    use_regex: bool  # New SillyTavern field
    insertion_order: int
    extensions: dict  # Contains depth, probability


# =============================================================================
# Global State
# =============================================================================

_lorebook_registry: Optional[LorebookRegistry] = None
_lorebook_trigger: Optional[LorebookTrigger] = None

# Use unified data directory
LOREBOOKS_DIR = get_lorebooks_dir()


def get_registry() -> LorebookRegistry:
    """Get the global lorebook registry."""
    global _lorebook_registry
    if _lorebook_registry is None:
        _lorebook_registry = LorebookRegistry()
        for lb in DEFAULT_LOREBOOKS.values():
            _lorebook_registry.register(lb)
    return _lorebook_registry


def get_trigger() -> LorebookTrigger:
    """Get the global lorebook trigger."""
    global _lorebook_trigger
    if _lorebook_trigger is None:
        _lorebook_trigger = LorebookTrigger()
    return _lorebook_trigger


# =============================================================================
# YAML Persistence
# =============================================================================

def save_lorebook_to_yaml(lorebook: Lorebook) -> Path:
    """Save a lorebook to YAML file.

    Args:
        lorebook: The lorebook to save

    Returns:
        Path to saved file
    """
    filename = f"{normalize_id(lorebook.name)}.yaml"
    file_path = LOREBOOKS_DIR / filename

    data = lorebook.model_dump(mode='json', exclude_none=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved lorebook to {file_path}")
    return file_path


def load_lorebook_from_yaml(file_path: Path) -> Optional[Lorebook]:
    """Load a lorebook from YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Lorebook or None if invalid
    """
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return Lorebook(**data)
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def delete_lorebook_yaml(lorebook_id: str) -> bool:
    """Delete a lorebook YAML file.

    Args:
        lorebook_id: ID of the lorebook

    Returns:
        True if deleted, False if not found
    """
    filename = f"{lorebook_id}.yaml"
    file_path = LOREBOOKS_DIR / filename

    if file_path.exists():
        file_path.unlink()
        logger.info(f"Deleted lorebook YAML: {file_path}")
        return True
    return False


def scan_lorebooks_directory() -> int:
    """Scan the lorebooks directory and load all YAML files.

    Returns:
        Number of lorebooks loaded
    """
    count = 0
    ensure_dir(LOREBOOKS_DIR)

    for yaml_file in LOREBOOKS_DIR.glob("*.yaml"):
        lorebook = load_lorebook_from_yaml(yaml_file)
        if lorebook:
            get_registry().register(lorebook)
            count += 1

    logger.info(f"Scanned {count} lorebooks from {LOREBOOKS_DIR}")
    return count


# =============================================================================
# Response Builders
# =============================================================================

def build_lorebook_response(lorebook: Lorebook, lorebook_id: Optional[str] = None) -> LorebookResponse:
    """Build a LorebookResponse from a Lorebook.

    Args:
        lorebook: The lorebook
        lorebook_id: Optional ID override

    Returns:
        LorebookResponse
    """
    return LorebookResponse(
        id=lorebook_id or normalize_id(lorebook.name),
        name=lorebook.name,
        description=lorebook.description,
        category=lorebook.category,
        scan_depth=lorebook.scan_depth,
        token_budget=lorebook.token_budget,
        entry_count=len(lorebook.entries),
        created_at=serialize_datetime(lorebook.created_at),
        updated_at=serialize_datetime(lorebook.updated_at),
    )


def build_entry_response(entry: LorebookEntry) -> EntryResponse:
    """Build an EntryResponse from a LorebookEntry.

    Args:
        entry: The lorebook entry

    Returns:
        EntryResponse
    """
    return EntryResponse(
        id=entry.id,
        name=entry.name,
        comment=entry.comment,
        content=entry.content,
        keys=entry.keys,  # SillyTavern field name
        secondary_keys=entry.secondary_keys,  # SillyTavern field name
        enabled=entry.enabled,
        constant=entry.constant,
        position=entry.position,
        selective=entry.selective,
        use_regex=entry.use_regex,  # New SillyTavern field
        insertion_order=entry.insertion_order,
        extensions=entry.extensions,  # Contains depth, probability
    )


# =============================================================================
# Startup
# =============================================================================

@router.on_event("startup")
async def startup():
    """Initialize lorebooks on startup."""
    logger.info("Initializing lorebooks API...")
    ensure_dir(LOREBOOKS_DIR)
    scan_lorebooks_directory()
    logger.info(f"Lorebooks directory: {LOREBOOKS_DIR}")


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=List[LorebookResponse])
async def list_lorebooks(
    category: Optional[str] = None,
    search: Optional[str] = None,
):
    """List all available lorebooks."""
    registry = get_registry()

    if category:
        lorebooks = [lb for lb in registry.list_all() if lb.category == category]
    elif search:
        lorebooks = registry.search(search)
    else:
        lorebooks = registry.list_all()

    return [build_lorebook_response(lb) for lb in lorebooks]


@router.get("/categories")
async def list_categories():
    """List all lorebook categories and their counts."""
    registry = get_registry()
    lorebooks = registry.list_all()

    categories: Dict[str, int] = {}
    for lorebook in lorebooks:
        categories[lorebook.category] = categories.get(lorebook.category, 0) + 1

    return {"categories": categories}


@router.get("/{lorebook_id}", response_model=Lorebook)
async def get_lorebook(lorebook_id: str):
    """Get a specific lorebook by ID."""
    registry = get_registry()
    lorebook = or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")
    return lorebook


@router.post("", response_model=LorebookResponse)
async def create_lorebook(request: CreateLorebookRequest):
    """Create a new lorebook."""
    registry = get_registry()

    lorebook_id = normalize_id(request.name)
    or_409(registry.get(lorebook_id), f"Lorebook '{request.name}' already exists")

    lorebook = Lorebook(
        name=request.name,
        description=request.description,
        category=request.category or "custom",
        scan_depth=request.scan_depth or 10,
        token_budget=request.token_budget or 2000,
        entries=[],
    )

    registry.register(lorebook)
    save_lorebook_to_yaml(lorebook)

    return build_lorebook_response(lorebook, lorebook_id)


@router.put("/{lorebook_id}", response_model=LorebookResponse)
async def update_lorebook(lorebook_id: str, request: UpdateLorebookRequest):
    """Update an existing lorebook."""
    registry = get_registry()
    lorebook = or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")

    if request.name is not None:
        lorebook.name = request.name
    if request.description is not None:
        lorebook.description = request.description
    if request.category is not None:
        lorebook.category = request.category
    if request.scan_depth is not None:
        lorebook.scan_depth = request.scan_depth
    if request.token_budget is not None:
        lorebook.token_budget = request.token_budget

    save_lorebook_to_yaml(lorebook)
    return build_lorebook_response(lorebook, lorebook_id)


@router.delete("/{lorebook_id}")
async def delete_lorebook(lorebook_id: str):
    """Delete a lorebook."""
    registry = get_registry()
    or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")

    del registry.lorebooks[lorebook_id]
    delete_lorebook_yaml(lorebook_id)

    return {"deleted": True, "id": lorebook_id}


@router.get("/{lorebook_id}/entries", response_model=List[EntryResponse])
async def list_entries(lorebook_id: str):
    """List all entries in a lorebook."""
    registry = get_registry()
    lorebook = or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")

    return [build_entry_response(entry) for entry in lorebook.entries]


@router.post("/{lorebook_id}/entries", response_model=EntryResponse)
async def create_entry(lorebook_id: str, request: CreateEntryRequest):
    """Add an entry to a lorebook - SillyTavern compatible."""
    registry = get_registry()
    lorebook = or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")

    # Auto-generate ID if not provided (find max existing ID)
    if request.id is None:
        max_id = max((e.id for e in lorebook.entries), default=-1)
        entry_id = max_id + 1
    else:
        entry_id = request.id

    # Merge extensions with defaults
    extensions = request.extensions or {}
    if "depth" not in extensions:
        extensions["depth"] = 4
    if "probability" not in extensions:
        extensions["probability"] = 100

    entry = LorebookEntry(
        id=entry_id,
        name=request.name,
        comment=request.comment,
        content=request.content,
        keys=request.keys or [],  # SillyTavern field name
        secondary_keys=request.secondary_keys or [],  # SillyTavern field name
        enabled=request.enabled if request.enabled is not None else True,
        constant=request.constant if request.constant is not None else False,
        position=request.position or "after_char",
        case_sensitive=request.case_sensitive if request.case_sensitive is not None else False,
        selective=request.selective if request.selective is not None else False,
        use_regex=request.use_regex if request.use_regex is not None else False,
        insertion_order=request.insertion_order or 0,
        extensions=extensions,
    )

    try:
        lorebook.add_entry(entry)
    except ValueError as e:
        or_400(True, str(e))

    save_lorebook_to_yaml(lorebook)
    return build_entry_response(entry)


@router.put("/{lorebook_id}/entries/{entry_id}")
async def update_entry(lorebook_id: str, entry_id: int, request: UpdateEntryRequest):
    """Update a specific entry in a lorebook - SillyTavern compatible."""
    registry = get_registry()
    lorebook = or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")

    updates = {}
    if request.content is not None:
        updates["content"] = request.content
    if request.keys is not None:
        updates["keys"] = request.keys  # SillyTavern field name
    if request.secondary_keys is not None:
        updates["secondary_keys"] = request.secondary_keys  # SillyTavern field name
    if request.enabled is not None:
        updates["enabled"] = request.enabled
    if request.extensions is not None:
        updates["extensions"] = request.extensions

    success = lorebook.update_entry(str(entry_id), updates)
    or_404(success, entry_id, "Entry")

    save_lorebook_to_yaml(lorebook)

    entry = lorebook.get_entry(str(entry_id))
    return build_entry_response(entry)


@router.delete("/{lorebook_id}/entries/{entry_id}")
async def delete_entry(lorebook_id: str, entry_id: int):
    """Delete an entry from a lorebook."""
    registry = get_registry()
    lorebook = or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")

    success = lorebook.remove_entry(str(entry_id))
    or_404(success, entry_id, "Entry")

    save_lorebook_to_yaml(lorebook)
    return {"deleted": True, "entry_id": entry_id}


@router.post("/activate")
async def activate_lorebooks(request: ActivateRequest):
    """Activate lorebook entries based on conversation context.

    This is the main endpoint that scans conversation messages
    and returns activated entries with their content.
    """
    trigger = get_trigger()
    registry = get_registry()

    # Get linked lorebooks or all lorebooks
    if request.linked_lorebooks:
        lorebooks = [registry.get(lb_id) for lb_id in request.linked_lorebooks]
        lorebooks = [lb for lb in lorebooks if lb is not None]
    else:
        lorebooks = registry.list_all()

    # Collect all entries from all lorebooks
    all_entries = []
    for lorebook in lorebooks:
        all_entries.extend(lorebook.get_enabled_entries())

    # Trigger activation
    result = trigger.trigger(
        entries=all_entries,
        conversation_messages=request.messages,
        scan_depth=request.scan_depth,
        token_budget=request.token_budget,
    )

    before_content, after_content = result.render_context()

    return {
        "activated": result.to_dict(),
        "context": {
            "before": before_content,
            "after": after_content,
        },
    }


@router.post("/{lorebook_id}/link")
async def link_lorebook(
    lorebook_id: str,
    target_type: Literal["character", "session"] = Query(...),
    target_id: str = Query(...),
):
    """Link a lorebook to a character or session.

    This creates an association so the lorebook is automatically
    activated when that character/session is used.
    """
    registry = get_registry()
    or_404(registry.get(lorebook_id), lorebook_id, "Lorebook")

    # Store link in lorebook extensions
    lorebook = registry.get(lorebook_id)
    link_key = f"link_{target_type}"
    links = lorebook.extensions.get(link_key, [])
    if target_id not in links:
        links.append(target_id)
        lorebook.extensions[link_key] = links
        save_lorebook_to_yaml(lorebook)

    return {
        "linked": True,
        "lorebook_id": lorebook_id,
        "target_type": target_type,
        "target_id": target_id,
    }
