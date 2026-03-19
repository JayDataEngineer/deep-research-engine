"""Prompt Stacks API endpoints.

REST API for managing YAML-based prompt stacks with full CRUD operations.
Compatible with SillyTavern-style prompt management.
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from . import (
    PromptStack,
    StackLayer,
    StackVariable,
    get_stack_manager,
    init_stack_manager,
)
from src.shared.utils.api_helpers import (
    normalize_id,
    or_404,
    or_409,
    serialize_datetime,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts/stacks", tags=["prompt-stacks"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateStackRequest(BaseModel):
    """Request to create a new prompt stack."""
    name: str
    description: Optional[str] = None
    category: Optional[str] = "custom"
    tags: Optional[List[str]] = []
    layers: Optional[List[dict]] = []


class UpdateStackRequest(BaseModel):
    """Request to update an existing stack."""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    layers: Optional[List[dict]] = None


class AddLayerRequest(BaseModel):
    """Request to add a layer to a stack."""
    id: str
    content: str
    role: Optional[str] = "system"
    injection_depth: Optional[int] = 0
    injection_position: Optional[str] = "before"
    enabled: Optional[bool] = True
    priority: Optional[int] = 50
    name: Optional[str] = None
    comment: Optional[str] = None
    variables: Optional[dict] = None
    extensions: Optional[dict] = None


class UpdateLayerRequest(BaseModel):
    """Request to update a layer."""
    content: Optional[str] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    variables: Optional[dict] = None


class StackResponse(BaseModel):
    """Response containing stack details."""
    id: str
    name: str
    description: Optional[str]
    version: str
    author: Optional[str]
    category: str
    tags: List[str]
    layer_count: int
    created_at: Optional[str]
    updated_at: Optional[str]


class StackPreviewResponse(BaseModel):
    """Response containing rendered stack preview."""
    rendered: str
    layer_count: int
    variables: List[dict]


# =============================================================================
# Response Builders
# =============================================================================

def build_stack_response(stack: PromptStack, stack_id: Optional[str] = None) -> StackResponse:
    """Build a StackResponse from a PromptStack.

    Args:
        stack: The prompt stack
        stack_id: Optional ID override

    Returns:
        StackResponse
    """
    return StackResponse(
        id=stack_id or normalize_id(stack.name),
        name=stack.name,
        description=stack.description,
        version=stack.version,
        author=stack.author,
        category=stack.category,
        tags=stack.tags,
        layer_count=len(stack.layers),
        created_at=serialize_datetime(stack.created_at),
        updated_at=serialize_datetime(stack.updated_at),
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=List[StackResponse])
async def list_stacks(
    category: Optional[str] = None,
    search: Optional[str] = None,
):
    """List all available prompt stacks."""
    try:
        manager = get_stack_manager()
    except RuntimeError:
        # Manager not initialized, return empty list
        return []

    if category:
        stacks = manager.get_by_category(category)
    elif search:
        stacks = manager.search(search)
    else:
        stacks = manager.get_all()

    return [build_stack_response(stack) for stack in stacks]


@router.get("/categories")
async def list_categories():
    """List all stack categories and their counts."""
    manager = get_stack_manager()
    stacks = manager.get_all()

    categories: dict[str, int] = {}
    for stack in stacks:
        categories[stack.category] = categories.get(stack.category, 0) + 1

    return {"categories": categories}


@router.get("/{stack_id}", response_model=PromptStack)
async def get_stack(stack_id: str):
    """Get a specific stack by ID."""
    manager = get_stack_manager()
    stack = manager.get(stack_id)
    if not stack:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Stack '{stack_id}' not found")
    return stack


@router.post("", response_model=StackResponse)
async def create_stack(request: CreateStackRequest):
    """Create a new prompt stack."""
    manager = get_stack_manager()

    stack_id = normalize_id(request.name)
    or_409(manager.get(stack_id), f"Stack '{request.name}' already exists")

    stack = PromptStack(
        name=request.name,
        description=request.description,
        category=request.category or "custom",
        tags=request.tags or [],
        layers=[StackLayer(**layer) for layer in (request.layers or [])],
    )

    await manager.save_stack(stack)
    return build_stack_response(stack, stack_id)


@router.put("/{stack_id}", response_model=StackResponse)
async def update_stack(stack_id: str, request: UpdateStackRequest):
    """Update an existing stack."""
    manager = get_stack_manager()
    stack = or_404(manager.get(stack_id), stack_id, "Stack")

    if request.name is not None:
        stack.name = request.name
    if request.description is not None:
        stack.description = request.description
    if request.category is not None:
        stack.category = request.category
    if request.tags is not None:
        stack.tags = request.tags
    if request.layers is not None:
        stack.layers = [StackLayer(**layer) for layer in request.layers]

    await manager.save_stack(stack)
    return build_stack_response(stack, stack_id)


@router.delete("/{stack_id}")
async def delete_stack(stack_id: str):
    """Delete a stack."""
    manager = get_stack_manager()

    success = await manager.delete_stack(stack_id)
    or_404(success, stack_id, "Stack")

    return {"deleted": True, "id": stack_id}


@router.post("/{stack_id}/layers", response_model=StackResponse)
async def add_layer(stack_id: str, request: AddLayerRequest):
    """Add a layer to a stack."""
    manager = get_stack_manager()
    stack = or_404(manager.get(stack_id), stack_id, "Stack")

    layer = StackLayer(
        id=request.id,
        content=request.content,
        role=request.role or "system",
        injection_depth=request.injection_depth or 0,
        injection_position=request.injection_position or "before",
        enabled=request.enabled if request.enabled is not None else True,
        priority=request.priority or 50,
        name=request.name,
        comment=request.comment,
        variables=request.variables or {},
        extensions=request.extensions or {},
    )

    try:
        stack.add_layer(layer)
    except ValueError as e:
        or_404(True, str(e))

    await manager.save_stack(stack)
    return build_stack_response(stack, stack_id)


@router.put("/{stack_id}/layers/{layer_id}")
async def update_layer(stack_id: str, layer_id: str, request: UpdateLayerRequest):
    """Update a specific layer in a stack."""
    manager = get_stack_manager()
    stack = or_404(manager.get(stack_id), stack_id, "Stack")

    updates = {}
    if request.content is not None:
        updates["content"] = request.content
    if request.enabled is not None:
        updates["enabled"] = request.enabled
    if request.priority is not None:
        updates["priority"] = request.priority
    if request.variables is not None:
        updates["variables"] = request.variables

    success = stack.update_layer(layer_id, updates)
    or_404(success, layer_id, "Layer")

    await manager.save_stack(stack)
    return {"updated": True, "layer_id": layer_id}


@router.delete("/{stack_id}/layers/{layer_id}")
async def delete_layer(stack_id: str, layer_id: str):
    """Delete a layer from a stack."""
    manager = get_stack_manager()
    stack = or_404(manager.get(stack_id), stack_id, "Stack")

    success = stack.remove_layer(layer_id)
    or_404(success, layer_id, "Layer")

    await manager.save_stack(stack)
    return {"deleted": True, "layer_id": layer_id}


@router.post("/{stack_id}/preview", response_model=StackPreviewResponse)
async def preview_stack(
    stack_id: str,
    variables: Optional[dict] = None,
):
    """Preview a rendered stack with variable substitution."""
    manager = get_stack_manager()
    stack = or_404(manager.get(stack_id), stack_id, "Stack")

    rendered = stack.render(**(variables or {}))

    var_defs = [
        {"name": v.name, "default": v.default, "description": v.description}
        for v in stack.variables
    ]

    return StackPreviewResponse(
        rendered=rendered,
        layer_count=len(stack.get_enabled_layers()),
        variables=var_defs,
    )


@router.post("/sync")
async def sync_stacks():
    """Sync the registry with the filesystem."""
    manager = get_stack_manager()
    count = await manager.scan_directory()
    await manager.save_registry()

    return {"synced": True, "stacks_found": count}


@router.post("/import")
async def import_stack(file: UploadFile = File(...)):
    """Import a stack from a YAML file."""
    import yaml

    content = await file.read()
    try:
        data = yaml.safe_load(content)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    try:
        stack = PromptStack(**data)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid stack format: {e}")

    manager = get_stack_manager()
    await manager.save_stack(stack)

    return {
        "imported": True,
        "id": normalize_id(stack.name),
        "name": stack.name,
    }


@router.get("/{stack_id}/export")
async def export_stack(stack_id: str):
    """Export a stack as YAML file download."""
    import yaml
    from fastapi.responses import Response

    manager = get_stack_manager()
    stack = or_404(manager.get(stack_id), stack_id, "Stack")

    data = stack.model_dump(mode='json', exclude_none=True)
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False)

    return Response(
        content=yaml_content,
        media_type="text/yaml",
        headers={"Content-Disposition": f"attachment; filename={stack.name}.yaml"}
    )
