"""Chat History API Routes

FastAPI routes for chat history management.
Compatible with SillyTavern chat format.
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .storage import ChatMessage, ChatMetadata, get_chat_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chats", tags=["chats"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessageModel(BaseModel):
    """Request model for a chat message."""
    name: str
    is_user: bool
    is_name: bool = False
    send_date: Optional[int] = None
    mes: str = ""
    title: str = ""
    extra: Dict[str, Any] = {}


class ChatMetadataModel(BaseModel):
    """Request model for chat metadata."""
    chat_name: str = ""
    character_name: str = ""
    create_date: Optional[int] = None
    last_update: Optional[int] = None
    chat_id: Optional[str] = None
    is_group_chat: bool = False
    members: List[str] = []


class ChatSaveModel(BaseModel):
    """Request model for saving a chat."""
    messages: List[ChatMessageModel]
    metadata: Optional[ChatMetadataModel] = None


# =============================================================================
# Chat Endpoints
# =============================================================================

@router.get("/", response_model=List[Dict[str, Any]])
async def list_chats():
    """List all available chats."""
    storage = get_chat_storage()
    return storage.list_chats()


@router.get("/{chat_name}")
async def get_chat(chat_name: str):
    """Get a specific chat with all messages."""
    storage = get_chat_storage()
    result = storage.load_chat(chat_name)

    if not result:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_name}' not found")

    messages, metadata = result
    return {
        "messages": [m.to_dict() for m in messages],
        "metadata": metadata.to_dict()
    }


@router.post("/")
async def save_chat(data: ChatSaveModel):
    """Save a chat."""
    storage = get_chat_storage()

    messages = [ChatMessage(
        name=m.name,
        is_user=m.is_user,
        is_name=m.is_name,
        send_date=m.send_date,
        mes=m.mes,
        title=m.title,
        extra=m.extra,
    ) for m in data.messages]

    metadata = None
    if data.metadata:
        metadata = ChatMetadata(
            chat_name=data.metadata.chat_name,
            character_name=data.metadata.character_name,
            create_date=data.metadata.create_date,
            last_update=data.metadata.last_update,
            chat_id=data.metadata.chat_id,
            is_group_chat=data.metadata.is_group_chat,
            members=data.metadata.members,
        )

    filepath = storage.save_chat(messages, metadata)

    return {
        "success": True,
        "filepath": str(filepath),
        "chat_name": metadata.chat_name if metadata else messages[0].name if messages else "chat"
    }


@router.delete("/{chat_name}")
async def delete_chat(chat_name: str):
    """Delete a chat."""
    storage = get_chat_storage()
    success = storage.delete_chat(chat_name)

    if not success:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_name}' not found")

    return {"success": True, "message": f"Chat '{chat_name}' deleted"}


# =============================================================================
# Import/Export Endpoints
# =============================================================================

@router.post("/import")
async def import_chat(
    file: UploadFile = File(...),
    format: str = Query("auto", description="File format (auto, jsonl, json)")
):
    """Import a chat from a file.

    Supports JSONL and JSON formats compatible with SillyTavern.
    """
    storage = get_chat_storage()

    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = storage.import_chat(tmp_path, format)

        if result:
            messages, metadata = result
            return {
                "success": True,
                "chat_name": metadata.chat_name,
                "message_count": len(messages),
                "metadata": metadata.to_dict()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to import chat")
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/{chat_name}/export")
async def export_chat(
    chat_name: str,
    format: str = Query("jsonl", description="Export format: jsonl or json"),
    include_metadata: bool = Query(True, description="Include metadata file")
):
    """Export a chat to file.

    Returns the exported file for download.
    """
    storage = get_chat_storage()

    # Export to temp directory
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / f"{chat_name}_export.{format}"

    result = storage.export_chat(chat_name, output_path, format, include_metadata)

    if not result:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_name}' not found")

    media_type = "application/x-jsonlines" if format == "jsonl" else "application/json"

    return FileResponse(
        result,
        media_type=media_type,
        filename=f"{chat_name}.{format}"
    )


@router.get("/stats")
async def get_storage_stats():
    """Get storage statistics."""
    storage = get_chat_storage()
    return storage.get_storage_stats()
