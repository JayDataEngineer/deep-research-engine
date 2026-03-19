"""
Universal Chat Extension

Provides the universal chat endpoint that routes requests to the appropriate app mode.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.extensions.core.base import BaseExtension, ExtensionStatus
from src.shared.sse_events import SSEvent
from src.shared.models.api_schema import ChatRequest
from src.config import settings

logger = logging.getLogger(__name__)


def register_extension():
    """Register the universal chat extension.

    This factory function creates an instance of the UniversalChatExtension.
    The loader calls this without arguments and expects an instance.
    """
    from pathlib import Path
    from shared.config import load_yaml_config

    ext_dir = Path(__file__).parent
    config = load_yaml_config(ext_dir / "config.yaml")
    return UniversalChatExtension(config, ext_dir)


class UniversalChatExtension(BaseExtension):
    """
    Universal chat extension.

    Provides the universal chat endpoint that delegates to mode extensions.
    All business logic is handled by mode extensions.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router = None

    def _create_router(self) -> APIRouter:
        """Create the FastAPI router for universal chat."""
        router = APIRouter(prefix="/api/v1", tags=["chat"])

        @router.post("/chat")
        async def chat_endpoint(request: Request, body: ChatRequest):
            """Universal chat endpoint - delegates to extension system."""
            logger.info(f"[CHAT] Received request. session_id={body.session_id}, mode={body.brainMode or body.chatMode}")

            extension_manager = request.app.state.extension_manager
            if not extension_manager:
                raise HTTPException(status_code=500, detail="Extension manager not initialized")

            # Determine mode
            brain_mode = body.brainMode or None
            brain_mode_mapping = {
                "direct": "direct",
                "tool_call": "tool_call",
                "deep_agent": "deep_agent",
                "research": "deep_research",
            }
            chat_mode_mapping = {
                "generic": "tool_call",
                "enhanced": "tool_call",
                "smart_tools": "tool_call",
                "deep_research": "deep_research",
                "deep-research": "deep_research",
            }

            if brain_mode and brain_mode.lower() in brain_mode_mapping:
                backend_mode = brain_mode_mapping[brain_mode.lower()]
            else:
                chat_mode = body.chatMode or "direct"
                backend_mode = chat_mode_mapping.get(chat_mode.lower(), "direct")

            extension = extension_manager.get_extension(f"chat_{backend_mode}")
            if not extension:
                extension = extension_manager.get_extension(backend_mode)

            if not extension:
                raise HTTPException(status_code=400, detail=f"Unknown chat mode: {backend_mode}")

            thread_id = body.session_id or f"{backend_mode}-{time.time()}"

            # Build state
            if backend_mode == "deep_research":
                from langchain_core.messages import HumanMessage
                user_input = ""
                if body.messages:
                    msg = body.messages[0]
                    if hasattr(msg, 'content'):
                        user_input = str(msg.content)
                    elif isinstance(msg, dict):
                        content = msg.get("content", "")
                        user_input = str(content) if isinstance(content, str) else ""

                initial_state = {
                    "raw_query": user_input,
                    "query": user_input,
                    "messages": [HumanMessage(content=user_input)],
                    "user_input": user_input,
                    "thread_id": thread_id,
                }
            else:
                initial_state = extension.build_state(body)

            # Stream response
            async def stream_generator():
                config = {"configurable": {"thread_id": thread_id}}

                if backend_mode == "deep_research":
                    from src.extensions.content.deep_research import load_research_config
                    use_case = body.use_case or "rag"
                    variant = body.variant or "default"
                    yaml_config = load_research_config(use_case=use_case, variant=variant)
                    yaml_config["_use_case"] = use_case
                    yaml_config["_variant"] = variant
                    config["configurable"].update(yaml_config)

                if backend_mode == "deep_research":
                    graph = extension.get_graph() if hasattr(extension, 'get_graph') else extension.graph
                    stream_source = graph.astream(initial_state, config, stream_mode=["messages", "custom", "updates"])
                else:
                    stream_source = extension.stream_chat(initial_state, config)

                async for chunk in stream_source:
                    if chunk is None:
                        continue

                    if await request.is_disconnected():
                        yield SSEvent.custom("finish", finishReason="interrupted")
                        return

                    if isinstance(chunk, tuple):
                        stream_mode, data = chunk
                    else:
                        stream_mode, data = "messages", chunk

                    if stream_mode == "custom" and isinstance(data, dict):
                        # Custom events (thinking, tool calls, etc.)
                        event_type = data.get("type", "custom")
                        if event_type == "thinking" or event_type == "status":
                            yield SSEvent.thinking(data.get("message", data.get("content", "")))
                        elif event_type == "tool_start":
                            yield SSEvent.tool_call(
                                name=data.get("toolName", "tool"),
                                args=data.get("input", {}),
                                tool_call_id=data.get("toolCallId")
                            )
                        elif event_type == "tool_end":
                            yield SSEvent.tool_result(
                                name=data.get("toolName", "tool"),
                                result=data.get("output", ""),
                                tool_call_id=data.get("toolCallId")
                            )
                        else:
                            yield SSEvent.custom(event_type, **data)
                            
                    elif stream_mode == "messages":
                        if isinstance(data, tuple) and len(data) == 2:
                            msg, metadata = data
                            from langchain_core.messages import AIMessage

                            if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                                content = msg.content

                                if backend_mode == "deep_research":
                                    source_node = None
                                    if isinstance(metadata, dict):
                                        lg_metadata = metadata.get("langgraph__metadata", {})
                                        if isinstance(lg_metadata, dict):
                                            source_node = lg_metadata.get("source")
                                        if not source_node:
                                            source_node = metadata.get("source", "")
                                    if source_node != "synthesis":
                                        continue

                                if isinstance(content, str) and content.strip():
                                    yield SSEvent.text(content)

            response = StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = "keep-alive"
            response.headers["X-Accel-Buffering"] = "no"
            return response

        @router.get("/health")
        async def health_check():
            return {"status": "healthy", "version": "0.2.0", "model": settings.llm_model}

        @router.get("/config")
        async def get_config():
            return {
                "model": settings.llm_model,
                "defaultAudioVoice": settings.default_audio_voice,
                "defaultModalities": settings.default_modalities.split(","),
                "supportedFormats": {"image": [], "audio": [], "video": []},
                "maxUploadSize": settings.max_upload_size,
            }

        return router

    def register_routes(self, app) -> None:
        """Register universal chat API routes."""
        if self._router is None:
            self._router = self._create_router()
        app.include_router(self._router)

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Get router metadata for frontend discovery."""
        return {
            "prefix": "/api/v1",
            "name": "chat",
            "tags": ["chat", "universal", "streaming"],
            "description": "Universal chat endpoint that delegates to mode extensions",
        }

    async def initialize(self, context) -> None:
        """Initialize the universal chat extension."""
        self._status = ExtensionStatus.LOADED

    async def shutdown(self) -> None:
        """Shutdown the universal chat extension."""
        self._status = ExtensionStatus.UNLOADED
