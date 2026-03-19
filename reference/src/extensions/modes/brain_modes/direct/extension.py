"""Direct Mode (○) - Pure LiteLLM Call, No Tools.

Simplest mode - just LiteLLM with no tool calling.
Direct LLM response to user input.
"""

import logging
from typing import Optional, Dict, Any, AsyncGenerator
from ....core.base import TextModeExtension
from ....core.factory import register_extension_factory
from src.shared.sse_events import SSEvent

logger = logging.getLogger(__name__)


class DirectExtension(TextModeExtension):
    """Direct mode - Pure LiteLLM call with no tools."""

    async def build_graph(self):
        """No graph needed - direct LiteLLM calls."""
        return None

    def build_state(self, request):
        """Build state for direct mode."""
        state = super().build_state(request)
        state["enable_tools"] = False
        return state

    async def stream_chat(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat response using direct LiteLLM call (no tools)."""
        from src.shared import get_llm
        
        if config is None:
            config = {"configurable": {"thread_id": state.get("thread_id")}}
        
        messages = state.get("messages", [])
        temperature = state.get("temperature", 0.7)
        
        # Get LLM without tools
        llm = await get_llm(model=state["model"], temperature=temperature)
        
        # Stream response
        async for chunk in llm.astream(messages):
            yield chunk


register_extension = register_extension_factory(DirectExtension)
