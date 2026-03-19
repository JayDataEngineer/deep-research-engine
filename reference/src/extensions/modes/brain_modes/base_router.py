"""Base Router Extension - Simplified LiteLLM Tool Calling.

Both tool_call and deep_agent modes use direct LiteLLM tool calling.
No complex router pattern needed - LiteLLM handles tool selection natively.
"""

import logging
from typing import Optional, List, AsyncGenerator, Dict, Any
from langchain_core.messages import AIMessage
from ...core.base import TextModeExtension
from src.constants import DEFAULT_LLM_MODEL
from src.shared.sse_events import SSEvent

logger = logging.getLogger(__name__)


class BaseRouterExtension(TextModeExtension):
    """Base extension for direct LiteLLM tool calling.

    Uses LiteLLM's native tool calling instead of complex router pattern.
    LiteLLM automatically selects and calls tools based on their descriptions.
    """

    async def shutdown(self) -> None:
        """Shutdown the extension."""
        if hasattr(self, '_graph') and self._graph is not None:
            self._graph = None

    def get_default_tools(self) -> List[str]:
        """Return the default tools for this mode.

        Subclasses override this method to specify which tools are available.
        """
        raise NotImplementedError("Subclasses must implement get_default_tools()")

    async def build_graph(self):
        """No graph needed - using direct LiteLLM calls."""
        # No LangGraph graph needed
        # Tool calling is handled directly by LiteLLM
        return None

    def build_state(self, request):
        """Build state with model configuration."""
        state = super().build_state(request)

        # Get enabled tools from config or use defaults
        enabled_tools = self.config.get("enabled_tools", None)
        if enabled_tools is None:
            enabled_tools = self.get_default_tools()

        # Add model and tools for LiteLLM
        model = getattr(request, "model", DEFAULT_LLM_MODEL)
        if model == "select":
            from src.config import settings
            model = settings.llm_model or DEFAULT_LLM_MODEL
            
        state["model"] = model
        state["enabled_tools"] = enabled_tools
        state["temperature"] = getattr(request, "temperature", 0.7)

        return state

    async def stream_chat(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat response using direct LiteLLM tool calling.

        LiteLLM handles tool selection and execution automatically.
        """
        from src.shared import get_llm
        from src.shared.tools.registry import load_tools
        
        if config is None:
            config = {"configurable": {"thread_id": state.get("thread_id")}}
        
        thread_id = state.get("thread_id")
        messages = state.get("messages", [])
        enabled_tools = state.get("enabled_tools", [])
        temperature = state.get("temperature", 0.7)
        
        # Load tools
        tools = load_tools(enabled_tools) if enabled_tools else []
        
        # Get LLM with tool calling
        llm = await get_llm(model=state["model"], temperature=temperature)
        
        # Stream with tools
        async for chunk in llm.astream(messages, tools=tools):
            yield chunk
