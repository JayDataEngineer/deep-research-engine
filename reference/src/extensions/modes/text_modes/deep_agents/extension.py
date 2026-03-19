"""
Deep Agents Mode Extension

Deep agents all the way down - infinite conversation with:
- Router for query analysis and search strategy
- RAG integration with vector search
- Tool-enabled research with web + RAG
- Modern LangGraph V4 patterns
"""

from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage

from src.extensions.core.base import TextModeExtension
from src.shared.logging_utils import get_extension_logger

logger = get_extension_logger("deep_agents")


class DeepAgentsExtension(TextModeExtension):
    """
    Deep Agents mode extension with RAG + Web search.

    Features:
    - Router system for intelligent query routing
    - RAG integration for knowledge base search
    - Web search for current information
    - Modern LangGraph V4 patterns (Command API, Send API, Runtime Config)
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router = None

    async def build_graph(self):
        """Build and return the deep agents graph with V4 patterns."""
        from .graph import build_deep_agents_graph
        from src.shared.checkpointer import get_shared_checkpointer

        # Initialize RAG if enabled
        from src.shared.tools.ulli_langchain.memory.rag import initialize_rag
        try:
            initialize_rag(self.config)
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")

        # Get shared checkpointer for memory persistence
        checkpointer = await get_shared_checkpointer()

        return build_deep_agents_graph(checkpointer=checkpointer)

    async def shutdown(self):
        """Shutdown the extension."""
        self._graph = None

    def build_state(self, request) -> Dict[str, Any]:
        """
        Build initial state from chat request.

        Converts the standard chat request to the GlobalState format
        expected by the new LangGraph V4 deep agents graph.

        Args:
            request: The chat request from the API

        Returns:
            Initial state dictionary compatible with GlobalState schema
        """
        import time

        # Get base state from parent (has messages, user_input, thread_id, etc.)
        base_state = super().build_state(request)

        # Extract messages and convert to LangChain format if needed
        messages = base_state.get("messages", [])
        user_input = base_state.get("user_input", "")
        thread_id = base_state.get("thread_id", f"deep_agents_thread_{uuid.uuid4()}")

        # Build chat_history from messages (excluding the last user message which is the current query)
        chat_history = []
        for msg in messages[:-1]:  # Exclude last message (current query)
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))
            elif hasattr(msg, 'content'):
                # Already a LangChain message
                chat_history.append(msg)

        # Build state compatible with GlobalState schema
        from .graph import GlobalState, RouterDecision

        state: GlobalState = {
            # ===== User Input =====
            "query": user_input,
            "chat_history": chat_history,
            "user_profile": {},

            # ===== Router Artifacts =====
            "routing_decision": {
                "decision": "rag",
                "reasoning": "",
                "num_workers": 0,
                "worker_tasks": []
            },
            "boss_guidance": None,

            # ===== Worker Artifacts =====
            "worker_tasks": [],
            "worker_outputs": [],

            # ===== Synthesis Artifacts =====
            "sources": [],
            "citations": [],
            "response": None,

            # ===== Metadata =====
            "thread_id": thread_id,
            "research_id": f"research_{uuid.uuid4()}",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "current_phase": "router",
            "iteration_count": 1,

            # ===== Refinement =====
            "requires_revision": False,
            "revision_instructions": None,
        }

        return state

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        if self._router is None:
            from .routes import router
            self._router = router
        return self._router


# Register extension factory
from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(DeepAgentsExtension)
