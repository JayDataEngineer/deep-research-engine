"""
Deep Research Mode Extension

Multi-agent research with configurable use cases.
"""

from pathlib import Path
import logging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from .logging_utils import get_logger

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from src.extensions.core.base import LogicExtension

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


class DeepResearchExtension(LogicExtension):
    """Deep research mode extension."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._graph = None

    def register_routes(self, app) -> None:
        """Register API routes with the FastAPI application."""
        from .routes import deep_research_router, research_router

        # Both routers have their own prefixes defined
        app.include_router(deep_research_router)
        app.include_router(research_router)

    def get_router_info(self) -> dict:
        """Get router information for frontend discovery."""
        return {
            "prefix": "/api/v1/research",
            "name": "deep_research",
            "tags": ["research", "ai-agents"],
            "description": "Multi-agent deep research endpoints"
        }

    async def initialize(self, context):
        """Initialize the deep research mode."""
        await super().initialize(context)

        # Import graph builder (local to extension)
        from .builder import build_research_graph

        # Build the graph (with default use_case)
        self._graph = await build_research_graph(use_case="rag", variant="default")
        logger.info("Deep research mode initialized with graph")

    async def shutdown(self):
        """Shutdown the extension."""
        self._graph = None

    async def process(self, input_data: dict, **kwargs):
        """
        Process a research request.

        Args:
            input_data: Should contain 'args' (list with query) or 'messages'
            **kwargs: Additional parameters like thread_id, use_case

        Returns:
            Streaming response or result
        """
        # Extract query from args or messages
        args = input_data.get('args', [])
        messages = input_data.get('messages', [])

        # If args provided and first arg is a string, use it as the query
        if args and isinstance(args[0], str) and not messages:
            query = args[0]
            messages = [{"role": "user", "content": query}]
        elif messages:
            # Extract query from first message content
            query = messages[0].get("content", str(messages[0])) if isinstance(messages[0], dict) else str(messages[0])
        else:
            query = ""

        thread_id = kwargs.get('thread_id') or str(uuid.uuid4())
        use_case = kwargs.get('use_case', 'rag')

        # Rebuild graph if use_case changed (local import)
        from .builder import build_research_graph

        # Run the graph - always provide thread_id for checkpointer
        # Include all required state keys with defaults
        config = {"configurable": {"thread_id": thread_id}}
        graph_input = {
            # User input
            "query": query,
            "messages": messages,
            # Historical context
            "chat_history": [],
            "user_profile": {},
            # Phase I artifacts
            "research_brief": None,
            # Phase II artifacts
            "routing_decision": "",
            "workers_to_spawn": [],
            "worker_tasks": {},
            "worker_dependencies": None,
            "worker_types": None,
            "iteration_count": 1,
            # Phase III artifacts
            "worker_outputs": [],
            # Phase IV artifacts
            "critique": None,
            "critique_history": [],
            # Final output
            "final_response": None,
            "final_sources": [],
            "all_citations": [],
            # Metadata
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "current_phase": "worker",
            # Refinement
            "is_refinement": False,
            "refine_feedback": "",
            "refine_custom_flow": None,
            "previous_findings": "",
        }

        # Stream the response
        async for chunk in self._graph.astream(graph_input, config):
            yield chunk

    def get_graph(self):
        """Get the LangGraph for this mode."""
        return self._graph

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(DeepResearchExtension)
