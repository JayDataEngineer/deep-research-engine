"""
Deep Think Extension

Multi-agent research spawner with configurable workflows.
"""

import logging
from pathlib import Path

from src.extensions.core.base import JobExtension

logger = logging.getLogger(__name__)


class DeepThinkExtension(JobExtension):
    """Deep Think research extension."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._spawner = None

    async def initialize(self, context):
        """Initialize the deep think extension."""
        await super().initialize(context)

        # Import the existing deep_research graph builder
        # This uses the existing implementation
        logger.info("Deep Think extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        self._spawner = None

    async def execute(self, prompt: str, **kwargs):
        """
        Execute a deep think research task.

        Args:
            prompt: The research prompt
            **kwargs: Additional parameters like use_case, depth, parallel_agents

        Yields:
            Research results/chunks
        """
        use_case = kwargs.get('use_case', 'rag')

        # Import the deep research graph (from deep_research extension)
        from src.extensions.content.deep_research.builder import build_research_graph

        # Build graph for this use case
        graph = await build_research_graph(use_case=use_case)

        # Create thread_id if not provided
        thread_id = kwargs.get('thread_id')

        # Run the research
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

        async for chunk in graph.astream({"messages": [{"role": "user", "content": prompt}]}, config):
            yield chunk

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(DeepThinkExtension)
