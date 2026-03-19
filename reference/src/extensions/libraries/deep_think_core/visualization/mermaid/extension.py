"""
Mermaid Graph Visualizer Extension

Generates Mermaid diagrams from Deep Think configurations.
"""

import logging
from pathlib import Path

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class MermaidVisualizerExtension(LogicExtension):
    """Mermaid graph visualizer extension."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)

    async def initialize(self, context):
        """Initialize the visualizer."""
        await super().initialize(context)
        logger.info("Mermaid visualizer extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def execute(self, config_data: dict, **kwargs):
        """
        Generate a Mermaid diagram from config.

        Args:
            config_data: The configuration dict (e.g., Deep Think node structure)
            **kwargs: Additional parameters

        Yields:
            Mermaid diagram string
        """
        format_type = kwargs.get('format', 'mermaid')

        if format_type == 'mermaid':
            mermaid = self._generate_mermaid(config_data)
            yield mermaid
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _generate_mermaid(self, config: dict) -> str:
        """Generate a Mermaid diagram from configuration."""
        # Simple implementation - generates a basic graph
        # TODO: Implement actual Deep Think config parsing

        nodes = config.get('nodes', [])
        edges = config.get('edges', [])

        lines = ["graph TD"]

        # Add nodes
        for i, node in enumerate(nodes):
            node_id = node.get('id', f'N{i}')
            node_label = node.get('label', f'Node {i}')
            node_type = node.get('type', 'process')

            shape = {
                'process': '[ ]',
                'decision': '{ }',
                'start': '([ ])',
                'end': '([ ])',
            }.get(node_type, '[ ]')

            lines.append(f"  {node_id}{shape} {node_label}")

        # Add edges
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            label = edge.get('label', '')

            if label:
                lines.append(f"  {source} -->|{label}| {target}")
            else:
                lines.append(f"  {source} --> {target}")

        return '\n'.join(lines)

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(MermaidVisualizerExtension)
