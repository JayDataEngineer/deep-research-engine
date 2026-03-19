"""Node functions for deep research graph.
Each phase of the research pipeline is implemented as a separate node:
- Boss: Strategy determination and routing to Lab Architect (unified)
- Lab Architect: Pro-Flash hybrid orchestration for deep research
- Critic: Quality gate using Reflexion prompting (unified)
- Synthesis: Final output generation with citations (unified)
- Formatter: Post-synthesis content formatting (tweets, podcast, etc.)

UNIFIED ARCHITECTURE:
- unified_boss_node: Draft + Extract pattern
- unified_critic_node: Draft + Extract pattern
- unified_synthesis_node: Draft + Extract pattern
"""

from .lab_architect import lab_architect_node
from .formatter import formatter_node

# Unified nodes (current architecture)
from .unified import (
    unified_boss_node,
    unified_critic_node,
    unified_synthesis_node,
)

# Aliases for backward compatibility
boss_node = unified_boss_node
critic_node = unified_critic_node

__all__ = [
    "boss_node",
    "lab_architect_node",
    "critic_node",
    "formatter_node",
    # Unified architecture
    "unified_boss_node",
    "unified_critic_node",
    "unified_synthesis_node",
]
