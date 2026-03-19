"""Unified deep research nodes using draft_extract pattern.

Universal pattern for ALL inference nodes:
1. DRAFT: deep_agent with inference_tools, no output schema
2. EXTRACT: llm.with_retry().with_structured_output(Schema)

All nodes share same code - differ only in:
- System prompt
- Output schema
"""

from .base import draft_extract, INFERENCE_TOOLS
from .boss import unified_boss_node, BossDecision
from .critic import unified_critic_node, CriticEvaluation
from .synthesis import unified_synthesis_node, SynthesisOutput

__all__ = [
    # Universal pattern
    "draft_extract",
    "INFERENCE_TOOLS",
    # Unified nodes
    "unified_boss_node",
    "unified_critic_node",
    "unified_synthesis_node",
    # Schemas
    "BossDecision",
    "CriticEvaluation",
    "SynthesisOutput",
]
