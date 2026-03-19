"""LangGraph nodes for deep research workflow."""

from .boss import boss_node, route_after_boss
from .critic import critic_node, route_after_critic
from .synthesis import route_after_synthesis, synthesis_node
from .workers import route_after_workers, workers_node

__all__ = [
    "boss_node",
    "critic_node",
    "route_after_boss",
    "route_after_critic",
    "route_after_synthesis",
    "route_after_workers",
    "synthesis_node",
    "workers_node",
]
