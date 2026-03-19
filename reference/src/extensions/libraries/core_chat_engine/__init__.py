"""
Companion Brain - Generalist AI Companion with RP Mode

Uses LangGraph StateGraph with LangChain tool calling:
- Native LangChain StructuredTool format
- LLM decides when to call tools (no separate intent classifier)
- ToolNode for tool execution
- Better integration with LangChain ecosystem
"""

from .config.brain_config import BrainConfig
from .state.brain_state import BrainState, UserProfile, ProcessedResponse
from .state.intent_types import IntentType, IntentResult
from .config.personality_config import PersonalityConfig

# LangGraph v2 with LangChain tool calling
# Renamed graph.py to brain_graph.py to avoid conflict with graph/ folder (Neo4j)
from .brain_graph import CompanionBrainGraphV2, create_companion_brain_graph_v2

# Research brain (Spawner integration)
try:
    from .graph_brain_research import ResearchBrainGraph, create_research_brain
    _has_research = True
except ImportError:
    _has_research = False
    ResearchBrainGraph = None
    create_research_brain = None

__all__ = [
    # Main brain class
    "CompanionBrainGraphV2",
    "create_companion_brain_graph_v2",
    # Research brain (Spawner)
    "ResearchBrainGraph",
    "create_research_brain",
    # Common exports
    "BrainConfig",
    "ProcessedResponse",
    "BrainState",
    "UserProfile",
    "IntentType",
    "IntentResult",
    "PersonalityConfig",
]

__version__ = "3.0.0"
