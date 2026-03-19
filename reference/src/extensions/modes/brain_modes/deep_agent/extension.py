"""Deep Agent Mode (🧠🧠) - Direct LiteLLM Tool Calling with Advanced Tools.

Uses LiteLLM's native tool calling with advanced research tools.
LiteLLM automatically selects between fast search and deep research.

Available tools:
- web_search: Fast web search
- web_get: Page scraping  
- deep_agent: Multi-step autonomous research
- web_research_agent: Deep autonomous web research
"""

from typing import List
from ....core.base import TextModeExtension
from ....core.factory import register_extension_factory
from ..base_router import BaseRouterExtension
from .....shared.logging_utils import get_extension_logger

logger = get_extension_logger("deep_agent")


class DeepAgentExtension(BaseRouterExtension):
    """Deep Agent mode - LiteLLM with advanced research tools.

    Includes deep_agent for complex multi-step research.
    LiteLLM handles tool selection automatically.
    """

    def get_default_tools(self) -> List[str]:
        """Return default tools for deep_agent mode.

        Includes:
        - web_search: Fast web search
        - web_get: Page scraping
        - deep_agent: Multi-step autonomous planning
        - web_research_agent: Deep autonomous web research
        """
        return [
            "web_search",
            "web_get",
            "deep_agent",
            "web_research_agent",
        ]


register_extension = register_extension_factory(DeepAgentExtension)
