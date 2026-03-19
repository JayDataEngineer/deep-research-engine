"""Tool Call Mode (🧠) - Direct LiteLLM Tool Calling.

Uses LiteLLM's native tool calling to automatically select and execute tools.
No complex router pattern - LiteLLM handles tool selection based on descriptions.

Available tools:
- web_search: Fast web search
- web_get: Page scraping
"""

import logging
from typing import List
from ....core.base import TextModeExtension
from ....core.factory import register_extension_factory
from ..base_router import BaseRouterExtension

logger = logging.getLogger(__name__)


class ToolCallExtension(BaseRouterExtension):
    """Tool Call mode - Direct LiteLLM with tools.

    LiteLLM automatically selects and calls tools based on their descriptions.
    No separate router needed.
    """

    def get_default_tools(self) -> List[str]:
        """Return default tools for tool_call mode.

        Includes:
        - web_search: Fast web search
        - web_get: Page scraping
        """
        return [
            "web_search",
            "web_get",
        ]


register_extension = register_extension_factory(ToolCallExtension)
