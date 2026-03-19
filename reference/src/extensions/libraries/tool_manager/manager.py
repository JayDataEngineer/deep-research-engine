"""Tool Manager

Centralized management of tool availability and configuration.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ToolManager:
    """Manages tool registration and mode configuration."""

    def __init__(self):
        # Import tool registry
        from src.shared.tools.registry import (
            TOOL_CATALOG,
            get_all_tools,
            get_tool,
        )

        self.TOOL_CATALOG = TOOL_CATALOG
        self._all_tools = get_all_tools()

        # Mode-to-tools mapping (from brain_graph.py)
        self._mode_tools = {
            "standard": ["web_search", "web_get"],
            "companion": ["web_search", "web_get", "vector_search", "add_documents"],
            "roleplay": ["web_search", "edit_character_image", "set_character_reference", "clear_character_reference"],
            "deep_research": ["web_search", "web_get", "web_map", "web_crawl", "deep_agent"],
            "rp": ["web_search", "edit_character_image", "set_character_reference", "clear_character_reference"],
        }

    def list_all_tools(self) -> List[str]:
        """Get list of all available tools."""
        return self._all_tools

    def list_tools_for_mode(self, mode: str) -> List[str]:
        """Get tools enabled for a specific mode."""
        return self._mode_tools.get(mode, self._mode_tools.get("companion", []))

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        from src.shared.tools.registry import get_tool
        tool = get_tool(tool_name)
        if tool:
            return {
                "name": tool_name,
                "description": tool.description if hasattr(tool, 'description') else "No description",
                "function": tool.__name__ if hasattr(tool, '__name__') else tool_name,
            }
        return None

    def enable_tool_for_mode(self, mode: str, tool_name: str) -> bool:
        """Enable a tool for a specific mode."""
        if tool_name not in self._all_tools:
            logger.warning(f"Unknown tool: {tool_name}")
            return False

        if mode not in self._mode_tools:
            self._mode_tools[mode] = []

        if tool_name not in self._mode_tools[mode]:
            self._mode_tools[mode].append(tool_name)
            logger.info(f"Enabled tool '{tool_name}' for mode '{mode}'")
            return True

        return False

    def disable_tool_for_mode(self, mode: str, tool_name: str) -> bool:
        """Disable a tool for a specific mode."""
        if mode not in self._mode_tools:
            return False

        if tool_name in self._mode_tools[mode]:
            self._mode_tools[mode].remove(tool_name)
            logger.info(f"Disabled tool '{tool_name}' for mode '{mode}'")
            return True

        return False

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a category."""
        category_map = {
            "web": ["web_search", "web_search_and_scrape", "web_get", "web_map", "web_crawl", "fact_check"],
            "image": ["edit_character_image", "set_character_reference", "clear_character_reference"],
            "memory": ["vector_search", "add_documents", "list_collections"],
            "vision": ["analyze_image", "analyze_video", "extract_text_from_image", "diagnose_error",
                      "understand_diagram", "analyze_data_viz", "image_caption", "batch_caption"],
            "osint": ["username_search", "email_investigation", "domain_whois", "breach_lookup", "google_dork"],
            "agent": ["deep_agent", "deep_research_subprocess", "index_tts_subprocess", "run_cli_script"],
        }
        return category_map.get(category, [])
