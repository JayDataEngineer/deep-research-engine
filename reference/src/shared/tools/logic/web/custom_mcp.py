"""MCP Web Tools using langchain-mcp-adapters.

This module provides high-level functions for MCP web operations.
All operations use the proper langchain-mcp-adapters library for
MCP tool integration.

IMPORTANT: This replaces the previous custom SSE implementation with
the standard langchain-mcp-adapters approach.

Reference: https://docs.langchain.com/oss/python/langchain/mcp

Usage:
    from logic.web import mcp_web_search, mcp_web_scrape

    results = await mcp_web_search("AI news")
    content = await mcp_web_scrape("https://example.com")
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Literal, Optional

from .logging_utils import get_tool_logger

logger = get_tool_logger("mcp_web")

# Configuration - use direct Tailscale IP via environment variable
# Direct Tailscale IP: http://100.102.244.97:8000/mcp
CUSTOM_MCP_BASE_URL = os.getenv("MCP_WEB_SERVER_URL", "http://100.102.244.97:8000/mcp")


class MCPWebClient:
    """Client for MCP web server using langchain-mcp-adapters.

    This is the proper LangChain way to integrate MCP tools.
    Uses MultiServerMCPClient with HTTP transport.

    Reference: https://docs.langchain.com/oss/python/langchain/mcp
    """

    def __init__(self, base_url: str = CUSTOM_MCP_BASE_URL, timeout: float = 120.0):
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[Any] = None
        self._tools: Dict[str, Any] = {}
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    async def _ensure_client(self) -> Any:
        """Ensure MCP client is connected using langchain-mcp-adapters."""
        if self._client is not None:
            return self._client

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._client = MultiServerMCPClient(
                {
                    "web_tools": {
                        "transport": "http",  # streamable-http transport
                        "url": self.base_url,
                        "timeout": self.timeout,
                    }
                }
            )

            logger.info(f"[MCP_WEB] Connected to MCP server at {self.base_url}")
            return self._client

        except ImportError:
            logger.error("[MCP_WEB] langchain-mcp-adapters not installed")
            raise RuntimeError(
                "langchain-mcp-adapters required. Install with: uv add langchain-mcp-adapters"
            )
        except Exception as e:
            logger.error(f"[MCP_WEB] Failed to connect to MCP server: {e}")
            raise

    async def _get_tool(self, tool_name: str) -> Any:
        """Get a specific tool by name from the MCP server."""
        if tool_name in self._tools:
            return self._tools[tool_name]

        client = await self._ensure_client()

        try:
            tools = await client.get_tools()
            for tool in tools:
                self._tools[tool.name] = tool

            if tool_name in self._tools:
                return self._tools[tool_name]

            raise RuntimeError(f"Tool '{tool_name}' not found on MCP server")

        except Exception as e:
            logger.error(f"[MCP_WEB] Failed to get tool '{tool_name}': {e}")
            raise

    async def list_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """List available tools from MCP server.

        Args:
            force_refresh: Force refresh cache

        Returns:
            List of tool definitions
        """
        if self._tools_cache and not force_refresh:
            return self._tools_cache

        try:
            client = await self._ensure_client()
            tools = await client.get_tools()

            self._tools_cache = [
                {"name": t.name, "description": t.description}
                for t in tools
            ]

            logger.info(f"[MCP_WEB] Available tools: {[t.get('name') for t in self._tools_cache]}")
            return self._tools_cache

        except Exception as e:
            logger.error(f"[MCP_WEB] Failed to list tools: {e}")
            return []

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool on the MCP server using langchain-mcp-adapters.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result (extracted from MCP response)
        """
        logger.info(f"[MCP_WEB] Calling tool: {tool_name}({list(arguments.keys())})")

        try:
            tool = await self._get_tool(tool_name)
            result = await tool.ainvoke(arguments)

            # Handle structured tool responses
            # MCP tools return content in various formats
            if hasattr(result, 'content'):
                # StructuredTool response
                return result.content
            elif isinstance(result, str):
                return result
            elif isinstance(result, dict):
                # May have structured content
                if "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        first = content[0]
                        if isinstance(first, dict) and first.get("type") == "text":
                            return first.get("text", "")
                return json.dumps(result)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"[MCP_WEB] Tool call failed: {e}")
            raise

    async def close(self) -> None:
        """Close the MCP client and cleanup resources."""
        self._client = None
        self._tools = {}
        self._tools_cache = None
        logger.info("[MCP_WEB] Client closed")


# =============================================================================
# Global client instance with connection pooling
# =============================================================================

_global_client: Optional[MCPWebClient] = None
_client_lock = asyncio.Lock()


async def get_mcp_client() -> MCPWebClient:
    """Get or create the global MCP client instance.

    Uses connection pooling for efficiency.
    """
    global _global_client

    async with _client_lock:
        if _global_client is None:
            _global_client = MCPWebClient()
        return _global_client


async def close_mcp_client() -> None:
    """Close the global MCP client."""
    global _global_client

    async with _client_lock:
        if _global_client:
            await _global_client.close()
            _global_client = None


# =============================================================================
# High-level API functions (matching previous interface)
# =============================================================================

async def mcp_web_search(
    query: str,
    max_results: int = 10,
    recency: str | None = None,
    category: str = "general",
) -> str:
    """Search the web using MCP server via langchain-mcp-adapters.

    Args:
        query: Search query
        max_results: Maximum results to return
        recency: Time filter (not supported by all backends)
        category: Search category (not supported by all backends)

    Returns:
        Formatted search results
    """
    try:
        client = await get_mcp_client()

        # MCP server uses search_web tool
        arguments = {
            "query": query,
            "pages": min(10, max(1, (max_results + 9) // 10)),  # Convert to pages
            "exclude_blacklist": True,
        }

        result = await client.call_tool("search_web", arguments)
        logger.info(f"[MCP_WEB] Search returned {len(str(result))} chars")
        return str(result)

    except Exception as e:
        logger.error(f"[MCP_WEB] Search failed: {e}")
        return f"MCP search error: {str(e)}"


async def mcp_web_fetch(url: str) -> str:
    """Fetch content from a URL using MCP server via langchain-mcp-adapters.

    Args:
        url: URL to fetch

    Returns:
        Page content
    """
    try:
        client = await get_mcp_client()

        arguments = {"url": url}

        result = await client.call_tool("scrape_url", arguments)
        logger.info(f"[MCP_WEB] Fetch returned {len(str(result))} chars for {url[:50]}")
        return str(result)

    except Exception as e:
        logger.error(f"[MCP_WEB] Fetch failed: {e}")
        return f"MCP fetch error: {str(e)}"


async def mcp_web_scrape(url: str) -> str:
    """Scrape content from a URL using MCP server via langchain-mcp-adapters.

    Args:
        url: URL to scrape

    Returns:
        Scraped content in markdown format
    """
    try:
        client = await get_mcp_client()

        arguments = {"url": url}

        result = await client.call_tool("scrape_url", arguments)
        logger.info(f"[MCP_WEB] Scrape returned {len(str(result))} chars for {url[:50]}")
        return str(result)

    except Exception as e:
        logger.error(f"[MCP_WEB] Scrape failed: {e}")
        return f"MCP scrape error: {str(e)}"


__all__ = [
    "MCPWebClient",
    "get_mcp_client",
    "close_mcp_client",
    "mcp_web_search",
    "mcp_web_fetch",
    "mcp_web_scrape",
    "CUSTOM_MCP_BASE_URL",
]
