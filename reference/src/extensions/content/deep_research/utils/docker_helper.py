"""MCP helper utilities for deep research.

This module provides utilities to ensure the MCP web server is accessible
before research starts.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Load environment variables BEFORE any other imports
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger

# Import service registry for URL resolution and mode detection
from src.shared.services.registry import (
    get_registry,
    ServiceMode,
    ServiceStatus,
)

logger = get_logger(__name__, __name__.split('.')[-1])


async def check_mcp_services() -> dict:
    """Check if MCP web server is accessible.

    Uses the Service Registry to check the MCP server.

    Returns:
        Dict with service status:
        {
            "mcp_running": bool,
            "all_running": bool,
            "mode": "local" | "remote" | "explicit"
        }
    """
    registry = get_registry()

    # Check MCP web service
    mcp_status = await registry.check_service("mcp_web")

    mode = mcp_status.mode.value  # "local", "remote", or "explicit"

    logger.debug(
        f"[MCP_HELPER] Service check ({mode}): "
        f"MCP={mcp_status.accessible}"
    )

    return {
        "mcp_running": mcp_status.accessible,
        "all_running": mcp_status.accessible,
        "mode": mode,
    }


async def ensure_mcp_accessible(
    timeout: int = 30
) -> dict:
    """Ensure MCP web server is accessible.

    Checks if the MCP server is accessible. Does NOT start it (it should be
    running on a remote server or started separately).

    Args:
        timeout: Max seconds to wait for services (default: 30)

    Returns:
        Dict with operation result:
        {
            "success": bool,
            "started": bool,
            "already_running": bool,
            "mode": "local" | "remote" | "explicit",
            "error": str | None
        }
    """
    registry = get_registry()

    # Get current mode
    mode = registry.get_mode("mcp_web")
    mode_str = mode.value

    # Check if services are already running
    status = await check_mcp_services()

    if status["all_running"]:
        logger.info(f"[MCP_HELPER] MCP server accessible ({mode_str} mode)")
        return {
            "success": True,
            "started": False,
            "already_running": True,
            "mode": mode_str,
            "error": None,
        }

    # MCP server not accessible
    mcp_url = registry.get_url("mcp_web")
    logger.error(f"[MCP_HELPER] MCP server not accessible at {mcp_url}")
    return {
        "success": False,
        "started": False,
        "already_running": False,
        "mode": mode_str,
        "error": f"MCP server not accessible at {mcp_url}. "
                f"Ensure the MCP server is running.",
    }


def ensure_lang_tools_sync() -> dict:
    """Synchronous wrapper for ensure_mcp_accessible.

    Use this from non-async contexts (like CLI commands).

    Returns:
        Same dict as ensure_mcp_accessible
    """
    return asyncio.run(ensure_mcp_accessible())
