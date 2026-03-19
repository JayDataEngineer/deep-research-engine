"""Configuration for web tools library.

All web operations go through the MCP server.
No local fallbacks - MCP or bust.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Tool toggles
ENABLE_WEB_TOOLS = os.getenv("ENABLE_WEB_TOOLS", "true").lower() == "true"
ENABLE_OSINT_TOOLS = os.getenv("ENABLE_OSINT_TOOLS", "false").lower() == "true"

# =============================================================================
# MCP Server Configuration
# =============================================================================
# MCP server for web search/scrape - use direct Tailscale IP
CUSTOM_MCP_BASE_URL = os.getenv("MCP_WEB_SERVER_URL", "http://100.102.244.97:8000")
