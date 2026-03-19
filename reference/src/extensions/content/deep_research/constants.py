"""Spawner Core Constants

Centralized magic numbers and configuration defaults to improve maintainability
and make codebase more DRY.

IMPORTANT: Many configuration values are now DERIVED from model capabilities
instead of being hardcoded magic numbers. See model_capabilities.py for details.
"""

import os
import sys
from pathlib import Path

# Import system-wide constants from central location
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.constants import (
    DATA_DIR,
    ROOT_DATA_DIR,
    # LLAMACPP_PORT removed - now using llama.local via shared infrastructure
    OLLAMA_PORT,
    NEO4J_PORT,
    MINIO_PORT,
    POSTGRES_PORT,
    POSTGRES_ALT_PORT,
)

# Default llama.cpp port (for backward compatibility, but prefer llama.local)
LLAMACPP_PORT = 18734

# =============================================================================
# CONNECTION STRING DEFAULTS
# =============================================================================
# Centralized connection strings with environment variable overrides.
# These are used as defaults when config.yaml doesn't specify values.
# Environment variables take precedence over these defaults.

DEFAULT_POSTGRES_CONNECTION_STRING = os.getenv(
    "CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING",
    f"postgresql://ulli:ulli_password@localhost:{POSTGRES_ALT_PORT}/ulli_db"
)

DEFAULT_NEO4J_URL = os.getenv(
    "NEO4J_URL",
    f"bolt://localhost:{NEO4J_PORT}"
)

DEFAULT_NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME",
    "neo4j"
)

DEFAULT_NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "neo4j_test_password"
)

DEFAULT_MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    f"http://localhost:{MINIO_PORT}"
)

DEFAULT_MINIO_ACCESS_KEY = os.getenv(
    "MINIO_ACCESS_KEY",
    "minioadmin"
)

DEFAULT_MINIO_SECRET_KEY = os.getenv(
    "MINIO_SECRET_KEY",
    "minioadmin"
)

DEFAULT_MINIO_BUCKET = os.getenv(
    "MINIO_BUCKET",
    "context-engine"
)

# =============================================================================
# Module Paths (self-contained, no src.constants dependency)
# =============================================================================

# This module's location
_MODULE_DIR = Path(__file__).parent

# Backend directory (where docker-compose files live)
# From: backend/src/extensions/content/deep_research/constants.py
# To:   backend/
BACKEND_DIR = _MODULE_DIR.parent.parent.parent.parent

# Project root (one level up from backend)
PROJECT_ROOT = BACKEND_DIR.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Deep research config directory
DEEP_RESEARCH_CONFIG_DIR = DATA_DIR / "config" / "deep_research"

# Research extensions directory
RESEARCH_EXTENSIONS_DIR = PROJECT_ROOT / "backend" / "src" / "extensions" / "content" / "deep_research"

# =============================================================================
# LLM Token Limits
# =============================================================================
# LLM Token Limits
# =============================================================================

DEFAULT_MAX_TOKENS = 2000  # Default max tokens for most LLM calls
HIGH_MAX_TOKENS = 12000  # High token limit for comprehensive tasks
MAX_MAX_TOKENS = 32000  # Maximum token limit (for full video scripts, etc)

# Token overhead estimates for context budgeting
ESTIMATED_SYSTEM_TOKENS = 3500  # Estimated tokens for system messages
ESTIMATED_TEMPLATE_TOKENS = 1500  # Estimated tokens for prompt templates

# =============================================================================
# Vector Store & Search Constants
# =============================================================================

DEFAULT_SEARCH_TOP_K = 5  # Default number of vector search results
HIGH_SEARCH_TOP_K = 10  # Higher number for comprehensive search
OSINT_SEARCH_TOP_K = 15  # Even higher for OSINT use case

FLASHRANK_INITIAL_K = 50  # Initial K for flashrank reranking
FLASHRANK_THRESHOLD_HIGH = 50  # High threshold for flashrank

MAX_CONTENT_PREVIEW = 500  # Max characters for document preview
MAX_SNIPPET_LINES = 50  # Max lines for web content snippet

# =============================================================================
# Web Crawler & Mapper Constants
# =============================================================================

DEFAULT_WEB_MAP_LIMIT = 5000  # Default max URLs to discover
MAX_WEB_MAP_LIMIT = 30000  # Maximum URLs for web mapping
DEFAULT_WEB_CRAWL_LIMIT = 100  # Default max pages to crawl

# =============================================================================
# Timeout & Retry Constants
# =============================================================================

DEFAULT_TIMEOUT = 30  # Default timeout in seconds for API calls
LONG_TIMEOUT = 60  # Longer timeout for expensive operations
CRAWL_TIMEOUT = 120  # Timeout for web crawling operations

DEFAULT_RECURSION_LIMIT = 10  # Default LangGraph recursion limit
HIGH_RECURSION_LIMIT = 15  # Higher limit for complex workflows
DEFAULT_REMAINING_STEPS = 999  # Default remaining steps when not explicitly set

# =============================================================================
# Worker Execution Constants
# =============================================================================

MIN_WORKERS = 1  # Minimum number of workers
MAX_WORKERS = 10  # Maximum number of workers
DEFAULT_WORKERS = 3  # Default worker count

# =============================================================================
# Configuration File Constants
# =============================================================================

DEFAULT_USE_CASE = "rag"  # Default use case
DEFAULT_VARIANT = "default"  # Default config variant

# =============================================================================
# Output Formatting Constants
# =============================================================================

MAX_LINE_LENGTH = 100  # Max line length for code formatting
DEFAULT_INDENT = 2  # Default indentation spaces

# =============================================================================
# Testing & Development Constants
# =============================================================================

TEST_QUERY_LENGTH = 50  # Typical length of test queries
MOCK_API_KEY = "test-key"  # Mock API key for testing

# =============================================================================
# Feature Flags (for experimental features)
# =============================================================================

ENABLE_SEQUENTIAL_MODE = True  # Keep sequential mode for special cases
ENABLE_DYNAMIC_GRAPH_MODE = True  # Enable topological sorting for dependencies
ENABLE_PARALLEL_MODE = True  # Enable parallel worker execution

# =============================================================================
# NEEDS_RESEARCH KEYWORDS
# =============================================================================
# These keywords determine which worker types get research tools.
# If a worker_type contains any of these keywords, it will be assigned
# web_search, web_reader, and other research tools.
#
# IMPORTANT: When adding new specialist worker types, add their keywords here
# AND update validation.py to check for them.
#
# See: agents/flash_explorer.py (research tools assignment)
# See: validation/config_validation.py (worker tools validation)
# =============================================================================

NEEDS_RESEARCH_KEYWORDS = {
    'research',
    'aggregator',
    'news',
    'analyst',
    'character',
    'generic',
    'motivation',
    'hammer',
    'anti',
    'excuse',
    'truth',
    'bomb',
    'habit',
    'builder',
    'community',
    'call',
    'synthesis',
    'bridger',
    'framer',
    'hunter',
    'connector',
    'philosopher',
    'fitness',
    'relevancy',
    'marketer',
    'trend',
    'audience',
    'engagement',
}


def worker_needs_research(worker_type: str) -> bool:
    """Check if a worker type needs research tools.

    Args:
        worker_type: The worker type string to check

    Returns:
        True if the worker should get research tools, False otherwise

    Example:
        >>> worker_needs_research("news_analyst")
        True
        >>> worker_needs_research("formatting_bot")
        False
    """
    if not worker_type:
        return False

    worker_lower = worker_type.lower()
    return any(keyword in worker_lower for keyword in NEEDS_RESEARCH_KEYWORDS)
