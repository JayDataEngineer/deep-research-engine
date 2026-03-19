"""Configuration - Models, store config.

All models via LiteLLM proxy only.
Aligns with .env.example and shared-docker-infra settings.
"""

import os
from typing import Literal, Optional

# =============================================================================
# MODEL TIERS (LiteLLM only, no parameters)
# =============================================================================

# Support both naming conventions for compatibility
SMART_MODEL = os.getenv("SMART_MODEL") or os.getenv("MAIN_MODEL", "claude-sonnet-4-6")
FAST_MODEL = os.getenv("FAST_MODEL") or os.getenv("TOOL_MODEL", "claude-haiku-4-5")

ModelTier = Literal["smart", "fast"]


def get_model(tier: ModelTier = "smart"):
    """Get model via LiteLLM proxy.

    Args:
        tier: "smart" (Sonnet) or "fast" (Haiku)

    Returns:
        ChatOpenAI instance configured for LiteLLM

    Note:
        All parameters handled by LiteLLM proxy config.
        We only specify model name.
    """
    from langchain_openai import ChatOpenAI

    model = SMART_MODEL if tier == "smart" else FAST_MODEL

    # Use LiteLLM as OpenAI-compatible endpoint
    base_url = os.getenv("LITELLM_BASE_URL", "http://litellm.local:4000/v1")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=os.getenv("LITELLM_API_KEY", "fake-key"),  # LiteLLM doesn't need real key
    )


# =============================================================================
# STORE CONFIGURATION
# =============================================================================

# Support both naming conventions for shared-docker-infra compatibility
POSTGRES_URL = (
    os.getenv("POSTGRES_URL") or
    os.getenv("CONTEXT_POSTGRES_URL") or
    "postgresql://postgres:postgres@localhost:5433/deep_research"
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "384"))


# =============================================================================
# LANGFUSE CONFIGURATION
# =============================================================================

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://langfuse.local")


# =============================================================================
# OPTIONAL: MINIO (for large files)
# =============================================================================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")


# =============================================================================
# OPTIONAL: NEO4j
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# Support both NEO4J_USER and NEO4J_USERNAME
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


# =============================================================================
# OPTIONAL: MONGODB
# =============================================================================

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
