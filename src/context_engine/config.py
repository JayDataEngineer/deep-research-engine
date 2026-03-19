"""Configuration - Models, store config.

All models via LiteLLM proxy only.
"""

import os
from typing import Literal, Optional

# =============================================================================
# MODEL TIERS (LiteLLM only, no parameters)
# =============================================================================

SMART_MODEL = os.getenv("SMART_MODEL", "claude-sonnet-4-6")  # Main agent, Critic
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-4-5")    # Flash explorers, specialists

ModelTier = Literal["smart", "fast"]


def get_model(tier: ModelTier = "smart"):
    """Get model via LiteLLM proxy.

    Args:
        tier: "smart" (Sonnet) or "fast" (Haiku)

    Returns:
        ChatLiteLLM instance

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

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://ulli:ulli_password@localhost:5432/context_engine"
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "384"))


# =============================================================================
# LANGFUSE CONFIGURATION
# =============================================================================

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


# =============================================================================
# OPTIONAL: MINIO (for large files)
# =============================================================================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")


# =============================================================================
# OPTIONAL: NEO4J
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
