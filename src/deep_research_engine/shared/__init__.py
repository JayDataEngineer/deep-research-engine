"""Shared core utilities for the Deep Research Engine.

This module provides:
1. LLM access via LiteLLM (chat_llm)
2. Multimodal utilities (multimodal.py)
3. Identity resolution types
"""

# Identity Resolution Types
from pydantic import BaseModel, Field


class IdentityMapping(BaseModel):
    """Mapping from generic IDs to names."""

    mapping: dict[str, str] = Field(
        description="Mapping from Speaker ID or Visual Cluster ID to real name"
    )


# LLM utilities
from .chat_llm import ChatLLM, get_chat_llm, get_langchain_llm

# Configuration
from .config import (
    DEFAULT_MODEL,
    INGESTION_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TOOL_MODEL,
    get_llm_endpoint,
    is_cloud_model,
    map_model_name,
)

# Orchestrator SDK bridge
from .orch_client import (
    get_orch,
    is_available as orch_is_available,
    orch_bash,
    orch_crawl,
    orch_deep_research,
    orch_extract,
    orch_research,
    orch_scrape,
    orch_search,
)

__all__ = [
    # Types
    "IdentityMapping",
    # LLM
    "get_chat_llm",
    "get_langchain_llm",
    "ChatLLM",
    # Config
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    "DEFAULT_MODEL",
    "INGESTION_MODEL",
    "TOOL_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "get_llm_endpoint",
    "is_cloud_model",
    "map_model_name",
    # Orchestrator SDK
    "get_orch",
    "orch_is_available",
    "orch_search",
    "orch_scrape",
    "orch_research",
    "orch_crawl",
    "orch_extract",
    "orch_bash",
    "orch_deep_research",
]
