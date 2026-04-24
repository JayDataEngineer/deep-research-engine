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
]
