"""Embedding Factory - Centralized embedding model instantiation.

This module provides a factory pattern for creating embedding instances
that support multiple providers (OpenAI, local, ZAI, HuggingFace, etc.).

Example:
    from src.shared.embeddings.factory import get_embeddings

    embeddings = get_embeddings()  # Uses default from config
    embeddings = get_embeddings(provider="local", base_url="http://localhost:8080")
    embeddings = get_embeddings_from_config(config_dict)  # From config dict
"""

from .factory import (
    get_embeddings,
    get_embeddings_from_config,
    get_embedding_model,
    clear_embeddings_cache,
)

__all__ = [
    "get_embeddings",
    "get_embeddings_from_config",
    "get_embedding_model",
    "clear_embeddings_cache",
]
