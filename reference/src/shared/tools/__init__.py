"""Shared Tools Module - LangChain tools for all AI applications.

This module exports tools that can be shared across different modes:
- standard, companion, roleplay, deep_research

Tools are organized by category:
- logic/ - Pure implementation logic
- langchain/ - @tool decorated functions (search, rag, image, etc.)
- registry.py - Central tool catalog
- audio_utils.py - Audio processing utilities
"""

# Import audio utilities
from .audio_utils import AudioService, audio_service

# Import the central registry
from .registry import load_tools, get_all_tools, load_tools_by_category

__all__ = [
    # Audio utilities
    "AudioService",
    "audio_service",
    # Registry
    "load_tools",
    "get_all_tools",
    "load_tools_by_category",
]

# Optional: RAG tools (may not be available)
try:
    from .langchain.memory.rag import (
        vector_search,
        add_documents,
        list_collections,
        initialize_rag,
        get_rag_client,
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Add RAG tools to exports if available
if RAG_AVAILABLE:
    __all__.extend([
        "vector_search",
        "add_documents",
        "list_collections",
        "initialize_rag",
        "get_rag_client",
    ])
