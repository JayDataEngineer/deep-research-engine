"""Lorebook Engine - Keyword and Semantic Activation.

This library provides:
- Pydantic schemas for SillyTavern-compatible lorebooks
- Keyword-based activation with AND/OR logic
- Regex pattern matching
- Semantic activation using vector embeddings
- Token budget management
- Position-based injection (before_char, after_char)

Example:
    from src.extensions.libraries.lorebook_engine import (
        Lorebook,
        LorebookEntry,
        LorebookRegistry,
        LorebookTrigger,
        VectorActivator,
    )

    # Create lorebook
    lorebook = Lorebook(name="My World", entries=[...])

    # Keyword activation
    trigger = LorebookTrigger()
    result = trigger.trigger(entries, conversation_text)

    # Semantic activation
    vector = VectorActivator(embedding_model)
    vector.index_entries(entries)
    result = vector.activate(entries, query_text)
"""

# Schemas
from .schema import (
    Lorebook,
    LorebookEntry,
    LorebookRegistry,
    LorebookExtensions,
    DEFAULT_LOREBOOKS,
)

# Trigger
from .trigger import KeywordMatcher, LorebookTrigger, TriggerMatch

# Result
from .result import ActivationResult

# Vector
from .vector import VectorActivator

# API Router
from .routes import router as lorebooks_router

__all__ = [
    # Schemas
    "Lorebook",
    "LorebookEntry",
    "LorebookRegistry",
    "LorebookExtensions",
    "DEFAULT_LOREBOOKS",
    # Trigger
    "KeywordMatcher",
    "LorebookTrigger",
    "TriggerMatch",
    # Result
    "ActivationResult",
    # Vector
    "VectorActivator",
    # API Router
    "lorebooks_router",
]
