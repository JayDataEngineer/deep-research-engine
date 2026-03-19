"""Store module - LangGraph Store + Checkpointer setup.

Uses lazy imports to avoid requiring psycopg/libpq at import time.
"""

__all__ = [
    "get_store_and_checkpointer",
    "get_store",
    "get_checkpointer",
    "create_search_vectors_tool",
]


def __getattr__(name: str):
    """Lazy import for psycopg-dependent modules."""
    if name in ("get_store_and_checkpointer", "get_store", "get_checkpointer"):
        from context_engine.store.setup import get_checkpointer, get_store, get_store_and_checkpointer
        return locals()[name]

    if name == "create_search_vectors_tool":
        from context_engine.store.tools import create_search_vectors_tool
        return create_search_vectors_tool

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
