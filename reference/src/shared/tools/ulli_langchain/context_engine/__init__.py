"""
Context Engine Tool Wrappers

Simplified exports for context_engine tools.

Legacy retrieval tools (query_postgres, get_artifact, read_vfs_raw, query_neo4j)
have been replaced by LangGraph Store:
- read_file("/memories/...") replaces get_artifact, read_vfs_raw
- write_file("/memories/...", content) replaces save_finding, save_claim
- search_memories("query") replaces query_postgres

Exports:
- search_memories: Semantic search over agent memories
- ArXiv: Academic paper search
"""

# Re-export from context_engine extension
from src.extensions.features.tools.context_engine.tools.search_memories import search_memories
from src.extensions.features.tools.context_engine.tools.arxiv import (
    search_arxiv,
    get_arxiv_abstract,
)

__all__ = [
    "search_memories",
    "search_arxiv",
    "get_arxiv_abstract",
]
