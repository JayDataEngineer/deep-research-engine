"""
Context Engine tools.

Simplified tool set using LangGraph Store for semantic memory:

- search_memories: Semantic search over agent memories (LangGraph Store)
- think_tool: Strategic reflection during research (LangChain pattern)
- ArXiv: Academic paper search
- Observability: Logging, metrics, run history
- Session: Research session management

Legacy tools (query_postgres, get_artifact, save_finding, etc.) have been
replaced by LangGraph Store with automatic embeddings:
- read_file("/memories/...") replaces get_artifact, read_vfs_raw
- write_file("/memories/...", content) replaces save_finding, save_claim
- search_memories("query") replaces query_postgres
"""

from .search_memories import search_memories
from .think import think_tool
from .arxiv import (
    search_arxiv,
    get_arxiv_abstract
)
from .observability import (
    log_observation,
    save_metrics,
    get_logs,
    get_run_history,
    save_output,
    CONTEXT_ENGINE_OBSERVABILITY_TOOLS,
)
from .session import (
    create_research_session,
    get_session_status,
    update_session_status,
    list_sessions,
    cleanup_session,
    SessionStatus,
    CONTEXT_ENGINE_SESSION_TOOLS,
)

# All available tools
ALL_CONTEXT_ENGINE_TOOLS = [
    # Memory (LangGraph Store semantic search)
    search_memories,
    # Strategic reflection (LangChain pattern)
    think_tool,
    # ArXiv
    search_arxiv,
    get_arxiv_abstract,
    # Observability
    log_observation,
    save_metrics,
    get_logs,
    get_run_history,
    save_output,
    # Session
    create_research_session,
    get_session_status,
    update_session_status,
    list_sessions,
    cleanup_session,
]

__all__ = [
    # Memory (LangGraph Store)
    "search_memories",
    # Strategic reflection
    "think_tool",
    # ArXiv
    "search_arxiv",
    "get_arxiv_abstract",
    # Observability
    "log_observation",
    "save_metrics",
    "get_logs",
    "get_run_history",
    "save_output",
    "CONTEXT_ENGINE_OBSERVABILITY_TOOLS",
    # Session
    "create_research_session",
    "get_session_status",
    "update_session_status",
    "list_sessions",
    "cleanup_session",
    "SessionStatus",
    "CONTEXT_ENGINE_SESSION_TOOLS",
    # All
    "ALL_CONTEXT_ENGINE_TOOLS",
]
