"""Research Sessions - Session management for deep research.

.. deprecated:: 1.0.0
    This module is DEPRECATED. Use context_engine session tools instead.

    Migration Guide:
        - OLD: from .sessions import SessionManager, SessionContext
        - NEW: from src.shared.tools.registry import get_tool
               create_session = get_tool("create_research_session")

    All session management is now handled through context_engine:
    - Session creation: `create_research_session` tool
    - Session status: `get_session_status` tool
    - Session updates: `update_session_status` tool
    - Session cleanup: `cleanup_session` tool

    See: src/extensions/features/tools/context_engine/tools/session.py

    This module is kept for backward compatibility only.
"""

import warnings

# Issue deprecation warning when module is imported
warnings.warn(
    "The 'sessions' module is deprecated. Use context_engine session tools instead. "
    "See: src/extensions/features/tools/context_engine/tools/session.py",
    DeprecationWarning,
    stacklevel=2
)

from .research_sessions import (
    SessionStatus,
    ResearchSession,
    SessionStorage,
    SessionManager,
    SessionContext,
    get_session_manager,
    get_session_manager_sync,
)

__all__ = [
    "SessionStatus",
    "ResearchSession",
    "SessionStorage",
    "SessionManager",
    "SessionContext",
    "get_session_manager",
    "get_session_manager_sync",
]
