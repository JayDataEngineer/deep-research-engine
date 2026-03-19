"""Observability API routes for LangGraph checkpoint database.

This module provides the router for LangGraph checkpointer endpoints.
The custom observability UI has been replaced by LangFuse.

Data sources:
- Postgres checkpointer: LangGraph state snapshots

For full observability with traces, scores, and dashboards, use LangFuse:
- Self-hosted: make langfuse-up
- Access: http://localhost:8899/langfuse/
"""

from fastapi import APIRouter

# Import checkpointer router (used for LangGraph state management)
from .checkpointer import router as checkpointer_router

# Create main router with correct API prefix
router = APIRouter(prefix="/observability/api", tags=["observability"])

# Include checkpointer router for LangGraph state management
router.include_router(checkpointer_router)

__all__ = ["router"]
