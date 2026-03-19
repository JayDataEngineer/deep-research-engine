"""Deep Agents Extension API endpoints.

The deep agents mode is primarily accessed via the main /api/v1/chat endpoint with mode="deep_agents".
This router provides deep agents-specific endpoints like health checks.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/deep_agents", tags=["deep_agents"])


@router.get("/health")
async def deep_agents_health():
    """Health check endpoint for deep agents mode.

    Returns the status of deep agents mode features.
    """
    return {
        "status": "healthy",
        "mode": "deep_agents",
        "features": {
            "router_system": True,
            "ultimate_memory": True,
            "subgraphs": ["chat", "web_research", "vector_search", "tools"],
        },
        "message": "Deep Agents mode is operational"
    }
