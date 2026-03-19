"""LangFuse integration for deep research observability.

With LiteLLM integration, LangFuse tracing is automatic via LiteLLM's
built-in Langfuse integration. This module only provides utilities
for adding evaluation scores to traces.

LiteLLM handles:
- Automatic Langfuse tracing (no callbacks needed)
- Session tracking via metadata in LLM calls

This module provides:
- score_research_quality() - Add quality gate scores
- score_worker_confidence() - Add worker confidence scores
- score_phase_completion() - Add phase completion scores
- score_source_quality() - Add source quality scores
- score_token_efficiency() - Add token efficiency scores

Usage:
    from src.extensions.integrations.langfuse import (
        score_research_quality,
        score_worker_confidence,
    )

    # Add quality gate score
    score_research_quality(
        trace_id="trace-123",
        score=0.85,
        passed=True,
        comment="Meets all quality criteria",
    )

    # Add worker confidence score
    score_worker_confidence(
        trace_id="trace-123",
        worker_id="researcher_0",
        confidence=0.92,
    )

Environment Variables (for LiteLLM):
    LITELLM_BASE_URL - LiteLLM proxy URL (default: http://litellm.local/v1)
    LITELLM_API_KEY - LiteLLM API key

    LangFuse is configured in LiteLLM's config, not here.
    See: ~/Documents/programs/shared-docker-infra/litellm/
"""

from .client import (
    get_langfuse_client,
    is_langfuse_enabled,
    flush_langfuse_client,
    shutdown_langfuse_client,
    reset_langfuse_client,
)

from .scores import (
    score_research_quality,
    score_worker_confidence,
    score_phase_completion,
    score_source_quality,
    score_token_efficiency,
)

__all__ = [
    # Client management (for scores)
    "get_langfuse_client",
    "is_langfuse_enabled",
    "flush_langfuse_client",
    "shutdown_langfuse_client",
    "reset_langfuse_client",
    # Score utilities
    "score_research_quality",
    "score_worker_confidence",
    "score_phase_completion",
    "score_source_quality",
    "score_token_efficiency",
]

__version__ = "2.0.0"  # Simplified for LiteLLM integration
