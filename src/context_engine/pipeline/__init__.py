"""Pipeline module - Orchestration."""

from context_engine.pipeline.runner import build_context, run_pipeline

__all__ = [
    "build_context",
    "run_pipeline",
]
