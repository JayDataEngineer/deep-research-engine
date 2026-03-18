"""Deep Research Engine.

Config-driven multi-agent workflow engine for AI ingestion tasks.

Usage:
    from deep_research_engine import run_workflow, load_config
    
    config = load_config("configs/rag.yaml")
    async for event in run_workflow("query", config):
        print(event)
"""

from .engine import run_workflow
from .config import load_config, WorkflowConfig
from .state import EngineState, WorkerResult, CriticResult, create_initial_state
from .workers import BaseWorker, WorkerConfig, WorkerRegistry
from .storage import StorageBackend, StorageRegistry, StorageManager

__version__ = "0.1.0"
__all__ = [
    # Main entry points
    "run_workflow",
    "load_config",
    "WorkflowConfig",
    
    # State types
    "EngineState",
    "WorkerResult",
    "CriticResult",
    "create_initial_state",
    
    # Worker types
    "BaseWorker",
    "WorkerConfig",
    "WorkerRegistry",
    
    # Storage types
    "StorageBackend",
    "StorageRegistry",
    "StorageManager",
]
