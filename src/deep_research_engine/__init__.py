"""Deep Research Engine.

Config-driven multi-agent workflow engine for AI ingestion tasks.

Usage:
    from deep_research_engine import run_workflow, load_config

    config = load_config("configs/rag.yaml")
    async for event in run_workflow("query", config):
        print(event)
"""

from deep_research_engine.config import WorkflowConfig, load_config
from deep_research_engine.engine import run_workflow
from deep_research_engine.errors import (
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    DeepResearchError,
    GraphCompilationError,
    GraphError,
    GraphExecutionError,
    GraphStateError,
    NodeError,
    NodeExecutionError,
    NodeNotFoundError,
    StorageConnectionError,
    StorageError,
    StorageInitializationError,
    StorageNotFoundError,
    StorageQueryError,
    SynthesisError,
    SynthesisGenerationError,
    SynthesisValidationError,
    WorkerError,
    WorkerExecutionError,
    WorkerNotFoundError,
    WorkerSetupError,
    WorkerTimeoutError,
    WorkflowError,
    WorkflowExecutionError,
    WorkflowNotFoundError,
    WorkflowValidationError,
)
from deep_research_engine.nodes import (
    ingestion_node,
    drafter_node,
    critic_node,
    neo4j_builder_node,
    synthesizer_node,
)
from deep_research_engine.state import (
    CriticResult,
    EngineState,
    WorkerResult,
    create_initial_state,
)
from deep_research_engine.storage import (
    StorageBackend,
    StorageManager,
    StorageRegistry,
)
from deep_research_engine.version import __version__
from deep_research_engine.workers import (
    BaseWorker,
    WorkerConfig,
    WorkerRegistry,
)

__all__ = [
    "BaseWorker",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "CriticResult",
    "DeepResearchError",
    "EngineState",
    "GraphCompilationError",
    "GraphError",
    "GraphExecutionError",
    "GraphStateError",
    "NodeError",
    "NodeExecutionError",
    "NodeNotFoundError",
    "StorageBackend",
    "StorageConnectionError",
    "StorageError",
    "StorageInitializationError",
    "StorageManager",
    "StorageNotFoundError",
    "StorageQueryError",
    "StorageRegistry",
    "SynthesisError",
    "SynthesisGenerationError",
    "SynthesisValidationError",
    "WorkerConfig",
    "WorkerError",
    "WorkerExecutionError",
    "WorkerNotFoundError",
    "WorkerRegistry",
    "WorkerResult",
    "WorkerSetupError",
    "WorkerTimeoutError",
    "WorkflowConfig",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowNotFoundError",
    "WorkflowValidationError",
    "__version__",
    "ingestion_node",
    "drafter_node",
    "create_initial_state",
    "critic_node",
    "neo4j_builder_node",
    "synthesizer_node",
    "load_config",
    "run_workflow",
]
