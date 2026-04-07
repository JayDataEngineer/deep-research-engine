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
    ArtifactGenerationError,
    ArtifactMakerError,
    ArtifactValidationError,
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
    artifact_maker_node,
    critic_node,
    drafter_node,
    ingestion_node,
    neo4j_builder_node,
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
    "ArtifactGenerationError",
    "ArtifactMakerError",
    "ArtifactValidationError",
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
    "artifact_maker_node",
    "create_initial_state",
    "critic_node",
    "drafter_node",
    "ingestion_node",
    "load_config",
    "neo4j_builder_node",
    "run_workflow",
]
