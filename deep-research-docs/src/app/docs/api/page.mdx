# API Reference

Complete API reference for the Deep Research Engine.

## Main Entry Points

### run_workflow

Run a workflow with the given configuration.

**Signature:**
```python
async def run_workflow(
    input_data: Union[str, Dict[str, Any]],
    config: WorkflowConfig,
    session_id: Optional[str] = None,
    stream: bool = True,
    storage_backends: Optional[List[str]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Run a workflow with the given configuration.
    
    Args:
        input_data: Query string or dict with workflow input
        config: Workflow configuration
        session_id: Optional session ID for tracing
        stream: If True, yield events as they happen
        storage_backends: List of backends to enable
        
    Yields:
        Events (if streaming) or final result
        
    Example:
        >>> config = load_config("configs/rag.yaml")
        >>> async for event in run_workflow("Research query", config):
        ...     print(event)
    """
```

**Event Types:**
```python
# Phase start
{"type": "phase_start", "phase": "boss"}

# Phase complete
{"type": "phase_complete", "phase": "boss", "duration": 1.5}

# Phase retry
{"type": "phase_retry", "phase": "workers", "reason": "critic_revision"}

# Worker complete
{"type": "worker_complete", "worker_id": "worker_1", "success": True}

# Synthesis complete
{"type": "synthesis_complete", "output": "# Report..."}

# Error
{"type": "error", "error": "Error message", "phase": "workers"}

# Complete
{"type": "complete", "success": True, "output": "Final output", "state": {...}}
```

**Usage:**
```python
from deep_research_engine import run_workflow, load_config

# Streaming
config = load_config("configs/rag.yaml")
async for event in run_workflow("What is quantum computing?", config):
    if event["type"] == "complete":
        print(f"Result: {event['output']}")

# Non-streaming
async for event in run_workflow("Query", config, stream=False):
    result = event  # Single complete event

print(result["output"])
```

### load_config

Load workflow configuration from YAML file.

**Signature:**
```python
def load_config(config_path: str) -> WorkflowConfig:
    """Load workflow configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        WorkflowConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config validation fails
        
    Example:
        >>> config = load_config("configs/rag.yaml")
        >>> print(config.workflow.name)
        'rag'
    """
```

**Usage:**
```python
from deep_research_engine import load_config

# Load from relative path
config = load_config("configs/rag.yaml")

# Load from absolute path
config = load_config("/path/to/config.yaml")

# Access config properties
print(f"Workflow: {config.workflow.name}")
print(f"Type: {config.workflow.type}")
print(f"Storage: {config.storage}")
```

## Configuration Classes

### WorkflowConfig

Pydantic model for workflow configuration.

**Properties:**
```python
class WorkflowConfig(BaseModel):
    workflow: Dict[str, Any]       # Workflow metadata
    input: Dict[str, Any]          # Input configuration
    phases: Union[Dict, List]      # Phase configuration
    storage: Dict[str, Any]        # Storage configuration
    llm: Dict[str, Any]            # LLM configuration
    tools: Dict[str, Any]          # Tools configuration
    execution: Dict[str, Any]      # Execution settings
    output: Dict[str, Any]         # Output configuration
```

**Usage:**
```python
from deep_research_engine import WorkflowConfig

# Create from dict
config = WorkflowConfig(
    workflow={"name": "test", "type": "research"},
    input={"type": "query", "source": "test"},
    phases=[{"name": "boss", "node": "BossNode"}],
    storage={"memory": {"enabled": True}}
)

# Validate config
try:
    config = WorkflowConfig(**config_dict)
except ValidationError as e:
    print(f"Invalid config: {e}")
```

## State Classes

### EngineState

TypedDict for workflow state.

**Structure:**
```python
from typing import TypedDict, List, Dict, Any, Optional

class EngineState(TypedDict, total=False):
    # Workflow identification
    workflow_id: str
    workflow_name: str
    workflow_type: str
    
    # Execution tracking
    current_phase: str
    phase_history: List[str]
    iteration_count: int
    max_iterations: int
    
    # Input/Output
    input_data: Dict[str, Any]
    plan: List[str]
    plan_metadata: Dict[str, Any]
    synthesis_output: str
    
    # Results
    worker_results: List[Dict[str, Any]]
    worker_errors: List[Dict[str, Any]]
    critique: Dict[str, Any]
    critique_history: List[Dict[str, Any]]
    
    # Errors
    errors: List[Dict[str, Any]]
    fatal_error: str
    
    # Metadata
    session_id: str
    run_id: str
    context: Dict[str, Any]
    phase_metadata: Dict[str, Any]
    synthesis_metadata: Dict[str, Any]
    
    # Flags
    critic_passed: bool
```

**Usage:**
```python
from deep_research_engine import EngineState

# Access state properties
state: EngineState = {...}

print(f"Current phase: {state['current_phase']}")
print(f"Worker results: {len(state['worker_results'])}")
print(f"Plan: {state['plan']}")
```

### create_initial_state

Factory function for creating initial state.

**Signature:**
```python
def create_initial_state(
    workflow_name: str,
    workflow_type: str,
    input_data: Dict[str, Any],
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    max_iterations: int = 3,
    context: Optional[Dict[str, Any]] = None,
) -> EngineState:
    """Create initial engine state.
    
    Args:
        workflow_name: Name of workflow
        workflow_type: Type of workflow
        input_data: Workflow input data
        session_id: Optional session ID
        run_id: Optional run ID
        max_iterations: Maximum critic iterations
        context: Optional custom context
        
    Returns:
        EngineState with initialized values
    """
```

**Usage:**
```python
from deep_research_engine import create_initial_state

state = create_initial_state(
    workflow_name="rag_research",
    workflow_type="research",
    input_data={"query": "Research query"},
    session_id="session-123",
    max_iterations=5,
    context={"custom": "value"}
)
```

### WorkerResult

Standardized worker output contract.

**Structure:**
```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class WorkerResult:
    worker_id: str                      # Worker identifier
    worker_type: str                    # Worker type
    task: Any                           # Executed task
    success: bool                       # Success status
    output: Optional[Any] = None        # Output data
    output_type: str = "generic"        # Output type
    confidence: float = 0.0             # Confidence score
    timing: Optional[Dict] = None       # Timing information
    sources: Optional[List[Dict]] = None  # Sources
    source_count: int = 0               # Source count
    retry_count: int = 0                # Retry count
    error: Optional[str] = None         # Error message
    error_type: Optional[str] = None    # Error type
```

**Usage:**
```python
from deep_research_engine import WorkerResult

# Create result
result = WorkerResult(
    worker_id="worker_1",
    worker_type="web_researcher",
    task="Research ML",
    success=True,
    output={"findings": [...]},
    confidence=0.85,
    sources=[{"url": "http://example.com"}]
)

# Access properties
print(f"Success: {result.success}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.source_count}")
```

### CriticResult

Standardized critic output contract.

**Structure:**
```python
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class CriticResult:
    passed: bool                        # Pass/fail decision
    score: float                        # Overall score (0-1)
    criteria_scores: Dict[str, float]   # Individual criteria scores
    issues: List[str]                   # Identified issues
    suggestions: List[str]              # Improvement suggestions
    recommendation: str                 # pass/revise/terminate
```

## Worker Classes

### BaseWorker

Abstract base class for all workers.

**Signature:**
```python
from abc import ABC, abstractmethod

class BaseWorker(ABC):
    """Abstract base class for workers."""
    
    worker_type: str          # Worker type identifier
    description: str          # Worker description
    
    def __init__(self, config: WorkerConfig):
        """Initialize worker with configuration."""
        self.config = config
    
    @abstractmethod
    async def execute(
        self,
        state: EngineState,
        config: WorkerConfig
    ) -> WorkerResult:
        """Execute worker task.
        
        Args:
            state: Current engine state
            config: Worker configuration
            
        Returns:
            WorkerResult with success status and output
        """
        pass
    
    def _start_timing(self) -> None:
        """Start timing execution."""
        pass
    
    def _create_result(
        self,
        success: bool,
        output: Any = None,
        **kwargs
    ) -> WorkerResult:
        """Create standardized result."""
        pass
```

**Usage:**
```python
from deep_research_engine import BaseWorker, WorkerConfig, WorkerRegistry

@WorkerRegistry.register("my_worker")
class MyWorker(BaseWorker):
    worker_type = "my_worker"
    description = "My custom worker"
    
    async def execute(self, state, config):
        self._start_timing()
        
        # Implementation
        result = await self.do_work(config.task)
        
        return self._create_result(
            success=True,
            output=result,
            confidence=0.9
        )
```

### WorkerConfig

Worker configuration dataclass.

**Structure:**
```python
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class WorkerConfig:
    worker_id: str
    worker_type: str
    task: Any
    tools: List[str]
    max_retries: int = 3
    timeout: int = 300
    enable_critic: bool = True
    critic_threshold: float = 0.7
    custom: Dict[str, Any] = None
```

### WorkerRegistry

Registry for worker types.

**Methods:**
```python
class WorkerRegistry:
    @staticmethod
    def register(name: str):
        """Register worker class with name."""
        pass
    
    @staticmethod
    def get(name: str) -> Type[BaseWorker]:
        """Get worker class by name."""
        pass
    
    @staticmethod
    def is_registered(name: str) -> bool:
        """Check if worker is registered."""
        pass
    
    @staticmethod
    def list_all() -> List[Tuple[str, Type[BaseWorker]]]:
        """List all registered workers."""
        pass
```

**Usage:**
```python
from deep_research_engine import WorkerRegistry

# Register worker
@WorkerRegistry.register("custom")
class CustomWorker(BaseWorker):
    pass

# Get worker class
WorkerClass = WorkerRegistry.get("custom")

# Check registration
if WorkerRegistry.is_registered("custom"):
    print("Custom worker is registered")

# List all workers
for name, cls in WorkerRegistry.list_all():
    print(f"{name}: {cls.description}")
```

## Storage Classes

### StorageBackend

Abstract base class for storage backends.

**Signature:**
```python
class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    backend_name: str
    description: str
    
    async def initialize(self) -> None:
        """Initialize backend."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    async def health_check(self) -> bool:
        """Check backend health."""
        pass
    
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default"
    ) -> StorageResult:
        """Store data."""
        pass
    
    async def retrieve(
        self,
        id: str,
        namespace: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve data."""
        pass
    
    async def delete(
        self,
        id: str,
        namespace: str = "default"
    ) -> bool:
        """Delete data."""
        pass
    
    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List items."""
        pass
    
    async def query(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query data."""
        pass
```

### StorageManager

Manages multiple storage backends.

**Methods:**
```python
class StorageManager:
    async def initialize(
        self,
        backend_name: str,
        config: Dict[str, Any]
    ) -> None:
        """Initialize storage backend."""
        pass
    
    def get_backend(
        self,
        name: str
    ) -> Optional[StorageBackend]:
        """Get backend by name."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup all backends."""
        pass
```

**Usage:**
```python
from deep_research_engine import StorageManager

storage = StorageManager()

# Initialize backends
await storage.initialize("postgres", config)
await storage.initialize("memory", {})

# Get backend
backend = storage.get_backend("postgres")

# Use backend
result = await backend.store(data, namespace="research")

# Cleanup
await storage.cleanup()
```

### StorageResult

Standardized storage result.

**Structure:**
```python
@dataclass
class StorageResult:
    success: bool
    id: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None
```

### StorageRegistry

Registry for storage backends.

**Usage:**
```python
from deep_research_engine import StorageRegistry, StorageBackend

@StorageRegistry.register("custom")
class CustomStorage(StorageBackend):
    backend_name = "custom"
    # Implementation
```

## Utility Functions

### Config Utilities

```python
# Expand environment variables
from deep_research_engine import expand_env_vars

config = expand_env_vars("${VAR-default}")

# Merge configs
from deep_research_engine import merge_configs

merged = merge_configs(base_config, override_config)

# Render Jinja2 template
from deep_research_engine import render_prompt_template

template = "Hello, {{ name }}!"
result = render_prompt_template(template, name="World")
```

### Logging Utilities

```python
from deep_research_engine.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging(level="info")

# Get logger
logger = get_logger("my_module")
logger.info("Message", extra_context="value")
```

## CLI Commands

```bash
# Research
uv run deep-research research "query" --config rag.yaml

# PDF summarization
uv run deep-research pdf /path/to.pdf --config pdf_summarizer.yaml

# Telegram ingestion
uv run deep-research ingest /path/to/export --config telegram_ingest.yaml

# Web scraping
uv run deep-research scrape https://example.com --config skill_generator.yaml

# With output file
uv run deep-research research "query" -o output.md

# Without streaming
uv run deep-research research "query" --no-stream
```

## Error Handling

```python
from deep_research_engine import run_workflow
from deep_research_engine.workers import WorkerError
from deep_research_engine.storage import StorageError

try:
    async for event in run_workflow("query", config):
        if event["type"] == "error":
            print(f"Error: {event['error']}")
            
except WorkerError as e:
    print(f"Worker error: {e}")
    
except StorageError as e:
    print(f"Storage error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Type Aliases

```python
from typing import Literal

PhaseName = Literal["boss", "workers", "critic", "synthesis"]
WorkflowType = Literal["research", "pdf_summarization", "skill_generation", "telegram_ingest"]
WorkerType = Literal["web_researcher", "chunk_processor", "scraper", "multimodal", "synthesis"]
OutputFormat = Literal["research_report", "markdown", "structured_json", "summary"]
```

## Reference

- [Workflow Configs](../workflow-configs/)
- [Workers Guide](../workers/)
- [Storage Backends](../storage/)
- [Examples](../examples/)
