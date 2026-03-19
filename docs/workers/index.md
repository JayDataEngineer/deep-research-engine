# Workers Guide

Complete guide to using, implementing, and customizing workers in the Deep Research Engine.

## Overview

Workers are autonomous agents that execute specific tasks within a workflow. Each worker:

- Receives a task from the boss node
- Executes autonomously using tools and LLM
- Returns structured results
- Supports retry and error handling

## Worker Architecture

```
┌─────────────┐
│ Boss Node   │ → Creates tasks
└─────────────┘
       ↓
┌─────────────┐
│ Worker Queue│ → Parallel execution
└─────────────┘
       ↓
┌─────────────┐
│ Worker N    │ → Executes task
└─────────────┘
       ↓
┌─────────────┐
│ Critic Node │ → Evaluates results
└─────────────┘
```

## Built-in Workers

### WebResearcherWorker

Autonomous web research agent using Flash Explorer.

**Configuration:**
```yaml
phases:
  - name: workers
    node: WorkerNode
    config:
      worker_type: web_researcher
      parallel: true
      max_concurrent: 10
```

**Features:**
- Web search integration
- Content extraction
- Source tracking
- Confidence scoring

**Output:**
```python
{
    "worker_id": "web_researcher_0",
    "success": True,
    "output": {
        "findings": [...],
        "sources": [...]
    },
    "confidence": 0.85,
    "sources": [...],
    "source_count": 5
}
```

### ChunkProcessorWorker

Processes PDF chunks for summarization.

**Configuration:**
```yaml
phases:
  - name: workers
    node: WorkerNode
    config:
      worker_type: chunk_processor
      parallel: true
      max_concurrent: 5
```

**Features:**
- Text chunk processing
- Summary generation
- Key point extraction
- Metadata preservation

**Output:**
```python
{
    "worker_id": "chunk_processor_0",
    "success": True,
    "output": {
        "summary": "...",
        "key_points": [...],
        "chunk_id": "chunk_1"
    },
    "confidence": 0.9
}
```

### ScraperWorker

Web scraping using make-llmstxt patterns.

**Configuration:**
```yaml
phases:
  - name: workers
    node: WorkerNode
    config:
      worker_type: scraper
      parallel: true
```

**Features:**
- URL content extraction
- Documentation scraping
- Structured data extraction
- robots.txt compliance

**Output:**
```python
{
    "worker_id": "scraper_0",
    "success": True,
    "output": {
        "content": "...",
        "metadata": {
            "url": "...",
            "title": "..."
        }
    },
    "confidence": 0.88
}
```

### MultimodalWorker

Processes multimodal content (ASR, VLM, face recognition).

**Configuration:**
```yaml
phases:
  - name: workers
    node: WorkerNode
    config:
      worker_type: multimodal
      parallel: true
```

**Subtypes:**
- `asr` - Speech-to-text
- `vlm` - Vision-language model
- `face` - Face recognition
- `text` - Text processing

**Output:**
```python
{
    "worker_id": "multimodal_0",
    "success": True,
    "output": {
        "transcripts": [...],
        "summaries": [...],
        "entities": [...]
    },
    "confidence": 0.85
}
```

### SynthesisWorker

Synthesizes research findings into formatted output.

**Configuration:**
```yaml
phases:
  - name: synthesis
    node: SynthesisNode
    config:
      worker_type: synthesis
      format: research_report
```

**Features:**
- Report generation
- Citation formatting
- Section organization
- Metadata inclusion

**Output Formats:**
- `research_report` - Full report
- `markdown` - Markdown document
- `structured_json` - JSON structure
- `summary` - Brief summary

## Implementing Custom Workers

### Base Class

All workers inherit from `BaseWorker`:

```python
from deep_research_engine.workers import BaseWorker, WorkerConfig, WorkerRegistry
from deep_research_engine.state import EngineState, WorkerResult

@WorkerRegistry.register("my_custom_worker")
class MyCustomWorker(BaseWorker):
    """Custom worker implementation."""
    
    worker_type = "my_custom_worker"
    description = "My custom worker for specialized tasks"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute worker task.
        
        Args:
            state: Current engine state
            config: Worker configuration
            
        Returns:
            WorkerResult with success status and output
        """
        self._start_timing()
        
        try:
            # Your implementation here
            result = await self.do_work(config.task)
            
            return self._create_result(
                success=True,
                output=result,
                output_type="custom",
                confidence=0.9,
                sources=self._sources
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
```

### WorkerConfig

Configuration passed to workers:

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class WorkerConfig:
    worker_id: str              # Unique worker identifier
    worker_type: str            # Worker type name
    task: Any                   # Task to execute
    tools: List[str]            # Available tools
    max_retries: int = 3        # Maximum retry attempts
    timeout: int = 300          # Timeout in seconds
    enable_critic: bool = True  # Enable critic evaluation
    critic_threshold: float = 0.7  # Quality threshold
    custom: Dict[str, Any] = None   # Custom configuration
```

### WorkerResult

Standardized result contract:

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
    sources: Optional[List[Dict]] = None  # Source references
    source_count: int = 0               # Number of sources
    retry_count: int = 0                # Retry count
    error: Optional[str] = None         # Error message
    error_type: Optional[str] = None    # Error type
```

### Helper Methods

BaseWorker provides helper methods:

```python
class BaseWorker(ABC):
    
    def _start_timing(self) -> None:
        """Start timing execution."""
        self._start_time = time.time()
    
    def _create_result(
        self,
        success: bool,
        output: Any = None,
        output_type: str = "generic",
        confidence: float = 0.0,
        sources: List[Dict] = None,
        error: str = None,
        error_type: str = None
    ) -> WorkerResult:
        """Create standardized result."""
        # Implementation
    
    async def _retry(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Retry execution."""
        # Implementation
```

### Example: Database Query Worker

```python
from deep_research_engine.workers import BaseWorker, WorkerConfig, WorkerRegistry
from deep_research_engine.state import EngineState, WorkerResult
import asyncpg

@WorkerRegistry.register("database_query")
class DatabaseQueryWorker(BaseWorker):
    """Worker for executing database queries."""
    
    worker_type = "database_query"
    description = "Executes SQL queries against PostgreSQL database"
    
    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.connection = None
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute database query."""
        self._start_timing()
        
        try:
            # Get connection from state or config
            db_url = config.custom.get("connection_url")
            
            # Execute query
            conn = await asyncpg.connect(db_url)
            results = await conn.fetch(config.task)
            await conn.close()
            
            return self._create_result(
                success=True,
                output={"rows": [dict(r) for r in results]},
                output_type="database_results",
                confidence=0.95,
                sources=[{"type": "database", "url": db_url}]
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
```

### Example: API Integration Worker

```python
from deep_research_engine.workers import BaseWorker, WorkerConfig, WorkerRegistry
from deep_research_engine.state import EngineState, WorkerResult
import httpx

@WorkerRegistry.register("api_call")
class APICallWorker(BaseWorker):
    """Worker for making API calls."""
    
    worker_type = "api_call"
    description = "Makes HTTP requests to external APIs"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute API call."""
        self._start_timing()
        
        try:
            url = config.custom.get("url")
            method = config.custom.get("method", "GET")
            headers = config.custom.get("headers", {})
            
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, headers=headers)
                response.raise_for_status()
                
                return self._create_result(
                    success=True,
                    output=response.json(),
                    output_type="api_response",
                    confidence=0.9,
                    sources=[{"type": "api", "url": url}]
                )
                
        except Exception as e:
            return self._create_result(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
```

## Worker Registry

Workers are registered using the `@WorkerRegistry.register` decorator:

```python
from deep_research_engine.workers import WorkerRegistry

# Register worker
@WorkerRegistry.register("my_worker")
class MyWorker(BaseWorker):
    worker_type = "my_worker"

# Retrieve worker class
WorkerClass = WorkerRegistry.get("my_worker")

# Check if registered
if WorkerRegistry.is_registered("my_worker"):
    print("Worker is registered")

# List all registered workers
for name, cls in WorkerRegistry.list_all():
    print(f"{name}: {cls.description}")
```

## Worker Execution Flow

1. **Boss Node** creates tasks and worker configs
2. **WorkerNode** spawns workers in parallel
3. **Workers** execute autonomously:
   - Initialize
   - Execute task
   - Handle errors
   - Return results
4. **Results** accumulated in state
5. **Critic Node** evaluates quality

## Error Handling

Workers should handle errors gracefully:

```python
async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
    self._start_timing()
    
    try:
        # Implementation
        result = await self.do_work()
        return self._create_result(success=True, output=result)
        
    except TimeoutError as e:
        return self._create_result(
            success=False,
            error=f"Timeout: {e}",
            error_type="TimeoutError"
        )
        
    except ConnectionError as e:
        return self._create_result(
            success=False,
            error=f"Connection failed: {e}",
            error_type="ConnectionError"
        )
        
    except Exception as e:
        return self._create_result(
            success=False,
            error=str(e),
            error_type=type(e).__name__
        )
```

## Testing Workers

### Unit Test Example

```python
import pytest
from deep_research_engine.workers import WorkerConfig
from deep_research_engine.state import create_initial_state

class TestMyWorker:
    
    @pytest.mark.asyncio
    async def test_worker_success(self):
        """Test worker successful execution."""
        config = WorkerConfig(
            worker_id="test_1",
            worker_type="my_worker",
            task="test task",
            tools=[]
        )
        
        worker = MyWorker(config)
        state = create_initial_state("test", "research", {"query": "test"})
        
        result = await worker.execute(state, config)
        
        assert result.success is True
        assert result.output is not None
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_worker_error_handling(self):
        """Test worker error handling."""
        config = WorkerConfig(
            worker_id="test_1",
            worker_type="my_worker",
            task="invalid task",
            tools=[]
        )
        
        worker = MyWorker(config)
        state = create_initial_state("test", "research", {"query": "test"})
        
        result = await worker.execute(state, config)
        
        assert result.success is False
        assert result.error is not None
```

## Best Practices

1. **Keep workers focused** - Single responsibility
2. **Handle errors gracefully** - Return structured errors
3. **Track sources** - Enable citation and verification
4. **Report confidence** - Help critic evaluate quality
5. **Respect timeouts** - Don't hang indefinitely
6. **Use retry wisely** - Not all errors are retryable
7. **Document thoroughly** - Explain worker behavior
8. **Test extensively** - Unit and integration tests

## Performance Considerations

### Parallel Execution

```yaml
# Enable parallel execution
phases:
  - name: workers
    node: WorkerNode
    config:
      parallel: true
      max_concurrent: 10  # Limit concurrency
```

### Memory Management

```python
async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
    # Process in chunks for large data
    async for chunk in self.process_large_data(config.task):
        # Yield results incrementally
        yield chunk
```

### Caching

```python
from functools import lru_cache

class CachedWorker(BaseWorker):
    @lru_cache(maxsize=100)
    async def _cached_operation(self, key: str):
        # Cached operation
        pass
```

## Debugging Workers

### Enable Debug Logging

```python
from deep_research_engine.utils.logging import get_logger

logger = get_logger("workers.my_worker")

async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
    logger.info("Starting execution", task=config.task)
    
    try:
        result = await self.do_work()
        logger.info("Execution complete", success=True)
        return self._create_result(success=True, output=result)
    except Exception as e:
        logger.error("Execution failed", error=str(e), exc_info=True)
        return self._create_result(success=False, error=str(e))
```

## Reference

- [Workflow Configs](../workflow-configs/)
- [Storage Backends](../storage/)
- [API Reference](../api-reference/)
- [Examples](../examples/)
