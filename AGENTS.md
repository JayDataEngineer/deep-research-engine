# AGENTS.md - AI Development Guide

**Project**: Deep Research Engine  
**Location**: `/home/ubuntu/Documents/programs/deep-research-engine/`

---

## Quick Start for AI Agents

### 1. Read These First
1. `AI_REFERENCE.md` - Main implementation guide
2. `IMPLEMENTATION_STATUS.md` - What's done vs TODO
3. `pyproject.toml` - Dependencies and configuration

### 2. Project Structure
```
deep-research-engine/
├── src/deep_research_engine/    # Main package
├── configs/                      # Workflow configs
├── docs/                         # Documentation
├── design/                       # Architecture docs
└── tests/                        # Tests (TODO)
```

### 3. Development Commands

```bash
# Setup
cd /home/ubuntu/Documents/programs/deep-research-engine
uv sync --group dev

# Run CLI
uv run deep-research --help
uv run deep-research research "query" --config rag.yaml

# Testing
uv run pytest tests/
uv run pytest tests/unit/ -v

# Linting
uv run ruff check src/
uv run ruff format src/
uv run mypy src/

# All checks
uv run pre-commit run --all-files
```

---

## Architecture Overview

### Core Pattern
```
Config (YAML) → Engine → Boss → Workers → Critic → Synthesis
```

### Key Interfaces

**State** (`src/deep_research_engine/state/types.py`):
```python
class EngineState(TypedDict, total=False):
    """18 fields max - workflow-specific data in context dict"""
    workflow_id: str
    workflow_name: str
    current_phase: str
    plan: List[str]
    worker_results: Annotated[List[Dict], operator.add]
    context: Dict[str, Any]  # <-- Workflow-specific data
    # ... (18 fields total)
```

**Worker** (`src/deep_research_engine/workers/base.py`):
```python
class BaseWorker(ABC):
    @abstractmethod
    async def execute(self, state, config) -> WorkerResult:
        """All workers implement this"""
        pass
```

**Storage** (`src/deep_research_engine/storage/base.py`):
```python
class StorageBackend(ABC):
    async def store(self, data, namespace) -> StorageResult: ...
    async def retrieve(self, id, namespace) -> Optional[Dict]: ...
    async def query(self, query, namespace, limit) -> List[Dict]: ...
```

---

## Implementation Tasks

### AI 1: Graph Execution
**Files**: `src/deep_research_engine/graph/`

```python
# Template
from langgraph.graph import StateGraph, START, END
from ..state import EngineState

def build_workflow_graph(config):
    graph = StateGraph(EngineState)
    
    graph.add_node("boss", boss_node)
    graph.add_node("workers", workers_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesis", synthesis_node)
    
    graph.add_edge(START, "boss")
    graph.add_edge("boss", "workers")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"pass": "synthesis", "revise": "workers"}
    )
    
    return graph.compile()
```

**Reference**: `../../text/unified-llm-interface/backend/src/extensions/content/deep_research/graph/`

---

### AI 2: Worker Implementations
**Files**: `src/deep_research_engine/workers/implementations/`

```python
# Template
from ..base import BaseWorker, WorkerRegistry

@WorkerRegistry.register("web_researcher")
class WebResearcherWorker(BaseWorker):
    worker_type = "web_researcher"
    description = "Autonomous web research agent"
    
    async def execute(self, state, config) -> WorkerResult:
        self._start_timing()
        # Implementation
        return self._create_result(success=True, output=findings)
```

**Workers to implement**:
- `web_researcher.py` - Flash Explorer wrapper
- `chunk_processor.py` - PDF chunk processing
- `multimodal.py` - ASR/VLM/face processing
- `scraper.py` - Web scraping (make-llmstxt)

**Reference**: `../../text/unified-llm-interface/.../agents/flash_explorer.py`

---

### AI 3: Storage Backends
**Files**: `src/deep_research_engine/storage/backends/`

```python
# Template
from ..base import StorageBackend, StorageResult
from ..registry import StorageRegistry

@StorageRegistry.register("postgres")
class PostgresStorage(StorageBackend):
    backend_name = "postgres"
    
    async def initialize(self):
        import asyncpg
        self.pool = await asyncpg.create_pool(...)
        
    async def store(self, data, namespace) -> StorageResult:
        # Implementation
```

**Backends to implement**:
- `postgres.py` - PostgreSQL + pgvector
- `mongodb.py` - Media files
- `neo4j.py` - Knowledge graph
- `minio.py` - S3-compatible object storage

**Reference**: `../../operations/tracer/src/tracer/store/`

---

### AI 4: Tests + Documentation
**Files**: `tests/`, `docs/`

**Test Template**:
```python
import pytest
from deep_research_engine import load_config

def test_load_rag_config():
    config = load_config("configs/rag.yaml")
    assert config.workflow["name"] == "rag"
```

**Docs to write**:
- `docs/workflow-configs/` - Config reference
- `docs/workers/` - Worker guide
- `docs/storage/` - Storage backends
- `docs/examples/` - Example workflows

---

## Coding Standards

### Imports
```python
# Standard library
from typing import Dict, Any, Optional

# Third-party
from langgraph.graph import StateGraph

# Local
from ..state import EngineState
```

### Type Hints
Always use type hints. Required by mypy.

### Logging
```python
from ..utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Message", extra_context=value)
```

### Error Handling
```python
try:
    result = await operation()
except Exception as e:
    logger.error(f"Failed: {e}")
    return self._create_result(success=False, error=str(e))
```

---

## Reference Projects

| Project | Location | Purpose |
|---------|----------|---------|
| Deep Research | `../../text/unified-llm-interface/backend/.../deep_research/` | Boss/Critic/Synthesis |
| PDF Summarizer | `../../text/pdf-summarizer/src/pdf_summarizer/` | Chunk processing |
| make-llmstxt | `../../text/make-llmstxt/src/make_llmstxt/` | Scraping + generation |
| tracer | `../../operations/tracer/src/tracer/` | Multimodal processing |

---

## Infrastructure

Start shared infrastructure:
```bash
cd ../../shared-docker-infra
make core-up
make db-up
```

Services:
- LiteLLM: `http://litellm.local/v1`
- Postgres: `localhost:5433`
- Neo4j: `localhost:7687`
- MongoDB: `localhost:27017`
- MinIO: `localhost:9000`

---

## Questions?

1. Check `design/` for architecture decisions
2. Check `AI_REFERENCE.md` for detailed guides
3. Check existing implementations in `../../text/` and `../../operations/`
