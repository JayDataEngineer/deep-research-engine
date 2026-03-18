# AI Reference Guide - Deep Research Engine

**For**: 4 AIs working in parallel on implementation
**Project**: `/home/ubuntu/Documents/programs/deep-research-engine/`

---

## Project Overview

You are implementing a **config-driven multi-agent workflow engine** that unifies 4 existing projects:

1. **unified-llm-interface** - Deep research mode
2. **pdf-summarizer** - PDF processing
3. **make-llmstxt** - Skill generation
4. **tracer** - Telegram ingestion

**Key Insight**: All 4 projects use the same `Boss → Workers → Critic → Synthesis` pattern. The engine makes this configurable via YAML.

---

## Architecture

```
Config (YAML) → Engine → LangGraph Workflow
                     ↓
        Boss → Workers → Critic → Synthesis
```

### Core Design Decisions

1. **Generic State** - 18 fields max (not 50+)
   - Workflow-specific data in `context` dict
   - See: `src/deep_research_engine/state/types.py`

2. **Pluggable Storage** - Optional backends
   - PDF: Just Memory
   - Telegram: Postgres + Neo4j + MongoDB + MinIO
   - See: `src/deep_research_engine/storage/`

3. **Worker ABC** - Uniform interface
   - All workers implement `execute(state, config) -> WorkerResult`
   - See: `src/deep_research_engine/workers/base.py`

4. **Config-Driven** - New workflow = new YAML
   - See: `configs/*.yaml`

---

## Reference Materials

### Existing Implementations (Read These)

#### Deep Research (unified-llm-interface)
```
../../text/unified-llm-interface/backend/src/extensions/content/deep_research/
├── config.yaml              ← Main config (900 lines)
├── graph/
│   ├── graph_factory.py     ← LangGraph builder
│   ├── state.py             ← GlobalState (50+ fields - DON'T copy)
│   └── nodes/
│       └── unified/         ← Boss/Critic/Synthesis
└── agents/
    └── flash_explorer.py    ← Flash Explorer (1384 lines)
```

#### PDF Summarizer
```
../../text/pdf-summarizer/src/pdf_summarizer/
├── graph.py                 ← Linear pipeline
├── state.py                 ← PDFSummarizerState
└── nodes/
    ├── parser.py            ← pymupdf4llm
    ├── filter.py            ← Filler removal
    ├── dedupe.py            ← Semantic clustering
    └── summarize.py         ← Hierarchical summary
```

#### make-llmstxt
```
../../text/make-llmstxt/src/make_llmstxt/
├── generators/
│   ├── base_agent.py        ← DeepAgentGenerator
│   └── critic.py            ← Critic with structured output
└── scrapers/                ← MCP scraping
```

#### tracer
```
../../operations/tracer/src/tracer/
├── graph/
│   ├── ingestion/           ← Stage A: Parse → Process → Critic → Store
│   └── inference/           ← Stage B: Query → Decompose → Fan-out → Synthesize
└── tools/                   ← ASR, VLM, face recognition
```

### Infrastructure Documentation

```
../../shared-docker-infra/docs/
├── services/                ← LiteLLM, Postgres, Neo4j, MongoDB, MinIO
├── integration/             ← How to connect
└── architecture/            ← Network topology, security
```

### Design Documents (This Project)

```
design/
├── UNIFIED_CONFIG_SCHEMA.md      ← YAML schema
├── ENGINE_STATE_DESIGN.md        ← 18-field state
├── BASE_WORKER_DESIGN.md         ← Worker ABC
├── PLUGGABLE_STORAGE_DESIGN.md   ← Storage backends
└── WORKFLOW_MIGRATION_MAP.md     ← Map 4 projects to configs
```

---

## Implementation Tasks (Parallel)

### AI 1: Graph Execution

**Goal**: Implement LangGraph workflow with Boss → Workers → Critic → Synthesis

**Files to Create**:
```
src/deep_research_engine/graph/
├── __init__.py
├── builder.py              # build_workflow_graph(config)
├── nodes/
│   ├── __init__.py
│   ├── boss.py             # BossNode (planning)
│   ├── workers.py          # WorkerNode (parallel execution)
│   ├── critic.py           # CriticNode (quality gate)
│   └── synthesis.py        # SynthesisNode (final output)
└── edges.py                # Conditional routing
```

**Reference**:
- `../../text/unified-llm-interface/backend/src/extensions/content/deep_research/graph/graph_factory.py`
- `../../text/unified-llm-interface/backend/src/extensions/content/deep_research/graph/nodes/unified/`

**Key Patterns**:
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

def build_workflow_graph(config: WorkflowConfig):
    graph = StateGraph(EngineState)
    
    # Add nodes
    graph.add_node("boss", boss_node)
    graph.add_node("workers", workers_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesis", synthesis_node)
    
    # Add edges
    graph.add_edge(START, "boss")
    graph.add_edge("boss", "workers")
    graph.add_edge("workers", "critic")
    graph.add_edge("synthesis", END)
    
    # Conditional routing
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "pass": "synthesis",
            "revise": "workers",
        }
    )
    
    return graph.compile()
```

---

### AI 2: Worker Implementations

**Goal**: Implement worker types for all 4 workflows

**Files to Create**:
```
src/deep_research_engine/workers/implementations/
├── __init__.py
├── web_researcher.py       # Flash Explorer wrapper
├── chunk_processor.py      # PDF chunk processing
├── dedupe_worker.py        # Semantic deduplication
├── multimodal.py           # ASR/VLM/face processing
├── scraper.py              # Web scraping (make-llmstxt)
└── subtypes/
    ├── asr_worker.py
    ├── vlm_worker.py
    ├── face_worker.py
    └── text_worker.py
```

**Reference**:
- `../../text/unified-llm-interface/backend/src/extensions/content/deep_research/agents/flash_explorer.py`
- `../../text/pdf-summarizer/src/pdf_summarizer/nodes/filter.py`
- `../../text/make-llmstxt/src/make_llmstxt/generators/base_agent.py`

**Template**:
```python
from ..base import BaseWorker, WorkerConfig, WorkerRegistry
from ...state import EngineState, WorkerResult

@WorkerRegistry.register("web_researcher")
class WebResearcherWorker(BaseWorker):
    worker_type = "web_researcher"
    description = "Autonomous web research agent"
    required_tools = ["web_search", "web_get"]
    
    async def execute(self, state, config) -> WorkerResult:
        self._start_timing()
        
        # Your implementation here
        # Use Flash Explorer pattern from unified-llm-interface
        
        return self._create_result(
            success=True,
            output=findings,
            output_type="research_findings",
            sources=sources,
        )
```

---

### AI 3: Storage Backends

**Goal**: Implement pluggable storage backends

**Files to Create**:
```
src/deep_research_engine/storage/backends/
├── postgres.py             # PostgreSQL + pgvector
├── mongodb.py              # Media files
├── neo4j.py                # Knowledge graph
└── minio.py                # S3-compatible object storage
```

**Reference**:
- `../../text/unified-llm-interface/backend/src/shared/tools/registry.py`
- `../../operations/tracer/src/tracer/store/`
- `design/PLUGGABLE_STORAGE_DESIGN.md`

**Template**:
```python
from ..base import StorageBackend, StorageResult
from ..registry import StorageRegistry

@StorageRegistry.register("postgres")
class PostgresStorage(StorageBackend):
    backend_name = "postgres"
    description = "PostgreSQL + pgvector storage"
    
    async def initialize(self) -> None:
        import asyncpg
        from pgvector.asyncpg import register_vector
        
        self.pool = await asyncpg.create_pool(
            self.config.get("connection"),
            min_size=5,
            max_size=20,
        )
        
        async with self.pool.acquire() as conn:
            await register_vector(conn)
        
        await self._create_tables()
        self.initialized = True
    
    async def store(self, data, namespace="default") -> StorageResult:
        # Implementation
        pass
    
    async def retrieve(self, id, namespace="default"):
        # Implementation
        pass
    
    async def query(self, query, namespace="default", limit=10):
        # Semantic search with pgvector
        pass
```

---

### AI 4: Tests + Documentation

**Goal**: Comprehensive tests and documentation

**Files to Create**:
```
tests/
├── unit/
│   ├── test_state.py
│   ├── test_worker_registry.py
│   ├── test_storage_registry.py
│   └── test_config_loader.py
├── integration/
│   ├── test_boss_node.py
│   ├── test_worker_node.py
│   └── test_critic_node.py
└── workflows/
    ├── test_rag_workflow.py
    ├── test_pdf_workflow.py
    └── test_telegram_workflow.py

docs/
├── workflow-configs/
│   ├── index.md
│   ├── schema.md
│   └── examples.md
├── workers/
│   ├── index.md
│   └── custom.md
└── storage/
    ├── index.md
    └── backends.md
```

**Test Template**:
```python
import pytest
from deep_research_engine import load_config, WorkerRegistry

def test_load_rag_config():
    config = load_config("configs/rag.yaml")
    assert config.workflow["name"] == "rag"
    assert config.workflow["type"] == "research"

def test_worker_registry():
    from deep_research_engine.workers.implementations import WebResearcherWorker
    
    WorkerRegistry.register("test_worker")(WebResearcherWorker)
    klass = WorkerRegistry.get("test_worker")
    assert klass.worker_type == "test_worker"
```

---

## Coding Standards

### Imports
```python
# Standard library first
from typing import Dict, Any, Optional
from pathlib import Path

# Third-party
from langgraph.graph import StateGraph
from pydantic import BaseModel

# Local
from ..state import EngineState
from .base import BaseWorker
```

### Type Hints
```python
# Always use type hints
async def execute(
    self,
    state: EngineState,
    config: WorkerConfig,
) -> WorkerResult:
    pass
```

### Logging
```python
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Use structured logging
logger.info(
    "Worker started",
    worker_id=self.worker_id,
    task=self.task[:50],
)
```

### Error Handling
```python
try:
    result = await some_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return self._create_result(
        success=False,
        error=str(e),
    )
```

---

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (LLM, databases)

### Integration Tests
- Test graph nodes together
- Use real infrastructure (LiteLLM, Postgres)

### Workflow Tests
- End-to-end workflow execution
- Verify output format

---

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/graph-execution

# Commit frequently
git add src/deep_research_engine/graph/builder.py
git commit -m "Implement graph builder with retry policies"

# Push and create PR
git push origin feature/graph-execution
```

---

## Questions?

1. Check `design/` for architecture decisions
2. Check existing implementations in `../../text/` and `../../operations/`
3. Check `../../shared-docker-infra/docs/` for infrastructure

---

## Quick Reference

### Run Tests
```bash
cd ~/Documents/programs/deep-research-engine
uv run pytest tests/
```

### Run Workflow
```bash
deep-research research "query" --config rag.yaml
```

### Check Infrastructure
```bash
cd ~/Documents/programs/shared-docker-infra
make status
```
