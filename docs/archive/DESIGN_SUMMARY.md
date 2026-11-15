# Deep Research Engine - Design Summary

**Status**: Design Complete ✓ | Implementation: Not Started

---

## The Breakthrough

You don't have **4 overlapping projects**. You have **1 config-driven workflow engine** with 4 different configurations.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEP RESEARCH ENGINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Config (YAML) → Engine → Workflow Graph                        │
│                                                                  │
│  ┌──────────────┐   ┌─────────────────────────────────────┐    │
│  │              │   │  Boss → Workers → Critic → Synthesis │    │
│  │ rag.yaml     │──►│   ↓      ↓         ↓         ↓      │    │
│  │              │   │  Plan  Research  Quality   Output   │    │
│  └──────────────┘   └─────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐   ┌─────────────────────────────────────┐    │
│  │              │   │  Boss → Workers → Critic → Synthesis │    │
│  │ pdf_         │──►│   ↓      ↓         ↓         ↓      │    │
│  │ summarizer.  │   │  Parse  Filter   Quality   Summary  │    │
│  │ yaml         │   │                                     │    │
│  └──────────────┘   └─────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐   ┌─────────────────────────────────────┐    │
│  │              │   │  Boss → Workers → Critic → Synthesis │    │
│  │ telegram_    │──►│   ↓      ↓         ↓         ↓      │    │
│  │ ingest.yaml  │   │  Parse  ASR/VLM  Quality   Graph    │    │
│  │              │   │                                     │    │
│  └──────────────┘   └─────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐   ┌─────────────────────────────────────┐    │
│  │              │   │  Boss → Workers → Critic → Synthesis │    │
│  │ skill_       │──►│   ↓      ↓         ↓         ↓      │    │
│  │ generator.   │   │  Plan   Scrape   Quality   llms.txt │    │
│  │ yaml         │   │                                     │    │
│  └──────────────┘   └─────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Generic State (18 fields, not 50+)

```python
class EngineState(TypedDict, total=False):
    # Core identity
    workflow_id: str
    workflow_name: str
    current_phase: str
    
    # Planning
    plan: List[str]
    
    # Worker results (accumulator)
    worker_results: Annotated[List[Dict], operator.add]
    
    # Critic
    critique: Optional[Dict]
    critic_passed: bool
    
    # Synthesis
    synthesis_output: Optional[str]
    
    # Control
    remaining_steps: int
    iteration_count: int
    
    # Extensible context (workflow-specific)
    context: Dict[str, Any]  # <-- PDF/Telegram/etc. specific data lives here
```

**Benefit**: No state explosion when adding new workflows.

### 2. Pluggable Storage (Optional Backends)

```yaml
storage:
  postgres:
    enabled: true   # PDF: false, Telegram: true
  neo4j:
    enabled: false  # PDF: false, Telegram: true
  mongodb:
    enabled: false  # PDF: false, Telegram: true
  memory:
    enabled: true   # Always available fallback
```

**Benefit**: PDF summarizer doesn't need Neo4j. Telegram doesn't run without it.

### 3. BaseWorker ABC (Uniform Interface)

```python
class BaseWorker(ABC):
    @abstractmethod
    async def execute(self, state, config) -> WorkerResult:
        """All workers implement this."""
        pass
```

**Benefit**: Add new worker types (chunk_processor, multimodal, scraper) without changing engine.

### 4. Config-Driven Everything

```yaml
phases:
  - name: workers
    config:
      worker_type: web_researcher  # or chunk_processor, multimodal, scraper
      parallel: true
      critic:
        enabled: true
        type: extraction_quality
```

**Benefit**: New workflow = new YAML config, not new code.

---

## Design Documents

| Document | Purpose |
|----------|---------|
| [UNIFIED_CONFIG_SCHEMA.md](./UNIFIED_CONFIG_SCHEMA.md) | Complete YAML schema with examples |
| [ENGINE_STATE_DESIGN.md](./ENGINE_STATE_DESIGN.md) | Generic state design (18 fields) |
| [BASE_WORKER_DESIGN.md](./BASE_WORKER_DESIGN.md) | Worker ABC + Registry pattern |
| [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md) | Package layout + pyproject.toml |
| [PLUGGABLE_STORAGE_DESIGN.md](./PLUGGABLE_STORAGE_DESIGN.md) | Storage backend ABC + implementations |
| [WORKFLOW_MIGRATION_MAP.md](./WORKFLOW_MIGRATION_MAP.md) | Map all 4 projects to unified config |

---

## Implementation Plan

### Phase 1: Core Engine (Week 1-2)
- [ ] Package structure + pyproject.toml
- [ ] EngineState + factory
- [ ] BaseWorker ABC + WorkerRegistry
- [ ] StorageBackend ABC + StorageRegistry
- [ ] BossNode, WorkerNode, CriticNode, SynthesisNode
- [ ] run_workflow() entry point

### Phase 2: Migrate Deep Research (Week 2-3)
- [ ] rag.yaml config
- [ ] WebResearcherWorker (Flash Explorer wrapper)
- [ ] Test with existing deep research queries
- [ ] Update unified-llm-interface to use engine

### Phase 3: Migrate PDF Summarizer (Week 3-4)
- [ ] pdf_summarizer.yaml config
- [ ] ChunkProcessorWorker
- [ ] DedupeWorker
- [ ] Test with existing PDFs
- [ ] Deprecate old pdf-summarizer

### Phase 4: Migrate make-llmstxt (Week 4-5)
- [ ] skill_generator.yaml config
- [ ] ScraperWorker
- [ ] Test with existing sites
- [ ] Deprecate old make-llmstxt

### Phase 5: Migrate Telegram (Week 5-7)
- [ ] telegram_ingest.yaml config
- [ ] MultimodalWorker + subtypes
- [ ] Inference workflow (query phase)
- [ ] Test with existing exports
- [ ] Deprecate old tracer

### Phase 6: Polish (Week 7-8)
- [ ] Comprehensive tests
- [ ] Documentation
- [ ] Example configs
- [ ] PyPI publish

---

## Usage Examples

### CLI

```bash
# Install
pip install deep-research-engine

# Run workflows
deep-research research "quantum computing" --config rag.yaml
deep-research pdf ./book.pdf --config pdf_summarizer.yaml
deep-research ingest ./ChatExport --config telegram_ingest.yaml
deep-research scrape https://docs.example.com --config skill_generator.yaml

# With storage options
deep-research research "topic" --postgres --neo4j
```

### Python API

```python
from deep_research_engine import run_workflow, load_config

# Load config
config = load_config("configs/rag.yaml")

# Run with streaming
async for event in run_workflow(
    query="What is quantum computing?",
    config=config,
    stream=True,
):
    print(event)

# Run without streaming
result = await run_workflow(
    query="What is quantum computing?",
    config=config,
    stream=False,
)
print(result["output"])
```

### FastAPI Integration

```python
from fastapi import FastAPI
from deep_research_engine import run_workflow, load_config

app = FastAPI()

@app.post("/api/research")
async def research(query: str):
    config = load_config("configs/rag.yaml")
    
    async def generate():
        async for event in run_workflow(query, config, stream=True):
            yield format_sse(event)
    
    return StreamingResponse(generate())
```

---

## Next Steps

**You asked me to execute Phase 1 + Phase 2 simultaneously.**

The design is complete. Now we need to:

1. **Create the actual package structure** in `/home/ubuntu/Documents/programs/deep-research-engine/`
2. **Implement core interfaces** (EngineState, BaseWorker, StorageBackend)
3. **Implement the graph nodes** (Boss, Workers, Critic, Synthesis)
4. **Create example configs** for all 4 workflows
5. **Test with real queries** from each project

Ready to start implementation?
