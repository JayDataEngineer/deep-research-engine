# Deep Research Engine

**Config-driven multi-agent workflow engine for AI ingestion tasks.**

Unified engine for:
- Multi-agent research (RAG + web)
- PDF summarization
- Telegram export processing
- Documentation scraping (llms.txt)

## Quick Start

```bash
# Install
pip install deep-research-engine

# Setup LangFuse (prompt management)
cd ../../shared-docker-infra
make up-langfuse
# Get keys from http://langfuse.local

# Configure environment
cp .env.example .env
# Edit .env with your LangFuse keys

# Upload prompts to LangFuse
uv run python scripts/upload_prompts_to_langfuse.py

# Run a workflow
deep-research research "quantum computing" --config rag.yaml

# Run PDF summarizer
deep-research pdf ./book.pdf --config pdf_summarizer.yaml
```

## Architecture

```
Config (YAML) → Engine → Workflow Graph
                     ↓
        Boss → Workers → Critic → Synthesis
```

**One graph, many configurations.** All workflows use the same `Boss → Workers → Critic → Synthesis` pipeline. The YAML config determines:
- What workers spawn (web_researcher, chunk_processor, multimodal, scraper)
- What storage backends are used (Postgres, Neo4j, MongoDB, MinIO, Memory)
- What the output format is (markdown, JSON, graph)

## Documentation

| Section | Description |
|---------|-------------|
| **[Getting Started](docs/getting-started/)** | Installation and quick start |
| **[LangFuse Prompts](docs/langfuse-prompt-management.md)** | Prompt management setup |
| **[Workflow Configs](docs/workflow-configs/)** | YAML configuration reference |
| **[Workers](docs/workers/)** | Worker types and how to create custom ones |
| **[Storage](docs/storage/)** | Pluggable storage backends |
| **[API Reference](docs/api-reference/)** | Python API reference |
| **[Examples](docs/examples/)** | Example workflows and configs |

## Infrastructure Requirements

This engine integrates with your existing infrastructure:

```bash
# Required: Shared Docker Infrastructure
cd ~/Documents/programs/shared-docker-infra

# Start core services
make core-up

# Start databases (optional - depends on workflow)
make db-up

# Verify
curl http://litellm.local/v1/models
```

See [shared-docker-infra/docs/](../../shared-docker-infra/docs/) for full infrastructure documentation.

### Service Dependencies by Workflow

| Workflow | LiteLLM | Postgres | Neo4j | MongoDB | MinIO |
|----------|---------|----------|-------|---------|-------|
| RAG Research | ✓ | ✓ | ✓ | - | - |
| PDF Summarizer | ✓ | - | - | - | - |
| Telegram Ingest | ✓ | ✓ | ✓ | ✓ | ✓ |
| Skill Generator | ✓ | - | - | - | - |

## Python API

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

## FastAPI Integration

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

## Project Structure

```
deep-research-engine/
├── src/deep_research_engine/
│   ├── engine.py              # Main entry point
│   ├── config.py              # Config loading
│   ├── state/                 # EngineState, WorkerResult
│   ├── graph/                 # Boss, Workers, Critic, Synthesis nodes
│   ├── workers/               # BaseWorker ABC + implementations
│   ├── storage/               # StorageBackend ABC + backends
│   ├── llm/                   # LLM factory
│   ├── tools/                 # Tool registry
│   └── critic/                # Critic implementations
├── configs/                   # Example workflow configs
├── docs/                      # Documentation
└── tests/                     # Tests
```

## Related Projects

This engine extracts and unifies:
- [unified-llm-interface](../text/unified-llm-interface/) - Deep research mode
- [pdf-summarizer](../text/pdf-summarizer/) - PDF processing
- [make-llmstxt](../text/make-llmstxt/) - Skill generation
- [tracer](../operations/tracer/) - Telegram ingestion

## License

MIT License
