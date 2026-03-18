# CLAUDE.md - Deep Research Engine

## Project Overview

**Config-driven multi-agent workflow engine** unifying 4 AI ingestion projects:
- Deep research (RAG + web)
- PDF summarization
- Telegram export processing
- Documentation scraping (llms.txt)

**Key insight**: All 4 use `Boss → Workers → Critic → Synthesis` pattern.

---

## Quick Commands

```bash
# Setup
cd /home/ubuntu/Documents/programs/deep-research-engine
uv sync --group dev

# Run
uv run deep-research research "query" --config rag.yaml

# Test
uv run pytest tests/ -v

# Lint
uv run ruff check src/ && uv run ruff format src/
uv run mypy src/
```

---

## Architecture

```
Config (YAML) → Engine → LangGraph Workflow
                     ↓
        Boss → Workers → Critic → Synthesis
```

### Design Principles

1. **Generic State** - 18 fields (not 50+)
   - Workflow-specific data in `context` dict
2. **Pluggable Storage** - Optional backends
   - PDF: Memory only
   - Telegram: Postgres + Neo4j + MongoDB + MinIO
3. **Worker ABC** - Uniform interface
   - `execute(state, config) -> WorkerResult`
4. **Config-Driven** - New workflow = new YAML

---

## Core Files

| File | Purpose |
|------|---------|
| `src/deep_research_engine/state/types.py` | EngineState (18 fields) |
| `src/deep_research_engine/workers/base.py` | BaseWorker ABC |
| `src/deep_research_engine/storage/base.py` | StorageBackend ABC |
| `src/deep_research_engine/config.py` | YAML loader |
| `src/deep_research_engine/engine.py` | run_workflow() |
| `configs/*.yaml` | 4 workflow configs |

---

## Implementation TODO

### Priority 1: Graph Execution
- [ ] `graph/nodes/boss.py`
- [ ] `graph/nodes/workers.py`
- [ ] `graph/nodes/critic.py`
- [ ] `graph/nodes/synthesis.py`
- [ ] `graph/builder.py`

### Priority 2: Worker Implementations
- [ ] `workers/implementations/web_researcher.py`
- [ ] `workers/implementations/chunk_processor.py`
- [ ] `workers/implementations/multimodal.py`
- [ ] `workers/implementations/scraper.py`

### Priority 3: Storage Backends
- [ ] `storage/backends/postgres.py`
- [ ] `storage/backends/mongodb.py`
- [ ] `storage/backends/neo4j.py`
- [ ] `storage/backends/minio.py`

### Priority 4: Tests + Docs
- [ ] `tests/unit/`
- [ ] `tests/integration/`
- [ ] `docs/workflow-configs/`
- [ ] `docs/workers/`
- [ ] `docs/storage/`

---

## Reference Implementations

```
# Deep Research (Boss/Critic/Synthesis)
../../text/unified-llm-interface/backend/src/extensions/content/deep_research/

# PDF (Chunk processing)
../../text/pdf-summarizer/src/pdf_summarizer/

# make-llmstxt (Scraping)
../../text/make-llmstxt/src/make_llmstxt/

# tracer (Multimodal)
../../operations/tracer/src/tracer/
```

---

## Infrastructure

```bash
# Start shared services
cd ../../shared-docker-infra
make core-up
make db-up

# Verify
curl http://litellm.local/v1/models
```

---

## Key Documents

- `AGENTS.md` - AI development guide
- `AI_REFERENCE.md` - Detailed implementation guide
- `IMPLEMENTATION_STATUS.md` - What's done vs TODO
- `design/` - Architecture documentation

---

## Coding Standards

- **Python 3.12+**
- **Type hints required** (mypy strict mode)
- **Ruff** for linting + formatting
- **pytest** for testing
- **Latest packages** (see `pyproject.toml`)

---

## Environment

```bash
# Python
uv run python --version  # 3.13.9

# Dependencies
uv lock                   # Update lock file
uv sync --group dev       # Install dev deps

# Check
uv run deep-research --help
```
