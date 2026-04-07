# Deep Research Engine Documentation

Welcome to the Deep Research Engine documentation.

## Quick Start

1. **[Installation](getting-started/installation.md)** - Install the package
2. **[Quick Start](getting-started/quickstart.md)** - Run your first workflow
3. **[Configuration](workflow-configs/index.md)** - YAML config reference

## Documentation Sections

### [Getting Started](getting-started/index.md)
- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Infrastructure Requirements](getting-started/infrastructure.md)

### [Workflow Configs](workflow-configs/index.md)
- [Config Schema](workflow-configs/schema.md)
- [RAG Research](workflow-configs/rag.md)
- [PDF Summarizer](workflow-configs/pdf.md)
- [Telegram Ingestion](workflow-configs/telegram.md)
- [Skill Generator](workflow-configs/skill.md)

### [Workers](workers/index.md)
- [BaseWorker ABC](workers/base.md)
- [Web Researcher](workers/web_researcher.md)
- [Chunk Processor](workers/chunk_processor.md)
- [Multimodal](workers/multimodal.md)
- [Scraper](workers/scraper.md)
- [Custom Workers](workers/custom.md)

### [Storage](storage/index.md)
- [Pluggable Backends](storage/backends.md)
- [In-Memory](storage/memory.md)
- [PostgreSQL](storage/postgres.md)
- [MongoDB](storage/mongodb.md)
- [Neo4j](storage/neo4j.md)
- [MinIO/S3](storage/minio.md)

### [API Reference](api-reference/index.md)
- [run_workflow()](api-reference/run_workflow.md)
- [load_config()](api-reference/load_config.md)
- [EngineState](api-reference/state.md)
- [BaseWorker](api-reference/worker.md)

### [Examples](examples/index.md)
- [RAG Research Example](examples/rag.md)
- [PDF Summarizer Example](examples/pdf.md)
- [FastAPI Integration](examples/fastapi.md)

## Infrastructure Integration

This engine integrates with your existing infrastructure:

```bash
# Start shared infrastructure
cd ~/Documents/programs/shared-docker-infra
make core-up
make db-up
```

See [shared-docker-infra/docs/](../../shared-docker-infra/docs/) for:
- LiteLLM setup
- Database configuration
- Network topology
- Security model

## Related Projects

This engine unifies:
- [unified-llm-interface](../text/unified-llm-interface/) - Deep research
- [pdf-summarizer](../text/pdf-summarizer/) - PDF processing
- [make-llmstxt](../text/make-llmstxt/) - Skill generation
- [tracer](../operations/tracer/) - Telegram ingestion

---

**Last Updated:** March 18, 2026
**Version:** 0.1.0
