# Workflow Configuration Reference

Workflow configs define the entire pipeline: Boss → Workers → Critic → Synthesis.

## Schema

```yaml
workflow:
  name: string                    # Unique identifier
  type: "research" | "ingestion" | "generation"
  version: string                 # Semver
  description: string

input:
  type: "query" | "pdf" | "telegram_export" | "url"
  source: string                  # Path, URL, or query

phases:
  - name: boss
    config:
      strategy: "complexity_based" | "fixed"
      workers:
        min: 1
        max: 3
      
  - name: workers
    config:
      worker_type: "web_researcher" | "chunk_processor" | "multimodal" | "scraper"
      parallel: true
      max_concurrent: 10
      critic:
        enabled: true
        type: "extraction_quality"
        pass_threshold: 0.7
      
  - name: critic
    config:
      evaluate: "worker_outputs"
      criteria:
        - name: completeness
          weight: 0.4
          threshold: 0.7
      
  - name: synthesis
    config:
      strategy: "narrative" | "hierarchical" | "structured_report"
      output:
        format: "markdown" | "json" | "graph"

storage:
  postgres:
    enabled: true
    connection: "${POSTGRES_URL}"
  neo4j:
    enabled: false
  mongodb:
    enabled: false
  minio:
    enabled: false
  memory:
    enabled: true

llm:
  models:
    main: qwen-35-27
    tool: qwen-35-27-fast

tools:
  enabled:
    - web_search
    - vector_search

execution:
  max_steps: 100
  timeout: 600
```

## Environment Variables

Use `${VAR}` or `${VAR-default}` syntax:

```yaml
storage:
  postgres:
    connection: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5433/db"
```

## Examples

See:
- `configs/rag.yaml` - RAG research
- `configs/pdf_summarizer.yaml` - PDF processing
- `configs/telegram_ingest.yaml` - Telegram ingestion
- `configs/skill_generator.yaml` - Skill generation
