# Workflow Configuration Reference

Complete reference for configuring deep research workflows using YAML.

## Overview

The Deep Research Engine uses YAML configuration files to define workflow behavior. Each config specifies:

- **Workflow metadata** - Name, type, version
- **Input source** - Query, file, or URL
- **Phases** - Boss, workers, critic, synthesis
- **Storage backends** - PostgreSQL, MongoDB, Neo4j, MinIO
- **LLM configuration** - Models for different tasks
- **Tools** - Available tools for workers
- **Execution settings** - Timeouts, concurrency, streaming
- **Output format** - Destination and format

## Quick Start

```yaml
workflow:
  name: my_research
  type: research
  version: 1.0.0

input:
  type: query
  source: "${QUERY}"

phases:
  - name: boss
    node: BossNode
    config:
      strategy: complexity_based

  - name: workers
    node: WorkerNode
    config:
      worker_type: web_researcher
      parallel: true

  - name: critic
    node: CriticNode

  - name: synthesis
    node: SynthesisNode
    config:
      format: research_report

storage:
  memory:
    enabled: true

execution:
  max_steps: 100
  timeout: 600

output:
  destination:
    type: file
    path: ./output
  format: markdown
```

## Configuration Schema

### `workflow` Section

Workflow metadata and identification.

```yaml
workflow:
  name: string              # Unique workflow name
  type: string              # Workflow type: research, pdf_summarization, skill_generation, telegram_ingest
  version: string           # Semantic version (e.g., "1.0.0")
  description: string       # Optional description
```

**Example:**
```yaml
workflow:
  name: rag_research
  type: research
  version: 1.0.0
  description: Multi-agent RAG research with web search
```

### `input` Section

Defines workflow input source.

```yaml
input:
  type: string              # Input type: query, pdf, urls, telegram_export
  source: string|list       # Input source (query string, file path, URL list)
```

**Types:**

| Type | Source Format | Description |
|------|--------------|-------------|
| `query` | string | Text query for research |
| `pdf` | string | Path to PDF file |
| `urls` | list | List of URLs to process |
| `telegram_export` | string | Path to Telegram export directory |

**Examples:**

```yaml
# Query input
input:
  type: query
  source: "What is quantum computing?"

# PDF input
input:
  type: pdf
  source: "/path/to/document.pdf"

# Multiple URLs
input:
  type: urls
  source:
    - https://docs.python.org
    - https://docs.djangoproject.com
```

### `phases` Section

Defines workflow execution phases. Can be a list (ordered) or dict.

#### List Format (Recommended)

```yaml
phases:
  - name: boss
    node: BossNode
    config:
      # Boss configuration

  - name: workers
    node: WorkerNode
    config:
      # Workers configuration

  - name: critic
    node: CriticNode
    config:
      # Critic configuration

  - name: synthesis
    node: SynthesisNode
    config:
      # Synthesis configuration
```

#### Boss Phase Configuration

```yaml
phases:
  - name: boss
    node: BossNode
    config:
      strategy: string                    # Planning strategy
      workers:
        min: integer                      # Minimum workers
        max: integer                      # Maximum workers
      output:
        format: string                    # Output format
```

**Strategies:**

| Strategy | Description |
|----------|-------------|
| `complexity_based` | Adjusts worker count based on query complexity |
| `simple` | Single worker for simple queries |
| `comprehensive` | Maximum workers for thorough coverage |

**Example:**
```yaml
phases:
  - name: boss
    node: BossNode
    config:
      strategy: complexity_based
      workers:
        min: 1
        max: 5
      output:
        format: research_brief
```

#### Workers Phase Configuration

```yaml
phases:
  - name: workers
    node: WorkerNode
    config:
      worker_type: string                 # Worker type to use
      parallel: boolean                   # Execute in parallel
      max_concurrent: integer             # Maximum concurrent workers
      recursion_limit: integer            # Maximum recursion depth
      
      critic:
        enabled: boolean                  # Enable per-worker critic
        type: string                      # Critic type
        pass_threshold: float             # Quality threshold (0-1)
        max_retries: integer              # Retry count
      
      output:
        format: string                    # Output format
        accumulate: boolean               # Accumulate results
```

**Worker Types:**

| Type | Description |
|------|-------------|
| `web_researcher` | Web search and research |
| `chunk_processor` | PDF chunk processing |
| `scraper` | Web scraping |
| `multimodal` | ASR, VLM, face processing |
| `synthesis` | Report synthesis |

**Example:**
```yaml
phases:
  - name: workers
    node: WorkerNode
    config:
      worker_type: web_researcher
      parallel: true
      max_concurrent: 10
      recursion_limit: 50
      
      critic:
        enabled: true
        type: extraction_quality
        pass_threshold: 0.7
        max_retries: 3
      
      output:
        format: findings
        accumulate: true
```

#### Critic Phase Configuration

```yaml
phases:
  - name: critic
    node: CriticNode
    config:
      evaluate: string                    # What to evaluate
      criteria:                           # Evaluation criteria
        - name: string
          weight: float
          threshold: float
      
      on_fail:
        action: string                    # Action on failure
        max_retries: integer
```

**Criteria:**

| Name | Description | Typical Threshold |
|------|-------------|-------------------|
| `completeness` | Coverage completeness | 0.7 |
| `accuracy` | Factual accuracy | 0.8 |
| `relevance` | Query relevance | 0.75 |
| `quality` | Overall quality | 0.7 |

**On Fail Actions:**

| Action | Description |
|--------|-------------|
| `retry_workers` | Retry worker phase |
| `terminate` | Stop workflow |
| `proceed` | Continue despite failure |

**Example:**
```yaml
phases:
  - name: critic
    node: CriticNode
    config:
      evaluate: worker_outputs
      criteria:
        - name: completeness
          weight: 0.4
          threshold: 0.7
        - name: accuracy
          weight: 0.4
          threshold: 0.8
      
      on_fail:
        action: retry_workers
        max_retries: 3
```

#### Synthesis Phase Configuration

```yaml
phases:
  - name: synthesis
    node: SynthesisNode
    config:
      worker_type: string                 # Usually "synthesis"
      output:
        format: string                    # Output format
        citations:
          enabled: boolean
          format: string
      
      format_config:
        max_retries: integer
        include_metadata: boolean
        min_sections: integer
        min_findings: integer
```

**Output Formats:**

| Format | Description |
|--------|-------------|
| `research_report` | Full research report |
| `markdown` | Markdown document |
| `structured_json` | JSON structure |
| `summary` | Brief summary |

**Citation Formats:**

| Format | Example |
|--------|---------|
| `numbered` | [1], [2], [3] |
| `APA` | (Smith, 2024) |
| `MLA` | (Smith 123) |
| `chicago` | Smith, John. "Title." 2024. |

**Example:**
```yaml
phases:
  - name: synthesis
    node: SynthesisNode
    config:
      worker_type: synthesis
      output:
        format: research_report
        citations:
          enabled: true
          format: numbered
      
      format_config:
        max_retries: 10
        include_metadata: true
        min_sections: 3
        min_findings: 3
```

### `storage` Section

Configures storage backends.

```yaml
storage:
  postgres:
    enabled: boolean
    connection: string              # Connection URL
    vector_index: string            # Vector index name
  
  neo4j:
    enabled: boolean
    uri: string                     # Neo4j URI
    username: string
    password: string
  
  mongodb:
    enabled: boolean
    connection: string              # Connection URL
    database: string                # Database name
  
  minio:
    enabled: boolean
    endpoint: string                # MinIO endpoint
    access_key: string
    secret_key: string
    bucket: string
  
  memory:
    enabled: boolean                # In-memory storage (default)
```

**Example:**
```yaml
storage:
  postgres:
    enabled: true
    connection: "${POSTGRES_URL-postgresql://localhost:5433/deep_research}"
    vector_index: research_vectors
  
  neo4j:
    enabled: true
    uri: "${NEO4J_URI-bolt://localhost:7687}"
    username: neo4j
    password: "${NEO4J_PASSWORD}"
  
  mongodb:
    enabled: false
  
  minio:
    enabled: false
  
  memory:
    enabled: true
```

### `llm` Section

Configures LLM models.

```yaml
llm:
  models:
    main: string                      # Main model for reasoning
    tool: string                      # Model for tool use
    embedding: string                 # Embedding model
  
  provider: string                    # LLM provider (optional)
  api_key: string                     # API key (or use env var)
  base_url: string                    # Custom base URL (optional)
```

**Example:**
```yaml
llm:
  models:
    main: qwen-35-27
    tool: qwen-35-27-fast
    embedding: jina-v5-embed
  
  provider: openai
  api_key: "${OPENAI_API_KEY}"
  base_url: "http://litellm.local/v1"
```

### `tools` Section

Configures available tools.

```yaml
tools:
  enabled:                            # List of enabled tools
    - web_search
    - web_get
    - web_research_agent
    - vector_search
    - pdf_reader
    - scraper
  
  config:                             # Tool-specific config
    web_search:
      engine: string                  # Search engine
      num_results: integer
    vector_search:
      index: string
      top_k: integer
```

**Available Tools:**

| Tool | Description |
|------|-------------|
| `web_search` | Web search |
| `web_get` | URL content retrieval |
| `web_research_agent` | Autonomous web research |
| `vector_search` | Vector database search |
| `pdf_reader` | PDF processing |
| `scraper` | Web scraping |

**Example:**
```yaml
tools:
  enabled:
    - web_search
    - web_get
    - vector_search
  
  config:
    web_search:
      engine: google
      num_results: 10
    vector_search:
      index: research_index
      top_k: 5
```

### `execution` Section

Configures workflow execution.

```yaml
execution:
  max_steps: integer                # Maximum execution steps
  timeout: integer                  # Timeout in seconds
  logging: string                   # Log level
  
  streaming:
    enabled: boolean                # Enable streaming
    batch_size: integer             # Batch size for streaming
  
  retry:
    max_attempts: integer           # Max retry attempts
    delay: integer                  # Delay between retries
```

**Example:**
```yaml
execution:
  max_steps: 100
  timeout: 600
  logging: info
  
  streaming:
    enabled: true
    batch_size: 10
  
  retry:
    max_attempts: 3
    delay: 1
```

### `output` Section

Configures output destination and format.

```yaml
output:
  destination:
    type: string                    # Destination type
    path: string                    # Output path
  
  format: string                    # Output format
  include_metadata: boolean         # Include metadata
```

**Destination Types:**

| Type | Description |
|------|-------------|
| `file` | Write to file |
| `memory` | Store in memory |
| `both` | File + memory |
| `database` | Store in database |

**Output Formats:**

| Format | Description |
|--------|-------------|
| `markdown` | Markdown document |
| `json` | JSON format |
| `html` | HTML document |
| `text` | Plain text |

**Example:**
```yaml
output:
  destination:
    type: file
    path: ./output/research_report.md
  format: markdown
  include_metadata: true
```

## Environment Variables

Use environment variables with `${VAR}` or `${VAR-default}` syntax:

```yaml
workflow:
  name: ${WORKFLOW_NAME-my_workflow}

storage:
  postgres:
    connection: "${POSTGRES_URL-postgresql://localhost:5433/deep_research}"
  
  neo4j:
    password: "${NEO4J_PASSWORD}"

llm:
  api_key: "${OPENAI_API_KEY}"
```

## Complete Example Configs

### RAG Research

```yaml
workflow:
  name: rag_research
  type: research
  version: 1.0.0
  description: Multi-agent RAG research

input:
  type: query
  source: "${QUERY}"

phases:
  - name: boss
    node: BossNode
    config:
      strategy: complexity_based
      workers:
        min: 1
        max: 3
    
  - name: workers
    node: WorkerNode
    config:
      worker_type: web_researcher
      parallel: true
      max_concurrent: 10
      critic:
        enabled: true
        pass_threshold: 0.7
        max_retries: 3
    
  - name: critic
    node: CriticNode
    config:
      criteria:
        - name: completeness
          weight: 0.4
          threshold: 0.7
        - name: accuracy
          weight: 0.4
          threshold: 0.8
    
  - name: synthesis
    node: SynthesisNode
    config:
      format: research_report
      citations:
        enabled: true
        format: numbered

storage:
  postgres:
    enabled: true
    connection: "${POSTGRES_URL}"
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
  streaming:
    enabled: true

output:
  destination:
    type: both
    path: ./output
  format: markdown
```

### PDF Summarizer

```yaml
workflow:
  name: pdf_summarizer
  type: pdf_summarization
  version: 1.0.0

input:
  type: pdf
  source: "${PDF_PATH}"

phases:
  - name: boss
    node: BossNode
    config:
      strategy: simple
  
  - name: workers
    node: WorkerNode
    config:
      worker_type: chunk_processor
      parallel: true
      max_concurrent: 5
  
  - name: critic
    node: CriticNode
  
  - name: synthesis
    node: SynthesisNode
    config:
      format: summary

storage:
  memory:
    enabled: true

execution:
  max_steps: 50
  timeout: 300

output:
  destination:
    type: file
    path: ./output/summary.md
  format: markdown
```

## Usage

### CLI

```bash
# Run with config
uv run deep-research research "query" --config configs/rag.yaml

# Run PDF summarizer
uv run deep-research pdf /path/to/document.pdf --config configs/pdf_summarizer.yaml

# Run with custom config
uv run deep-research research "query" --config /path/to/custom.yaml
```

### Python API

```python
from deep_research_engine import run_workflow, load_config

# Load config
config = load_config("configs/rag.yaml")

# Run workflow
async for event in run_workflow("Research query", config):
    print(event)
```

## Best Practices

1. **Use environment variables** for sensitive data (API keys, passwords)
2. **Start simple** - Begin with minimal config, add complexity as needed
3. **Test phases individually** - Validate each phase before full workflow
4. **Set appropriate timeouts** - Prevent infinite loops
5. **Enable streaming** for long-running workflows
6. **Use critic** for quality-critical workflows
7. **Configure storage** based on data persistence needs

## Troubleshooting

### Common Issues

**Config validation errors:**
```yaml
# Ensure phases is a list or dict
phases:
  - name: boss
    # ...
```

**Environment variable not expanded:**
```yaml
# Use quotes around ${VAR} syntax
connection: "${POSTGRES_URL-default_value}"
```

**Worker type not found:**
```yaml
# Verify worker type is registered
worker_type: web_researcher  # Not "web_researcher_worker"
```

## Reference

- [Getting Started](../getting-started/)
- [Workers Guide](../workers/)
- [Storage Backends](../storage/)
- [API Reference](../api-reference/)
