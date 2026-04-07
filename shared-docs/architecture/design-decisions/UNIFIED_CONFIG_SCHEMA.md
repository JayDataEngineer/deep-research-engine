# Unified Workflow Configuration Schema

This schema defines the YAML format for all workflows in the deep-research-engine.

## Core Philosophy

**One graph, many configurations.** The engine provides a single `Boss → Workers → Critic → Synthesis` pipeline. Each workflow configures:
- What workers spawn
- How they process
- What storage backends are required
- What the output format is

---

## Complete Schema

```yaml
# =============================================================================
# WORKFLOW METADATA
# =============================================================================
workflow:
  name: string                    # Unique identifier (e.g., "rag", "pdf_summarizer")
  version: string                 # Semver version
  description: string             # Human-readable description
  type: "research" | "ingestion" | "generation"  # Workflow category
  
# =============================================================================
# INPUT SPECIFICATION
# =============================================================================
input:
  type: "query" | "pdf" | "telegram_export" | "url" | "directory"
  source: string                  # Path, URL, or query string
  schema:                         # Optional: validate input structure
    type: object
    properties:
      query: {type: string}
      documents: {type: array, items: {type: string}}
      
# =============================================================================
# PHASE CONFIGURATION (Core Graph)
# =============================================================================
phases:
  # -------------------------------------------------------------------------
  # PHASE 1: BOSS (Planning)
  # -------------------------------------------------------------------------
  - name: boss
    node: BossNode
    config:
      # Planning strategy
      strategy: "dynamic" | "fixed" | "complexity_based"
      
      # How many workers to spawn (can be dynamic)
      workers:
        min: 1
        max: 5
        complexity_thresholds:    # Optional: scale workers by query complexity
          - threshold: 0.3
            workers: 1
          - threshold: 0.7
            workers: 3
          - threshold: 0.9
            workers: 5
      
      # Prompt configuration
      prompt:
        system: string            # System prompt (Jinja2 template)
        variables:                # Variables to inject into prompt
          - query
          - context
      
      # Output: Research plan
      output:
        format: "research_brief" | "task_list" | "graph_spec"
      
  # -------------------------------------------------------------------------
  # PHASE 2: WORKERS (Parallel Processing)
  # -------------------------------------------------------------------------
  - name: workers
    node: WorkerNode | FlashExplorerNode
    config:
      # Worker type determines what tools/processing available
      worker_type: "web_researcher" | "chunk_processor" | "multimodal" | "scraper"
      
      # Parallel execution
      parallel: true
      max_concurrent: 10          # Max parallel workers
      
      # Recursion limit (for Flash Explorer style agents)
      recursion_limit: 50         # Max LLM calls per worker
      
      # Each worker gets a task from the boss plan
      task_assignment: "round_robin" | "specialized" | "dynamic"
      
      # Worker-specific configuration
      workers:
        - id: researcher_0
          type: web_researcher
          tools: [web_search, web_get, web_map]
          config:
            max_urls: 5
            max_depth: 2
            
        - id: researcher_1
          type: pdf_processor
          tools: [vector_search, add_documents]
          config:
            chunk_size: 1000
            overlap: 200
            
        - id: researcher_2
          type: multimodal
          tools: [asr_transcribe, vlm_analyze, face_recognize]
          config:
            batch_size: 10
      
      # Critic embedded in worker loop (Flash Explorer pattern)
      critic:
        enabled: true
        type: "extraction_quality" | "coverage" | "multimodal_accuracy"
        pass_threshold: 0.7
        max_retries: 3
        prompt:
          system: string          # Critic system prompt
          
      # What workers return
      output:
        format: "findings" | "processed_chunks" | "transcriptions"
        accumulate: true          # Accumulate results in state
      
  # -------------------------------------------------------------------------
  # PHASE 3: CRITIC (Quality Gate)
  # -------------------------------------------------------------------------
  - name: critic
    node: CriticNode
    config:
      # What to evaluate
      evaluate: "worker_outputs" | "synthesis_draft" | "final_output"
      
      # Evaluation criteria
      criteria:
        - name: "completeness"
          weight: 0.4
          threshold: 0.7
        - name: "accuracy"
          weight: 0.4
          threshold: 0.8
        - name: "coherence"
          weight: 0.2
          threshold: 0.6
      
      # Structured output schema for critic
      output_schema:
        type: object
        properties:
          passed: {type: boolean}
          score: {type: number, min: 0, max: 1}
          issues: {type: array, items: {type: string}}
          suggestions: {type: array, items: {type: string}}
      
      # What happens on failure
      on_fail:
        action: "retry_workers" | "revise_synthesis" | "terminate"
        max_retries: 3
        feedback_to: "boss" | "workers" | "synthesis"
      
  # -------------------------------------------------------------------------
  # PHASE 4: SYNTHESIS (Final Output)
  # -------------------------------------------------------------------------
  - name: synthesis
    node: SynthesisNode
    config:
      # What to synthesize
      sources: "worker_outputs" | "critique_history" | "all_artifacts"
      
      # Synthesis strategy
      strategy: "hierarchical" | "narrative" | "structured_report"
      
      # Output format
      output:
        format: "markdown" | "json" | "graph" | "podcast_script"
        template: string          # Optional: Jinja2 template
        sections:                 # For structured outputs
          - name: "Executive Summary"
            order: 1
          - name: "Key Findings"
            order: 2
          - name: "Sources"
            order: 3
      
      # Citation handling
      citations:
        enabled: true
        format: "numbered" | "apa" | "mla"
        require_url: true
        
# =============================================================================
# STORAGE BACKENDS (Pluggable)
# =============================================================================
storage:
  # Postgres (required for embeddings + semantic search)
  postgres:
    enabled: true
    connection: "${POSTGRES_URL}"  # Environment variable expansion
    tables:
      artifacts: "artifacts"
      embeddings: "artifact_embeddings"
    options:
      pool_min_size: 5
      pool_max_size: 20
      
  # MongoDB (optional - for media files)
  mongodb:
    enabled: false                 # Default: disabled
    connection: "${MONGO_URL}"
    collections:
      media: "media_files"
      metadata: "media_metadata"
      
  # Neo4j (optional - for knowledge graphs)
  neo4j:
    enabled: false                 # Default: disabled
    uri: "${NEO4J_URI}"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"
    
  # MinIO/S3 (optional - for large file storage)
  minio:
    enabled: false                 # Default: disabled
    endpoint: "${MINIO_ENDPOINT}"
    bucket: "deep-research"
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
    
  # In-memory fallback (for simple workflows)
  memory:
    enabled: true                  # Always available
    max_size_mb: 500
    
# =============================================================================
# LLM CONFIGURATION
# =============================================================================
llm:
  # Model slots (same pattern as deep research)
  models:
    main: "qwen-35-27"            # Complex reasoning
    tool: "qwen-35-27-fast"       # Fast operations
    embedding: "jina-v5-embed"    # Embeddings
    
  # Per-role configuration
  roles:
    boss:
      model_slot: main
      temperature: 0.5
      max_tokens: 2000
      enable_thinking: true
      
    worker:
      model_slot: tool
      temperature: 1.0
      max_tokens: 2000
      enable_thinking: false
      
    critic:
      model_slot: main
      temperature: 0.4
      max_tokens: 1500
      enable_thinking: true
      
    synthesis:
      model_slot: main
      temperature: 0.6
      max_tokens: 4000
      enable_thinking: true
      
# =============================================================================
# TOOLS CONFIGURATION
# =============================================================================
tools:
  # Enable tools by name or tag
  enabled:
    - web_search
    - web_get
    - vector_search
    - add_documents
    
  # Tool-specific configuration
  config:
    web_search:
      max_results: 10
      timeout: 30
    vector_search:
      top_k: 5
      threshold: 0.7
      
# =============================================================================
# EXECUTION SETTINGS
# =============================================================================
execution:
  # Recursion limit (global)
  max_steps: 100
  
  # Timeout (seconds)
  timeout: 600
  
  # Retry policy
  retry:
    max_attempts: 3
    backoff_factor: 2.0
    
  # Streaming
  streaming:
    enabled: true
    events: ["planning", "worker_start", "worker_complete", "critique", "synthesis"]
    
# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
output:
  # Where to write results
  destination:
    type: "filesystem" | "database" | "both"
    path: "./output"
    
  # Format
  format: "markdown" | "json" | "html"
  
  # Metadata to include
  metadata:
    include_timing: true
    include_token_usage: true
    include_model_info: true
```

---

## Workflow Examples

### Example 1: RAG Research (Original Deep Research)

```yaml
workflow:
  name: rag
  type: research
  
input:
  type: query
  source: "${QUERY}"
  
phases:
  - name: boss
    config:
      strategy: complexity_based
      workers:
        min: 1
        max: 3
        
  - name: workers
    config:
      worker_type: web_researcher
      parallel: true
      workers:
        - type: web_researcher
          tools: [web_search, web_get, web_research_agent]
        - type: web_researcher
          tools: [community_deep_agent]
      critic:
        enabled: true
        type: extraction_quality
        pass_threshold: 0.7
        
  - name: critic
    config:
      evaluate: worker_outputs
      on_fail:
        action: retry_workers
        
  - name: synthesis
    config:
      strategy: narrative
      output:
        format: markdown
        citations:
          enabled: true
          format: numbered

storage:
  postgres:
    enabled: true
  neo4j:
    enabled: true
  mongodb:
    enabled: false
  minio:
    enabled: false
```

### Example 2: PDF Summarizer

```yaml
workflow:
  name: pdf_summarizer
  type: ingestion
  
input:
  type: pdf
  source: "${PDF_PATH}"
  
phases:
  - name: boss
    config:
      strategy: fixed
      workers:
        min: 1
        max: 1
      output:
        format: task_list
        
  - name: workers
    config:
      worker_type: chunk_processor
      parallel: true
      max_concurrent: 5
      workers:
        - type: chunk_processor
          tools: [llm_classify, llm_extract]
          config:
            chunk_strategy: structural
      critic:
        enabled: true
        type: extraction_quality
        pass_threshold: 0.6
        
  - name: workers  # Second worker phase for deduplication
    config:
      worker_type: dedupe_worker
      tools: [embed, cluster]
      config:
        similarity_threshold: 0.85
        
  - name: synthesis
    config:
      strategy: hierarchical
      output:
        format: markdown
        sections:
          - name: Executive Summary
          - name: Chapter Summaries
          - name: Key Points

storage:
  postgres:
    enabled: false
  neo4j:
    enabled: false
  memory:
    enabled: true
```

### Example 3: Telegram Ingestion

```yaml
workflow:
  name: telegram_ingest
  type: ingestion
  
input:
  type: telegram_export
  source: "${EXPORT_PATH}"
  
phases:
  - name: boss
    config:
      strategy: fixed
      output:
        format: task_list
        
  - name: workers
    config:
      worker_type: multimodal
      parallel: true
      max_concurrent: 10
      workers:
        - type: asr_worker
          tools: [qwen3_asr]
        - type: vlm_worker
          tools: [qwen2_5_vl]
        - type: face_worker
          tools: [compreface]
        - type: text_worker
          tools: [entity_extraction]
      critic:
        enabled: true
        type: multimodal_accuracy
        
  - name: synthesis
    config:
      strategy: structured_report
      output:
        format: graph
        sections:
          - name: People
          - name: Topics
          - name: Media

storage:
  postgres:
    enabled: true
  mongodb:
    enabled: true
  neo4j:
    enabled: true
  minio:
    enabled: true
```

### Example 4: Skill Generator (make-llmstxt)

```yaml
workflow:
  name: skill_generator
  type: generation
  
input:
  type: url
  source: "${TARGET_URL}"
  
phases:
  - name: boss
    config:
      strategy: fixed
      output:
        format: task_list
        
  - name: workers
    config:
      worker_type: scraper
      parallel: true
      workers:
        - type: scraper
          tools: [mcp_scrape, search_docs]
          config:
            max_pages: 50
      critic:
        enabled: true
        type: coverage
        pass_threshold: 0.7
        
  - name: synthesis
    config:
      strategy: structured_report
      output:
        format: markdown
        template: llms_txt_template.jinja

storage:
  memory:
    enabled: true
  postgres:
    enabled: false
```

---

## Environment Variable Expansion

All `${VAR}` syntax expands from environment variables:

```yaml
storage:
  postgres:
    connection: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/db"
```

Supports defaults:
```yaml
models:
  main: "${MAIN_MODEL-qwen-35-27}"  # Default if not set
```

---

## Prompt Templates (Jinja2)

All prompts use Jinja2 templating:

```yaml
phases:
  - name: boss
    config:
      prompt:
        system: |
          You are a research planner.
          
          Query: {{ query }}
          
          {% if context %}
          Context:
          {{ context }}
          {% endif %}
          
          Plan {{ min_workers }}-{{ max_workers }} research angles.
```

Variables available:
- `query` - Input query
- `context` - Additional context
- `chat_history` - Conversation history
- `worker_outputs` - Results from workers (for synthesis)
- `critique` - Critic feedback (for revision)
