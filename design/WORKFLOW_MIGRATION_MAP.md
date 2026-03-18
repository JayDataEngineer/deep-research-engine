# Workflow Migration Map

This document maps each of the 4 existing projects to the unified workflow configuration.

---

## Project 1: unified-llm-interface (Deep Research)

**Current State**: Full multi-agent research with Boss → Workers → Critic → Synthesis

**Migration**: Direct mapping (this IS the reference implementation)

### Unified Config

```yaml
# configs/rag.yaml
workflow:
  name: rag
  type: research
  version: 1.0.0
  description: Multi-agent RAG research with web + document search

input:
  type: query
  source: "${QUERY}"

phases:
  # Phase 1: Boss (Planning)
  - name: boss
    node: BossNode
    config:
      strategy: complexity_based
      workers:
        min: 1
        max: 3
        complexity_thresholds:
          - threshold: 0.3
            workers: 1
          - threshold: 0.7
            workers: 3
      output:
        format: research_brief

  # Phase 2: Workers (Flash Explorer + Deep Agents)
  - name: workers
    node: FlashExplorerNode
    config:
      worker_type: web_researcher
      parallel: true
      max_concurrent: 10
      recursion_limit: 50
      
      workers:
        - id: researcher_0
          type: web_researcher
          tools: [web_search, web_get, web_research_agent]
          config:
            max_urls: 5
            max_depth: 2
            
        - id: researcher_1
          type: web_researcher
          tools: [community_deep_agent]
          config:
            platform: auto
            limit: 20
      
      critic:
        enabled: true
        type: extraction_quality
        pass_threshold: 0.7
        max_retries: 3
      
      output:
        format: findings
        accumulate: true

  # Phase 3: Critic (Quality Gate)
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
        - name: coherence
          weight: 0.2
          threshold: 0.6
      on_fail:
        action: retry_workers
        max_retries: 3
        feedback_to: workers

  # Phase 4: Synthesis (Final Output)
  - name: synthesis
    node: SynthesisNode
    config:
      sources: worker_outputs
      strategy: narrative
      output:
        format: markdown
        citations:
          enabled: true
          format: numbered
          require_url: true

storage:
  postgres:
    enabled: true
    connection: "${CONTEXT_POSTGRES_URL}"
    tables:
      artifacts: artifacts
      embeddings: artifact_embeddings
  
  neo4j:
    enabled: true
    uri: "${NEO4J_URI}"
    username: neo4j
    password: "${NEO4J_PASSWORD}"
  
  mongodb:
    enabled: false
  
  minio:
    enabled: false

llm:
  models:
    main: qwen-35-27
    tool: qwen-35-27-fast
    embedding: jina-v5-embed
  
  roles:
    boss:
      model_slot: main
      temperature: 0.5
      enable_thinking: true
    
    worker:
      model_slot: tool
      temperature: 1.0
      enable_thinking: false
    
    critic:
      model_slot: main
      temperature: 0.4
      enable_thinking: true
    
    synthesis:
      model_slot: main
      temperature: 0.6
      enable_thinking: true

tools:
  enabled:
    - web_search
    - web_get
    - web_map
    - web_crawl
    - community_scrape
    - web_research_agent
    - vector_search
  
  config:
    web_search:
      max_results: 10
      timeout: 30
    vector_search:
      top_k: 5
      threshold: 0.7

execution:
  max_steps: 100
  timeout: 600
  retry:
    max_attempts: 3
    backoff_factor: 2.0
  streaming:
    enabled: true
    events:
      - planning
      - worker_start
      - worker_complete
      - critic_evaluation
      - synthesis_complete

output:
  destination:
    type: both
    path: ./output
  format: markdown
  metadata:
    include_timing: true
    include_token_usage: true
```

### Code Changes Required

**Minimal** - Most logic already in place:
1. Extract `FlashExplorerNode` from `src/extensions/content/deep_research/agents/flash_explorer.py`
2. Wrap as worker type in `workers/implementations/web_researcher.py`
3. Move config to YAML format
4. Use StorageManager instead of direct Context Engine calls

---

## Project 2: pdf-summarizer

**Current State**: Linear pipeline PARSE → FILTER → DEDUPE → SUMMARIZE

**Migration**: Map to Boss → Workers → Critic → Synthesis

### Unified Config

```yaml
# configs/pdf_summarizer.yaml
workflow:
  name: pdf_summarizer
  type: ingestion
  version: 1.0.0
  description: Hierarchical PDF processing with filler removal

input:
  type: pdf
  source: "${PDF_PATH}"

phases:
  # Phase 1: Boss (Parse + Plan)
  - name: boss
    node: BossNode
    config:
      strategy: fixed
      workers:
        min: 1
        max: 1
      output:
        format: task_list
      
      # Custom parsing logic
      custom:
        parser: pymupdf4llm
        chunk_strategy: structural
        preserve_hierarchy: true

  # Phase 2: Workers (Parallel Chunk Processing)
  - name: workers
    node: WorkerNode
    config:
      worker_type: chunk_processor
      parallel: true
      max_concurrent: 5
      
      workers:
        - id: chunk_processor
          type: chunk_processor
          tools: [llm_classify, llm_extract]
          config:
            chunk_size: 1000
            overlap: 200
      
      critic:
        enabled: true
        type: extraction_quality
        pass_threshold: 0.6
        max_retries: 3
      
      output:
        format: processed_chunks
        accumulate: true

  # Phase 2b: Workers (Deduplication)
  - name: workers_dedupe
    node: WorkerNode
    config:
      worker_type: dedupe_worker
      parallel: false
      
      workers:
        - id: dedupe_worker
          type: dedupe_worker
          tools: [embed, cluster]
          config:
            similarity_threshold: 0.85
            method: cosine_similarity
      
      output:
        format: deduplicated_chunks

  # Phase 3: Critic (Quality Gate)
  - name: critic
    node: CriticNode
    config:
      evaluate: worker_outputs
      criteria:
        - name: completeness
          weight: 0.5
          threshold: 0.7
        - name: conciseness
          weight: 0.5
          threshold: 0.6
      on_fail:
        action: retry_workers
        max_retries: 2

  # Phase 4: Synthesis (Hierarchical Summary)
  - name: synthesis
    node: SynthesisNode
    config:
      sources: worker_outputs
      strategy: hierarchical
      output:
        format: markdown
        sections:
          - name: Executive Summary
            order: 1
          - name: Chapter Summaries
            order: 2
          - name: Key Points
            order: 3
        
        template: |
          # {{ title }} - Executive Summary
          
          > {{ one_line_thesis }}
          
          ## Overview
          {{ overview }}
          
          ## Key Points by Chapter
          {{ chapter_summaries }}
          
          ## Critical Data & Arguments
          {{ key_points }}
          
          ---
          *Generated from {{ pdf_path }}*
          *Original: {{ total_pages }} pages → Summarized: {{ chunk_count }} chunks*
          *Filler removed: {{ filler_percent }}%*

storage:
  # PDF summarizer doesn't need persistent storage
  postgres:
    enabled: false
  
  neo4j:
    enabled: false
  
  mongodb:
    enabled: false
  
  minio:
    enabled: false
  
  memory:
    enabled: true
    max_size_mb: 500

llm:
  models:
    main: qwen-35-27
    tool: qwen-35-27-fast
    embedding: jina-v5-embed

tools:
  enabled:
    - llm_classify
    - llm_extract
    - embed
    - cluster

execution:
  max_steps: 50
  timeout: 300
  streaming:
    enabled: true

output:
  destination:
    type: filesystem
    path: ./output
  format: markdown
```

### Code Changes Required

**Moderate** - Need to adapt pipeline to graph:
1. Create `ChunkProcessorWorker` in `workers/implementations/chunk_processor.py`
2. Create `DedupeWorker` in `workers/implementations/dedupe_worker.py`
3. Map existing nodes to Boss/Critic/Synthesis
4. Remove direct LangGraph usage (use engine instead)

---

## Project 3: tracer (Telegram Ingestion)

**Current State**: Two-stage pipeline (Ingestion → Inference)

**Migration**: Ingestion maps to Workers phase, Inference maps to Boss → Workers → Synthesis

### Unified Config

```yaml
# configs/telegram_ingest.yaml
workflow:
  name: telegram_ingest
  type: ingestion
  version: 1.0.0
  description: Telegram export processing with multimodal analysis

input:
  type: telegram_export
  source: "${EXPORT_PATH}"

phases:
  # Phase 1: Boss (Parse Export)
  - name: boss
    node: BossNode
    config:
      strategy: fixed
      workers:
        min: 1
        max: 1
      output:
        format: task_list
      
      custom:
        parser: telegram_html
        extract_media_paths: true

  # Phase 2: Workers (Multimodal Processing)
  - name: workers
    node: WorkerNode
    config:
      worker_type: multimodal
      parallel: true
      max_concurrent: 10
      
      workers:
        - id: asr_worker
          type: asr_worker
          tools: [qwen3_asr]
          config:
            model: qwen3-asr
            language: auto
        
        - id: vlm_worker
          type: vlm_worker
          tools: [qwen2_5_vl]
          config:
            model: qwen2.5-vl
            prompt: "Describe this image, focusing on people, objects, and context."
        
        - id: face_worker
          type: face_worker
          tools: [compreface]
          config:
            compreface_url: "${COMPREFACE_URL}"
            api_key: "${COMPREFACE_API_KEY}"
            similarity_threshold: 0.8
        
        - id: text_worker
          type: text_worker
          tools: [entity_extraction]
          config:
            extract_people: true
            extract_topics: true
      
      critic:
        enabled: true
        type: multimodal_accuracy
        pass_threshold: 0.7
        max_retries: 2
      
      output:
        format: multimodal_processing
        accumulate: true

  # Phase 3: Critic (Quality Gate)
  - name: critic
    node: CriticNode
    config:
      evaluate: worker_outputs
      criteria:
        - name: transcription_quality
          weight: 0.3
          threshold: 0.7
        - name: description_quality
          weight: 0.3
          threshold: 0.7
        - name: entity_extraction
          weight: 0.4
          threshold: 0.6
      on_fail:
        action: retry_workers
        max_retries: 2

  # Phase 4: Synthesis (Graph Build + Export)
  - name: synthesis
    node: SynthesisNode
    config:
      sources: worker_outputs
      strategy: structured_report
      output:
        format: graph
        sections:
          - name: People
          - name: Topics
          - name: Media
          - name: Timeline
      
      custom:
        neo4j_export: true
        create_person_nodes: true
        create_topic_nodes: true
        create_relationships: true

storage:
  # Telegram needs ALL backends
  postgres:
    enabled: true
    connection: "${POSTGRES_URL}"
    tables:
      messages: messages
      transcriptions: transcriptions
  
  mongodb:
    enabled: true
    connection: "${MONGO_URL}"
    collections:
      media: media_files
      metadata: media_metadata
  
  neo4j:
    enabled: true
    uri: "${NEO4J_URI}"
    username: neo4j
    password: "${NEO4J_PASSWORD}"
  
  minio:
    enabled: true
    endpoint: "${MINIO_ENDPOINT}"
    bucket: telegram-exports

llm:
  models:
    main: qwen-35-27
    tool: qwen-35-27-fast
    embedding: jina-v5-embed

tools:
  enabled:
    - qwen3_asr
    - qwen2_5_vl
    - compreface
    - entity_extraction
    - vector_search

execution:
  max_steps: 200
  timeout: 1800
  streaming:
    enabled: true

output:
  destination:
    type: database
  format: graph
```

### Code Changes Required

**Moderate-High** - Need to split into two workflows:
1. Create `MultimodalWorker` in `workers/implementations/multimodal.py`
2. Create `ASRWorker`, `VLMWorker`, `FaceWorker`, `TextWorker` as subtypes
3. Create inference workflow (separate config for query phase)
4. Adapt Neo4j export to SynthesisNode

---

## Project 4: make-llmstxt

**Current State**: DeepAgents Generator-Critic loop

**Migration**: Map to Boss → Workers (scraper) → Critic → Synthesis

### Unified Config

```yaml
# configs/skill_generator.yaml
workflow:
  name: skill_generator
  type: generation
  version: 1.0.0
  description: Generate llms.txt and skill packages from documentation sites

input:
  type: url
  source: "${TARGET_URL}"

phases:
  # Phase 1: Boss (Plan Scraping)
  - name: boss
    node: BossNode
    config:
      strategy: fixed
      workers:
        min: 1
        max: 1
      output:
        format: task_list
      
      custom:
        domain_analysis: true
        estimate_pages: true

  # Phase 2: Workers (Parallel Scraping)
  - name: workers
    node: WorkerNode
    config:
      worker_type: scraper
      parallel: true
      max_concurrent: 5
      
      workers:
        - id: scraper
          type: scraper
          tools: [mcp_scrape, search_docs]
          config:
            max_pages: 50
            max_depth: 3
            follow_sitemap: true
      
      critic:
        enabled: true
        type: coverage
        pass_threshold: 0.7
        max_retries: 3
      
      output:
        format: scraped_content
        accumulate: true

  # Phase 3: Critic (Coverage Validation)
  - name: critic
    node: CriticNode
    config:
      evaluate: worker_outputs
      criteria:
        - name: coverage
          weight: 0.5
          threshold: 0.7
        - name: accuracy
          weight: 0.3
          threshold: 0.8
        - name: completeness
          weight: 0.2
          threshold: 0.7
      on_fail:
        action: retry_workers
        max_retries: 3
        feedback_to: workers

  # Phase 4: Synthesis (Generate Output)
  - name: synthesis
    node: SynthesisNode
    config:
      sources: worker_outputs
      strategy: structured_report
      output:
        format: markdown
        template: llms_txt_template
        
        sections:
          - name: Core
          - name: API Reference
          - name: Optional
        
        template: |
          # {{ site_name }}
          > {{ one_line_description }}
          
          ## Core
          {% for page in core_pages %}
          - [{{ page.title }}]({{ page.url }}): {{ page.description }}
          {% endfor %}
          
          ## API Reference
          {% for page in api_pages %}
          - [{{ page.title }}]({{ page.url }}): {{ page.description }}
          {% endfor %}

storage:
  # make-llmstxt only needs memory
  postgres:
    enabled: false
  
  neo4j:
    enabled: false
  
  mongodb:
    enabled: false
  
  minio:
    enabled: false
  
  memory:
    enabled: true
    max_size_mb: 200

llm:
  models:
    main: qwen-35-27
    tool: qwen-35-27-fast

tools:
  enabled:
    - mcp_scrape
    - search_docs
    - get_doc_by_url

execution:
  max_steps: 50
  timeout: 600
  streaming:
    enabled: true

output:
  destination:
    type: filesystem
    path: ./output
  format: markdown
  metadata:
    include_timing: false
```

### Code Changes Required

**Moderate** - Wrap DeepAgents in worker:
1. Create `ScraperWorker` in `workers/implementations/scraper.py`
2. Wrap existing `DeepAgentGenerator` as worker implementation
3. Reuse existing critic with structured output
4. Map synthesis to template-based generation

---

## Migration Priority

| Project | Complexity | Code Changes | Priority |
|---------|-----------|--------------|----------|
| **unified-llm-interface** | Low | Minimal | Migrate first (reference) |
| **pdf-summarizer** | Medium | Moderate | Migrate second |
| **make-llmstxt** | Medium | Moderate | Migrate third |
| **tracer** | High | Moderate-High | Migrate last (most complex) |

---

## Migration Checklist

### Phase 1: Extract Core Engine
- [ ] Create `deep-research-engine` package structure
- [ ] Implement `EngineState` TypedDict
- [ ] Implement `BaseWorker` ABC + `WorkerRegistry`
- [ ] Implement `StorageBackend` ABC + `StorageRegistry`
- [ ] Implement `BossNode`, `WorkerNode`, `CriticNode`, `SynthesisNode`
- [ ] Implement `run_workflow()` entry point

### Phase 2: Migrate unified-llm-interface
- [ ] Create `rag.yaml` config
- [ ] Wrap Flash Explorer as `WebResearcherWorker`
- [ ] Test RAG workflow with new engine
- [ ] Update unified-llm-interface to use engine

### Phase 3: Migrate pdf-summarizer
- [ ] Create `pdf_summarizer.yaml` config
- [ ] Implement `ChunkProcessorWorker`
- [ ] Implement `DedupeWorker`
- [ ] Test PDF workflow
- [ ] Deprecate old pdf-summarizer code

### Phase 4: Migrate make-llmstxt
- [ ] Create `skill_generator.yaml` config
- [ ] Implement `ScraperWorker`
- [ ] Wrap DeepAgentGenerator
- [ ] Test skill generation
- [ ] Deprecate old make-llmstxt code

### Phase 5: Migrate tracer
- [ ] Create `telegram_ingest.yaml` config
- [ ] Implement `MultimodalWorker` + subtypes
- [ ] Implement inference workflow
- [ ] Test Telegram ingestion
- [ ] Deprecate old tracer code

### Phase 6: Polish
- [ ] Add comprehensive tests
- [ ] Write documentation
- [ ] Create example configs
- [ ] Publish to PyPI
