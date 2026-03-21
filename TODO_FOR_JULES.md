# Complete Implementation Plan for Deep Research Engine

This document outlines the explicitly remaining tasks required to reach 100% completion for the Deep Research Engine project, migrating it fully to the new 5-Node Pipeline (Ingestion -> Drafter <-> Critic -> Neo4j -> Synthesizer) with isolated storage and config-driven workers.

## 1. High Priority: Core Agent/Node Logic

### A. Drafter Node & Workers (`src/deep_research_engine/nodes/drafter.py` & `src/deep_research_engine/workers/implementations/`)
- [ ] **Flash Subagent Orchestration:** Implement the `web_researcher` worker wrapping the Flash Explorer pattern.
- [ ] **Specialist Fan-Out:** Implement specific sub-agents:
  - `chunk_processor.py` (PDF chunk processing)
  - `multimodal.py` (ASR/VLM/face processing)
  - `scraper.py` (Web scraping, e.g., make-llmstxt logic)
- [ ] **Dependency Integration:** Integrate heavily with `deepagents` for generic base logic.

### B. Ingestion Node (`src/deep_research_engine/nodes/ingestion.py`)
- [ ] **File Extraction:** Implement logic to extract raw PDFs, Images, and Audio.
- [ ] **"Nice Text Files" Conversion:** Ensure all multimodal inputs are converted to clean, "nice text files" before hitting the Drafter.
- [ ] **Multimodal Integration:** Integrate with shared multimodal tools (e.g., ASR, VLM shared docker infrastructure).

### C. Critic Node (`src/deep_research_engine/nodes/critic.py`)
- [ ] **Structured Evaluation:** Implement structured LLM outputs to evaluate Drafter output quality.
- [ ] **Feedback Loop:** Build the routing logic to provide specific, actionable feedback back to the Drafter node for revision.

---

## 2. Medium Priority: Pluggable Storage & CLI

### A. Storage Backends (`src/deep_research_engine/storage/backends/`)
Develop pluggable storage backends complying with the `StorageBackend` interface:
- [ ] `postgres.py`: PostgreSQL + pgvector for semantic search storage.
- [ ] `mongodb.py`: MongoDB for raw media file storage.
- [ ] `neo4j.py`: Neo4j for the knowledge graph builder.
- [ ] `minio.py`: MinIO S3-compatible backend for massive object storage.

### B. CLI Refinement
- [ ] **CLI Updates:** Update the CLI entrypoints (`uv run deep-research`) so that subcommands map accurately to the finalized nodes and configs.

---

## 3. Low Priority: Testing & Documentation

### A. Automated Testing (`tests/`)
- [ ] **Unit Tests:** 
  - `test_state.py`, `test_worker_registry.py`, `test_storage_registry.py`, `test_config_loader.py`.
- [ ] **Integration Tests:** 
  - Individual nodes (`test_boss_node.py`, `test_worker_node.py`, `test_critic_node.py`).
- [ ] **End-to-End Workflow Tests:** 
  - Build pipeline execution tests for `test_rag_workflow.py`, `test_pdf_workflow.py`, `test_telegram_workflow.py`.

### B. Final Deliverable Documentation (`docs/`)
- [ ] **Workflow Configs Guide:** Document schemas and provide examples (`docs/workflow-configs/`).
- [ ] **Workers Guide:** Provide a guide on how to build custom workers (`docs/workers/`).
- [ ] **Storage Guide:** Document the custom storage backends interface (`docs/storage/`).
