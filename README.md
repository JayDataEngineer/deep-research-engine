# Deep Research Engine

**Unified engine for autonomous research, ingestion, and knowledge graph construction.**

## Architecture Overview

Deep Research Engine is an **isolated, config-driven system** designed to run complex multi-agent workflows. It follows a clean 5-node pipeline:

1.  **Ingestion** (Optional): Converts non-text data (PDFs, Images, Audio) into clean, structured text files.
2.  **Drafter**: The core node using **Flash Subagents** and **Specialist Sub-sub agents** to gather context and build a comprehensive draft.
3.  **Critic**: Evaluates the draft quality and either passes it forward or loops back to the Drafter for improvements.
4.  **Neo4j Builder** (Optional): Extracts entities and relationships to construct a persistent knowledge graph.
5.  **Synthesizer** (Optional): Formats the final research output into specific products (Reports, Threads, Podcast Scripts).

## 🚀 Quick Start

```bash
# Setup Environment
cp .env.example .env
# Strapped to shared-docker-infra environment variables

# Run a Research Workflow
deep-research research "Explain Quantum Computing" --config rag.yaml

# Run Ingestion + Research
deep-research research "Analyze these documents" --config pdf_summarizer.yaml --input ./my_docs/
```

## 🛠️ Infrastructure Integration

This engine is **fully strapped** into the `shared-docker-infra` environment:

-   **LiteLLM**: All model calls are routed through the LiteLLM gateway (usually on port 4000).
-   **LangFuse**: Used for prompt management, versioning, and full trace observability.
-   **MCP**: Leverages Model Context Protocol for secure tool execution.
-   **Audio/Vision**: Integrated with shared multimodal tools for ingestion.
-   **Databases**: Pluggable support for PostgreSQL (pgvector), Neo4j, MongoDB, and MinIO.

## 📂 Project Structure

- `src/deep_research_engine/nodes/`: Core pipeline implementation.
- `src/deep_research_engine/workers/`: Specialized worker agents used by Drafter.
- `src/deep_research_engine/storage/`: Pluggable storage backend drivers.
- `configs/`: YAML-based workflow definitions.
- `docs/`: Detailed technical documentation.

## ⚖️ Design Principles

-   **Isolation**: Zero external dependencies beyond the core engine and its config.
-   **Nice Text Files**: All non-text data is normalized to clean text before agents start work.
-   **Hierarchical Agents**: Drafter manages subagents; subagents manage specialists.
-   **Config-Driven**: A single YAML file defines the entire behavior of the 5-node graph.
