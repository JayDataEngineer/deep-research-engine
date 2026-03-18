# Reference Documentation

External library documentation for development reference.

## LangChain Ecosystem

These repositories are cloned for offline reference:

### [LangGraph](https://github.com/langchain-ai/langgraph)
**Path**: `./langgraph/`  
**Key Docs**:
- `langgraph/docs/concepts/` - Core concepts
- `langgraph/docs/how-to/` - How-to guides
- `langgraph/docs/tutorials/` - Tutorials
- `langgraph/langgraph/graph/` - Graph implementation source

**Relevant for**:
- StateGraph implementation
- Conditional edges
- Checkpointer integration
- Send API for parallel execution

---

### [DeepAgents](https://github.com/langchain-ai/deepagents)
**Path**: `./deepagents/`  
**Key Docs**:
- `deepagents/docs/` - DeepAgents documentation
- `deepagents/examples/` - Example agents
- `deepagents/src/` - Source code

**Relevant for**:
- Agent harness patterns
- Filesystem middleware
- Backend protocol
- SubAgent implementation

---

### [LangChain](https://github.com/langchain-ai/langchain)
**Path**: `./langchain/`  
**Key Docs**:
- `langchain/libs/core/` - Core LangChain
- `langchain/libs/community/` - Community integrations
- `langchain/docs/` - Documentation

**Relevant for**:
- Tool implementations
- LLM wrappers
- Embedding models
- Vector stores
- Memory systems

---

## Infrastructure Documentation

For shared infrastructure documentation:

**Path**: `../../shared-docker-infra/docs/`

### Key Sections:
- `services/` - LiteLLM, Postgres, Neo4j, MongoDB, MinIO
- `integration/` - How to connect to services
- `architecture/` - Network topology, security model
- `api-reference/` - API documentation

### Quick Links:
- [LiteLLM](../../shared-docker-infra/docs/services/litellm.md) - Inference gateway
- [PostgreSQL](../../shared-docker-infra/docs/services/postgresql.md) - Vector storage
- [Neo4j](../../shared-docker-infra/docs/services/neo4j.md) - Knowledge graph
- [MongoDB](../../shared-docker-infra/docs/services/mongodb.md) - Media files
- [MinIO](../../shared-docker-infra/docs/services/minio.md) - Object storage

---

## Local Documentation

- `../docs/` - Deep Research Engine documentation
- `../AGENTS.md` - AI development guide
- `../CLAUDE.md` - Quick reference
- `../design/` - Architecture documentation

---

## Updating Reference Repos

To update cloned repositories:

```bash
cd reference/langgraph && git pull origin main
cd reference/deepagents && git pull origin main
cd reference/langchain && git pull origin main
```

Or re-clone shallow copies:

```bash
rm -rf reference/*
git clone --depth 1 https://github.com/langchain-ai/langgraph reference/langgraph
git clone --depth 1 https://github.com/langchain-ai/deepagents reference/deepagents
git clone --depth 1 https://github.com/langchain-ai/langchain reference/langchain
```
