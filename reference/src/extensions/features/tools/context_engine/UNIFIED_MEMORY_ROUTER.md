# Unified Memory Router Architecture

## Overview

The Unified Memory Router is a `CompositeBackend` from deepagents that centralizes all memory operations through a single interface. It implements the 4 entry points required for deep research:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEEP AGENT                                  │
│                                                                  │
│  write_file("/data/raw/dumps/...")  → S3 (context overflow)    │
│  write_file("/data/raw/web/...")    → S3 (web scrapes)         │
│  write_file("/data/insights/...")   → Postgres (vectorized)    │
│  write_file("/data/graph/...")      → Neo4j (knowledge graph)  │
│  write_file("/memories/...")        → LangGraph Store (search) │
│  write_file("/tmp/...")             → StateBackend (ephemeral) │
└─────────────────────────────────────────────────────────────────┘
```

## Entry Points

### Entry Point A: Context Overflow Dumps
When deep agents run out of context, they automatically dump state to S3.

```python
# Automatic - happens when context fills up
write_file("/data/raw/dumps/session_abc/state_step_5.json", state_dump)
```

### Entry Point B: Web Agent Synthesis
After critic approval, web agent synthesizes and dumps raw content.

```python
# In web_research_agent after critic pass
write_file("/data/raw/web/session_abc/url_hash_123.html", html_content)
write_file("/data/raw/web/session_abc/url_hash_123.json", metadata)
```

### Entry Point C: Worker Insights (Vectorized)
Parallel workers write findings that get auto-vectorized in Postgres.

```python
# In flash_explorer workers
write_file("/data/insights/session_abc/findings/worker_0.md", findings)
write_file("/data/insights/session_abc/claims/claim_1.md", claim)
```

### Entry Point D: Knowledge Graph
Graph builder writes entities and relationships to Neo4j.

```python
# In graph_builder_node
write_file("/data/graph/session_abc/entities/Apple", "Company|Technology company")
write_file("/data/graph/session_abc/relationships/Apple_PRODUCES_iPhone", "")
```

## Usage

### Basic Usage

```python
from deepagents import create_deep_agent
from src.extensions.features.tools.context_engine.storage.unified_memory_router import (
    create_unified_memory_router
)

# Create deep agent with unified memory router
agent = create_deep_agent(
    model=llm,
    backend=create_unified_memory_router,  # Factory function
    store=langgraph_store,
)
```

### With Explicit Session ID

```python
from src.extensions.features.tools.context_engine.storage.unified_memory_router import (
    unified_memory_router_factory
)

# Create factory with explicit session_id
backend_factory = unified_memory_router_factory(
    session_id="research_2024_01_15",
    enable_async_vectorization=True,
)

agent = create_deep_agent(
    model=llm,
    backend=backend_factory,
    store=langgraph_store,
)
```

### In Deep Research Graph

```python
# In flash_explorer.py or similar
from src.extensions.features.tools.context_engine.storage.unified_memory_router import (
    create_unified_memory_router
)

async def create_research_agent(yaml_config, session_id, store):
    llm = await create_llm_from_config(yaml_config, role="worker")

    def backend_factory(runtime):
        return create_unified_memory_router(
            runtime=runtime,
            session_id=session_id,
            include_s3=True,
            include_postgres=True,
            include_neo4j=True,
            include_memories=True,
        )

    return create_deep_agent(
        model=llm,
        tools=[web_search, web_get],
        backend=backend_factory,
        store=store,
    )
```

## Backend Implementations

### PostgresBackend

Implements `BackendProtocol` for Postgres + pgvector:

- **Path format**: `/data/insights/{session_id}/{artifact_type}/{filename}`
- **Auto-vectorization**: Content is automatically embedded on write
- **Deduplication**: Content hashing prevents duplicate artifacts
- **Session isolation**: Artifacts are scoped to session_id

### Neo4jBackend

Implements `BackendProtocol` for Neo4j knowledge graph:

- **Path format**: `/data/graph/{session_id}/entities/{entity_name}`
- **Content format**: `EntityType|Description`
- **MERGE semantics**: Idempotent entity creation
- **Session tracking**: session_ids array for isolation and cleanup

### MinioBackend (existing)

Implements `BackendProtocol` for S3/MinIO:

- **Path format**: `/{prefix}/{session_id}/{file_path}`
- **Full filesystem API**: read, write, edit, grep, glob, ls
- **Session isolation**: Automatic path prefixing with session_id

## Connection Pooling

Connection pools are managed at the router level, not per-node:

```python
# Postgres: Shared pool via PostgresVectorStore
postgres_store = PostgresVectorStore(
    connection_string=...,
    pool_min_size=5,
    pool_max_size=20,
)

# Neo4j: Native driver connection (shared)
neo4j_store = Neo4jStore(url=..., username=..., password=...)

# These are passed to backends via context_engine
```

## Async Vectorization (Optional)

Enable background vectorization to prevent blocking:

```python
backend_factory = unified_memory_router_factory(
    enable_async_vectorization=True,
)
```

This uses LangGraph background runs to process embeddings:

```python
# In PostgresBackend.awrite()
await langgraph_client.runs.create(
    thread_id=self.session_id,
    assistant_id="vectorization_worker",
    input={"artifact_id": artifact_id, "content": content}
)
```

## Error Handling

**CRITICAL**: All backends raise exceptions on failure (no silent errors).

```python
# Good: Raises RuntimeError on failure
try:
    await backend.awrite(path, content)
except RuntimeError as e:
    logger.error(f"Write failed: {e}")
    # Handle error explicitly
```

## Migration Guide

### From Direct S3Backend

```python
# Before
s3_backend = S3Backend(...)
s3_backend.write("/raw/session/file.md", content)

# After
# Just use write_file with the right path prefix
write_file("/data/raw/dumps/session/file.md", content)
```

### From Direct PostgresVectorStore

```python
# Before
postgres_store = PostgresVectorStore(...)
await postgres_store.add(artifact)

# After
# Just use write_file with /data/insights/ prefix
write_file("/data/insights/session/findings/file.md", content)
```

### From Direct Neo4jStore

```python
# Before
neo4j_store = Neo4jStore(...)
await neo4j_store.add(entity_artifact)

# After
# Just use write_file with /data/graph/ prefix
write_file("/data/graph/session/entities/EntityName", "Type|Description")
```

## Testing

```python
import pytest
from src.extensions.features.tools.context_engine.storage.unified_memory_router import (
    create_unified_memory_router,
    PostgresBackend,
    Neo4jBackend,
)

@pytest.mark.asyncio
async def test_unified_router_routes_correctly():
    """Test that paths are routed to correct backends."""
    from deepagents.backends import CompositeBackend

    router = create_unified_memory_router(
        runtime=mock_runtime,
        include_s3=True,
        include_postgres=True,
        include_neo4j=True,
    )

    assert isinstance(router, CompositeBackend)

    # Check routes exist
    routes = router.routes
    assert "/data/raw/dumps/" in routes
    assert "/data/raw/web/" in routes
    assert "/data/insights/" in routes
    assert "/data/graph/" in routes
    assert "/memories/" in routes
```

## Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           DEEP RESEARCH GRAPH           │
                    │                                         │
                    │  Boss → Architects → Flash → Research   │
                    │   ↓         ↓          ↓         ↓      │
                    │  write_file() calls throughout          │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │      UNIFIED MEMORY ROUTER              │
                    │      (CompositeBackend)                 │
                    │                                         │
                    │  Path-based routing + validation        │
                    └─────────────────────────────────────────┘
                          │         │         │         │
                          ▼         ▼         ▼         ▼
                    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
                    │   S3    │ │Postgres │ │  Neo4j  │ │  Store  │
                    │ (Raw)   │ │(Vector) │ │ (Graph) │ │(Search) │
                    └─────────┘ └─────────┘ └─────────┘ └─────────┘
                          │
                          ▼ (async)
                    ┌──────────────────┐
                    │ Vectorization    │
                    │ Background Worker│
                    └──────────────────┘
```
