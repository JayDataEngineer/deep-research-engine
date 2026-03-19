# Storage Backends Documentation

Complete documentation for the Deep Research Engine storage backends.

## Overview

The Deep Research Engine supports four production-ready storage backends, each optimized for different use cases:

| Backend | Best For | Docker Service |
|---------|----------|----------------|
| **PostgreSQL+pgvector** | Vector embeddings, semantic search, structured data | `shared-postgres` |
| **Neo4j** | Knowledge graphs, relationships, connected data | `shared-neo4j` |
| **MongoDB** | Documents, media metadata, flexible schemas | `shared-mongodb` |
| **MinIO/S3** | Large files, binary data, object storage | `shared-minio` |

## Quick Start

### 1. Start Docker Infrastructure

```bash
cd ~/Documents/programs/shared-docker-infra
make db-up
```

### 2. Install Dependencies

```bash
cd ~/Documents/programs/deep-research-engine
uv sync --extra all-storage
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Docker service credentials
```

### 4. Run Examples

```bash
uv run python -m examples.storage_examples
```

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [Integration Guide](integration.md) | Docker integration, configuration, and usage examples |
| [Configuration Reference](configuration.md) | Complete configuration options for all backends |

### API Reference

All backends implement the `StorageBackend` interface:

```python
from deep_research_engine.storage import PostgresStorage

storage = PostgresStorage(config)
await storage.initialize()

# Store data
result = await storage.store({"key": "value"}, namespace="default")

# Retrieve data
data = await storage.retrieve(result.id, namespace="default")

# Query data
results = await storage.query("search term", namespace="default", limit=10)

# Cleanup
await storage.cleanup()
```

## Backend Selection Guide

### Choose PostgreSQL+pgvector when:
- You need vector similarity search (RAG, embeddings)
- Your data has structured schema
- You need SQL queries and joins
- You want ACID transactions

### Choose Neo4j when:
- Your data is highly connected
- You need to traverse relationships
- You're building a knowledge graph
- You need graph algorithms (pathfinding, centrality)

### Choose MongoDB when:
- Your schema is flexible or evolving
- You store JSON documents
- You need horizontal scalability
- You store media file metadata

### Choose MinIO when:
- You store large files (PDFs, images, audio, video)
- You need S3 compatibility
- You want presigned URLs for secure access
- You need object lifecycle management

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Deep Research Engine                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Workers    │  │    Boss      │  │   Critic     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         └─────────────────┼─────────────────┘           │
│                           │                             │
│                  ┌────────▼────────┐                    │
│                  │ StorageManager  │                    │
│                  └────────┬────────┘                    │
│                           │                             │
└───────────────────────────┼─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼───────┐  ┌───────▼────────┐
│   PostgreSQL   │  │    Neo4j     │  │    MongoDB     │
│   + pgvector   │  │  Knowledge   │  │   Documents    │
│   Vectors      │  │    Graph     │  │   + GridFS     │
└────────────────┘  └──────────────┘  └────────────────┘
                            │
                   ┌────────▼────────┐
                   │      MinIO      │
                   │  Object Storage │
                   └─────────────────┘
```

## Common Patterns

### RAG Pipeline (PostgreSQL)

```python
# Store document with embedding
await storage.store({
    "id": "doc-1",
    "content": "Document text",
    "embedding": embedding_vector,
}, namespace="documents")

# Semantic search
results = await storage.query(
    query=",".join(map(str, query_embedding)),
    namespace="documents",
    limit=5,
    embedding_query=True,
)
```

### Knowledge Graph (Neo4j)

```python
# Create entity
await storage.store({
    "id": "entity-1",
    "labels": ["Person"],
    "name": "John Doe",
    "relationships": [
        {"target_id": "entity-2", "type": "KNOWS"}
    ]
}, namespace="graph")

# Find connections
path = await storage.find_path("entity-1", "entity-2")
```

### Media Processing (MongoDB + MinIO)

```python
# Store file in MinIO
file_result = await minio.store_file(
    file_data=file_bytes,
    filename="document.pdf",
    namespace="files",
)

# Store metadata in MongoDB
await mongo.store({
    "id": file_result.id,
    "type": "pdf",
    "minio_key": file_result.metadata["key"],
    "metadata": {"pages": 10},
}, namespace="media")
```

## Testing

Run the example module to test all backends:

```bash
uv run python -m examples.storage_examples
```

## Troubleshooting

### Connection Issues

1. **Check Docker services**:
   ```bash
   docker ps | grep shared-
   ```

2. **Check service logs**:
   ```bash
   docker logs shared-postgres
   docker logs shared-neo4j
   ```

3. **Verify ports**:
   ```bash
   netstat -tlnp | grep -E '25432|37687|37017|9000'
   ```

### Common Errors

**`StorageInitializationError`**:
- Check connection string format
- Verify credentials in `.env`
- Ensure Docker service is running

**`Connection refused`**:
- Docker service may not be started
- Check firewall settings
- Verify port mapping

## Performance Tips

1. **Use connection pooling** - All backends support pooling
2. **Batch operations** - Use `store_batch()` when available
3. **Index frequently queried fields** - Create custom indexes
4. **Use appropriate namespaces** - Isolate data by workflow
5. **Clean up resources** - Always call `cleanup()` or use context managers

## Security

- Store credentials in environment variables
- Use TLS/SSL for production connections
- Enable authentication on all services
- Use presigned URLs for secure file access
- Implement proper access control at application level

## Contributing

To add a new storage backend:

1. Create `src/deep_research_engine/storage/backends/new_backend.py`
2. Implement `StorageBackend` interface
3. Register with `@StorageRegistry.register("name")`
4. Add to `__init__.py` exports
5. Write tests and documentation

## Related Documentation

- [Worker Implementations](../workers/) - Workers that use storage
- [Workflow Configuration](../workflow-configs/) - Configuring storage in workflows
- [Shared Docker Infrastructure](../../shared-docker-infra/docs/) - Docker setup

## Support

For issues or questions:
- Check existing issues on GitHub
- Review the [Integration Guide](integration.md)
- See [Configuration Reference](configuration.md)
- Consult Docker infrastructure docs
