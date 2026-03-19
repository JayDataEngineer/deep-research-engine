# Storage Backends Integration Guide

This guide explains how to integrate the storage backends (PostgreSQL+pgvector, Neo4j, MongoDB, MinIO) with the shared Docker infrastructure.

## Overview

The Deep Research Engine supports four storage backends:

| Backend | Purpose | Docker Service | Port |
|---------|---------|----------------|------|
| **PostgreSQL+pgvector** | Vectors, embeddings, structured data | `shared-postgres` | 25432 |
| **Neo4j** | Knowledge graph, relationships | `shared-neo4j` | 37687 (Bolt), http://neo4j.local |
| **MongoDB** | Documents, media metadata | `shared-mongodb` | 37017 |
| **MinIO** | Object storage, files | `shared-minio` | http://s3.local |

## Prerequisites

1. **Shared Docker Infrastructure** must be running:
   ```bash
   cd ~/Documents/programs/shared-docker-infra
   make db-up
   ```

2. **Install storage dependencies**:
   ```bash
   cd ~/Documents/programs/deep-research-engine
   uv sync --group dev
   # Or install specific backends:
   uv sync --extra all-storage
   ```

## Starting the Database Services

Start all database services from the shared-docker-infra directory:

```bash
cd ~/Documents/programs/shared-docker-infra

# Start all databases
make db-up

# Or start individually
make up-postgres
make up-neo4j
make up-mongodb
make up-minio
```

Verify services are running:

```bash
docker ps | grep shared-
```

## Configuration

### Environment Variables

Copy the example environment file and configure:

```bash
cd ~/Documents/programs/deep-research-engine
cp .env.example .env
```

Update the following variables:

```bash
# PostgreSQL (pgvector)
CONTEXT_POSTGRES_URL=postgresql://postgres:e2e-test-postgres-password@localhost:25432/shared_db

# Neo4j
NEO4J_URI=bolt://localhost:37687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=e2e-test-neo4j-password

# MongoDB
MONGODB_URI=mongodb://admin:e2e-test-mongodb-password@localhost:37017

# MinIO/S3
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=e2e-test-minio-password
MINIO_REGION=us-east-1
```

### YAML Configuration

Create a workflow configuration file:

```yaml
# configs/storage-example.yaml
workflow:
  name: storage-example
  version: "1.0"

storage:
  # PostgreSQL for vectors and embeddings
  postgres:
    url: "postgresql://postgres:e2e-test-postgres-password@localhost:25432/shared_db"
    pool_size: 10
    embedding_dimension: 768

  # Neo4j for knowledge graph
  neo4j:
    uri: "bolt://localhost:37687"
    username: "neo4j"
    password: "e2e-test-neo4j-password"
    database: "neo4j"

  # MongoDB for documents
  mongodb:
    uri: "mongodb://admin:e2e-test-mongodb-password@localhost:37017"
    database: "deep_research"

  # MinIO for object storage
  minio:
    endpoint: "localhost:9000"
    access_key: "minioadmin"
    secret_key: "e2e-test-minio-password"
    secure: false
    region: "us-east-1"
    default_bucket: "deep-research"
```

## Usage Examples

### PostgreSQL+pgvector

```python
import asyncio
from deep_research_engine.storage import PostgresStorage

async def example_postgres():
    # Initialize
    storage = PostgresStorage({
        "url": "postgresql://postgres:e2e-test-postgres-password@localhost:25432/shared_db",
        "embedding_dimension": 768,
    })
    await storage.initialize()

    # Store with embedding
    from pgvector import Vector
    result = await storage.store({
        "id": "doc-1",
        "content": "This is a test document",
        "embedding": [0.1] * 768,  # Your embedding vector
        "metadata": {"source": "test"},
    }, namespace="documents")
    
    print(f"Stored: {result.id}")

    # Vector similarity search
    results = await storage.query(
        query=",".join(map(str, [0.1] * 768)),
        namespace="documents",
        limit=5,
        embedding_query=True,
        similarity_threshold=0.7,
    )
    
    for doc in results:
        print(f"Similarity: {doc['similarity']:.3f} - {doc['content']}")

    await storage.cleanup()

asyncio.run(example_postgres())
```

### Neo4j

```python
import asyncio
from deep_research_engine.storage import Neo4jStorage

async def example_neo4j():
    # Initialize
    storage = Neo4jStorage({
        "uri": "bolt://localhost:37687",
        "username": "neo4j",
        "password": "e2e-test-neo4j-password",
    })
    await storage.initialize()

    # Store node with relationships
    result = await storage.store({
        "id": "entity-1",
        "labels": ["Person", "Researcher"],
        "name": "John Doe",
        "affiliation": "University",
        "relationships": [
            {
                "target_id": "entity-2",
                "type": "COLLABORATED_WITH",
                "properties": {"since": "2020"},
            }
        ],
    }, namespace="knowledge-graph")

    # Query with Cypher
    results = await storage.query(
        "MATCH (n:Person {namespace: $namespace}) RETURN n LIMIT $limit",
        namespace="knowledge-graph",
        limit=10,
        cypher=True,
    )

    # Find path between nodes
    path = await storage.find_path(
        start_id="entity-1",
        end_id="entity-2",
        namespace="knowledge-graph",
        max_depth=5,
    )

    await storage.cleanup()

asyncio.run(example_neo4j())
```

### MongoDB

```python
import asyncio
from deep_research_engine.storage import MongoDBStorage

async def example_mongodb():
    # Initialize
    storage = MongoDBStorage({
        "uri": "mongodb://admin:e2e-test-mongodb-password@localhost:37017",
        "database": "deep_research",
    })
    await storage.initialize()

    # Store document
    result = await storage.store({
        "id": "media-1",
        "type": "pdf",
        "title": "Research Paper",
        "content": "Full text content...",
        "metadata": {
            "authors": ["John Doe"],
            "year": 2024,
            "pages": 10,
        },
    }, namespace="media")

    # Store binary file
    with open("document.pdf", "rb") as f:
        file_result = await storage.store_file(
            file_data=f.read(),
            filename="document.pdf",
            content_type="application/pdf",
            namespace="files",
        )

    # Query
    results = await storage.query(
        query="Research",
        namespace="media",
        limit=10,
    )

    await storage.cleanup()

asyncio.run(example_mongodb())
```

### MinIO

```python
import asyncio
from deep_research_engine.storage import MinIOStorage

async def example_minio():
    # Initialize
    storage = MinIOStorage({
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "e2e-test-minio-password",
        "secure": False,
        "default_bucket": "deep-research",
    })
    await storage.initialize()

    # Store JSON object
    result = await storage.store({
        "id": "config-1",
        "settings": {"key": "value"},
    }, namespace="configs")

    # Store file
    with open("document.pdf", "rb") as f:
        file_result = await storage.store_file(
            file_data=f.read(),
            filename="document.pdf",
            namespace="documents",
            content_type="application/pdf",
        )

    # Generate presigned URL for secure access
    url = await storage.get_presigned_url(
        key="documents/document.pdf",
        expires_in=3600,  # 1 hour
    )
    print(f"Download URL: {url}")

    await storage.cleanup()

asyncio.run(example_minio())
```

## Accessing Web Interfaces

### Neo4j Browser

Access the Neo4j Browser UI:
- **URL**: http://neo4j.local
- **Connection**: `bolt://localhost:37687`
- **Username**: `neo4j`
- **Password**: Check `.env` or use `e2e-test-neo4j-password`

### MinIO Console

Access the MinIO web console:
- **URL**: http://s3.local
- **Username**: `minioadmin`
- **Password**: Check `.env` or use `e2e-test-minio-password`

## Storage Manager

Use the `StorageManager` to manage multiple backends:

```python
import asyncio
from deep_research_engine.storage import StorageManager

async def example_manager():
    manager = StorageManager()

    # Initialize multiple backends
    await manager.initialize("postgres", {
        "url": "postgresql://postgres:e2e-test-postgres-password@localhost:25432/shared_db",
    })
    await manager.initialize("neo4j", {
        "uri": "bolt://localhost:37687",
        "username": "neo4j",
        "password": "e2e-test-neo4j-password",
    })

    # Store in different backends
    postgres_id = await manager.store(
        "postgres",
        {"content": "Vector data", "embedding": [0.1] * 768},
        namespace="vectors",
    )
    
    neo4j_id = await manager.store(
        "neo4j",
        {
            "id": "entity-1",
            "labels": ["Concept"],
            "name": "Machine Learning",
        },
        namespace="graph",
    )

    # Retrieve
    data = await manager.retrieve("postgres", postgres_id, namespace="vectors")

    # Cleanup
    await manager.cleanup()

asyncio.run(example_manager())
```

## Troubleshooting

### Connection Issues

1. **Check if services are running**:
   ```bash
   docker ps | grep shared-
   ```

2. **Check service logs**:
   ```bash
   docker logs shared-postgres
   docker logs shared-neo4j
   docker logs shared-mongodb
   docker logs shared-minio
   ```

3. **Verify ports**:
   ```bash
   netstat -tlnp | grep -E '25432|37687|37017|9000'
   ```

### Common Issues

**PostgreSQL/pgvector**:
- Error: `extension "vector" does not exist`
  - Solution: Ensure you're using `pgvector/pgvector:pg16` image

**Neo4j**:
- Authentication failed
  - Solution: Check password in `.env` file matches Neo4j configuration

**MongoDB**:
- Connection timeout
  - Solution: Verify MongoDB URI includes correct credentials

**MinIO**:
- Bucket not found
  - Solution: Default bucket is created on initialization, or create manually via console

## Best Practices

1. **Use appropriate backend for each data type**:
   - PostgreSQL+pgvector: Embeddings, vectors, structured data
   - Neo4j: Relationships, knowledge graphs
   - MongoDB: Documents, media metadata
   - MinIO: Large files, binary data

2. **Namespace isolation**: Use namespaces to separate data by workflow or tenant

3. **Connection pooling**: All backends support connection pooling - configure `pool_size` appropriately

4. **Cleanup**: Always call `cleanup()` or use context managers to release resources

5. **Error handling**: Wrap storage operations in try-except blocks

## Next Steps

- See `docs/workers/` for worker implementations that use storage
- See `configs/` for example workflow configurations
- See `reference/langchain/docs/` for LangChain vector store integration
