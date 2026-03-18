# Storage Backends Reference

Storage backends are pluggable. Enable only what you need in config.

## Base Interface

```python
from deep_research_engine.storage import StorageBackend, StorageRegistry

@StorageRegistry.register("backend_name")
class MyStorage(StorageBackend):
    backend_name = "backend_name"
    description = "Description"
    
    async def initialize(self) -> None:
        # Setup connections, pools, tables
        pass
    
    async def cleanup(self) -> None:
        # Close connections
        pass
    
    async def store(self, data, namespace="default") -> StorageResult:
        # Store data
        pass
    
    async def retrieve(self, id, namespace="default"):
        # Retrieve data
        pass
    
    async def query(self, query, namespace="default", limit=10):
        # Optional: semantic search
        pass
```

## Built-in Backends

### InMemoryStorage

**Purpose**: Default fallback, ephemeral storage

**Config**:
```yaml
storage:
  memory:
    enabled: true
    max_size_mb: 500
```

**Use cases**:
- PDF summarizer (no persistent storage needed)
- Testing
- Quick prototypes

---

### PostgresStorage (TODO)

**Purpose**: PostgreSQL + pgvector for semantic search

**Config**:
```yaml
storage:
  postgres:
    enabled: true
    connection: "${POSTGRES_URL}"
    tables:
      artifacts: artifacts
      embeddings: artifact_embeddings
```

**Features**:
- Artifact storage with deduplication
- Vector embeddings for semantic search
- Full-text search

**Implementation**:
```python
@StorageRegistry.register("postgres")
class PostgresStorage(StorageBackend):
    async def initialize(self):
        import asyncpg
        from pgvector.asyncpg import register_vector
        
        self.pool = await asyncpg.create_pool(
            self.config.get("connection"),
            min_size=5,
            max_size=20,
        )
        
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await self._create_tables()
    
    async def store(self, data, namespace="default"):
        import json
        resource_id = str(uuid.uuid4())
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO artifacts (id, namespace, data, created_at)
                VALUES ($1, $2, $3, $4)
                """,
                resource_id, namespace, json.dumps(data), datetime.now(),
            )
            
            if "embedding" in data:
                await conn.execute(
                    """
                    INSERT INTO artifact_embeddings (artifact_id, embedding)
                    VALUES ($1, $2)
                    """,
                    resource_id, data["embedding"],
                )
        
        return StorageResult(success=True, id=resource_id)
    
    async def query(self, query, namespace="default", limit=10):
        # Semantic search with pgvector
        embedding = await self.embed_query(query)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT a.data, 1 - (e.embedding <=> $1::vector) as similarity
                FROM artifacts a
                JOIN artifact_embeddings e ON a.id = e.artifact_id
                WHERE a.namespace = $2
                ORDER BY e.embedding <=> $1::vector
                LIMIT $3
                """,
                embedding, namespace, limit,
            )
            return [dict(row) for row in rows]
```

---

### MongoDBStorage (TODO)

**Purpose**: Media files and metadata

**Config**:
```yaml
storage:
  mongodb:
    enabled: true
    connection: "${MONGO_URL}"
    collections:
      media: media_files
      metadata: media_metadata
```

**Use cases**:
- Telegram image/video storage
- Large document storage

---

### Neo4jStorage (TODO)

**Purpose**: Knowledge graph

**Config**:
```yaml
storage:
  neo4j:
    enabled: true
    uri: "${NEO4J_URI}"
    username: neo4j
    password: "${NEO4J_PASSWORD}"
```

**Use cases**:
- Person/topic relationships
- Entity graph traversal

---

### MinIOStorage (TODO)

**Purpose**: S3-compatible object storage

**Config**:
```yaml
storage:
  minio:
    enabled: true
    endpoint: "${MINIO_ENDPOINT}"
    bucket: deep-research
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
```

**Use cases**:
- Large file storage
- Media file archival

---

## StorageManager

Use multiple backends:

```python
from deep_research_engine.storage import StorageManager

manager = StorageManager()

# Initialize backends
await manager.initialize("postgres", config)
await manager.initialize("neo4j", config)

# Store in all
ids = await manager.store_all(data, namespace="artifacts")
# Returns: {"postgres": "id1", "neo4j": "id2"}

# Store in specific
id = await manager.store("postgres", data, namespace="artifacts")

# Retrieve
data = await manager.retrieve("postgres", id, namespace="artifacts")

# Query
results = await manager.query("postgres", "search query", limit=10)

# Cleanup
await manager.cleanup()
```

---

## Configuration Examples

### Minimal (Memory only)

```yaml
storage:
  memory:
    enabled: true
```

### RAG Research

```yaml
storage:
  postgres:
    enabled: true
    connection: "${POSTGRES_URL}"
  neo4j:
    enabled: true
    uri: "${NEO4J_URI}"
  memory:
    enabled: true
```

### Telegram Ingestion

```yaml
storage:
  postgres:
    enabled: true
  mongodb:
    enabled: true
  neo4j:
    enabled: true
  minio:
    enabled: true
  memory:
    enabled: true
```

---

## Testing

```python
import pytest
from deep_research_engine.storage import StorageRegistry, InMemoryStorage

@pytest.mark.asyncio
async def test_memory_storage():
    config = {"max_size_mb": 100}
    storage = InMemoryStorage(config)
    
    await storage.initialize()
    
    # Store
    result = await storage.store(
        {"key": "value"},
        namespace="test",
    )
    assert result.success
    assert result.id
    
    # Retrieve
    data = await storage.retrieve(result.id, namespace="test")
    assert data["key"] == "value"
    
    # List
    items = await storage.list_items(namespace="test")
    assert len(items) == 1
    
    await storage.cleanup()
```
