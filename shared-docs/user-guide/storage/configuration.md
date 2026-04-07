# Storage Backends Configuration Reference

This document provides configuration examples for all storage backends.

## Quick Start Configuration

Minimal configuration for each backend:

```yaml
# configs/storage-minimal.yaml
workflow:
  name: storage-example
  version: "1.0"

storage:
  # Use memory storage (default, no configuration needed)
  default: memory
```

## Full Configuration Example

Complete configuration with all backends:

```yaml
# configs/storage-full.yaml
workflow:
  name: full-storage-example
  version: "1.0"
  description: "Example using all storage backends"

storage:
  # Default backend for general storage
  default: postgres

  # PostgreSQL for vectors and structured data
  postgres:
    url: "postgresql://postgres:${POSTGRES_PASSWORD}@localhost:25432/shared_db"
    pool_size: 10
    embedding_dimension: 768
    max_overflow: 5
    pool_timeout: 30

  # Neo4j for knowledge graph
  neo4j:
    uri: "bolt://localhost:37687"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"
    database: "neo4j"
    max_connection_pool_size: 50

  # MongoDB for documents and media
  mongodb:
    uri: "mongodb://${MONGO_USERNAME}:${MONGO_PASSWORD}@localhost:37017"
    database: "deep_research"
    max_pool_size: 50
    server_selection_timeout_ms: 5000

  # MinIO for object storage
  minio:
    endpoint: "localhost:9000"
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
    secure: false
    region: "us-east-1"
    default_bucket: "deep-research"
    max_connections: 50
```

## PostgreSQL Configuration

### Basic

```yaml
storage:
  postgres:
    url: "postgresql://postgres:password@localhost:25432/shared_db"
```

### Advanced

```yaml
storage:
  postgres:
    # Connection
    url: "postgresql://postgres:password@localhost:25432/shared_db"
    
    # Pool settings
    pool_size: 10              # Number of connections in pool
    max_overflow: 5            # Extra connections allowed
    pool_timeout: 30           # Seconds to wait for connection
    pool_recycle: 1800         # Recycle connections after 30 min
    
    # Vector settings
    embedding_dimension: 768   # Dimension for pgvector
    vector_index_type: "ivfflat"  # ivfflat or hnsw
    
    # Performance
    batch_size: 100            # Items per batch insert
    commit_every: 50           # Commit after N operations
```

### Environment Variables

```bash
# .env
POSTGRES_URL=postgresql://postgres:password@localhost:25432/shared_db
POSTGRES_POOL_SIZE=10
POSTGRES_EMBEDDING_DIM=768
```

```yaml
storage:
  postgres:
    url: "${POSTGRES_URL}"
    pool_size: "${POSTGRES_POOL_SIZE:10}"
    embedding_dimension: "${POSTGRES_EMBEDDING_DIM:768}"
```

## Neo4j Configuration

### Basic

```yaml
storage:
  neo4j:
    uri: "bolt://localhost:37687"
    username: "neo4j"
    password: "password"
```

### Advanced

```yaml
storage:
  neo4j:
    # Connection
    uri: "bolt://localhost:37687"
    username: "neo4j"
    password: "password"
    database: "neo4j"         # Database name (Neo4j 4.0+)
    
    # Pool settings
    max_connection_pool_size: 50
    connection_timeout: 30.0   # Seconds
    max_transaction_retry: 3
    
    # Performance
    fetch_size: 1000           # Records per fetch
    result_timeout: 60.0       # Seconds
```

### Neo4j Aura (Cloud)

```yaml
storage:
  neo4j:
    uri: "neo4j+s://your-instance.databases.neo4j.io"
    username: "neo4j"
    password: "your-aura-password"
    database: "neo4j"
```

## MongoDB Configuration

### Basic

```yaml
storage:
  mongodb:
    uri: "mongodb://localhost:37017"
    database: "deep_research"
```

### With Authentication

```yaml
storage:
  mongodb:
    uri: "mongodb://admin:password@localhost:37017"
    database: "deep_research"
    username: "admin"
    password: "password"
```

### Advanced

```yaml
storage:
  mongodb:
    # Connection
    uri: "mongodb://admin:password@localhost:37017"
    database: "deep_research"
    
    # Pool settings
    max_pool_size: 50
    min_pool_size: 10
    max_idle_time_ms: 300000
    server_selection_timeout_ms: 5000
    
    # Replica set (if applicable)
    replica_set: "rs0"
    read_preference: "secondaryPreferred"
    write_concern: "majority"
    
    # GridFS
    gridfs_bucket: "fs"
    gridfs_chunk_size: 261120  # 255KB
```

### MongoDB Atlas (Cloud)

```yaml
storage:
  mongodb:
    uri: "mongodb+srv://username:password@cluster.mongodb.net/deep_research?retryWrites=true&w=majority"
    database: "deep_research"
```

## MinIO Configuration

### Basic

```yaml
storage:
  minio:
    endpoint: "localhost:9000"
    access_key: "minioadmin"
    secret_key: "minioadmin"
    secure: false
    default_bucket: "deep-research"
```

### Advanced

```yaml
storage:
  minio:
    # Connection
    endpoint: "localhost:9000"
    access_key: "minioadmin"
    secret_key: "minioadmin"
    secure: false              # Use HTTPS if true
    region: "us-east-1"
    
    # Bucket settings
    default_bucket: "deep-research"
    auto_create_bucket: true
    bucket_versioning: false
    
    # Performance
    max_connections: 50
    multipart_threshold: 8388608    # 8MB
    multipart_chunksize: 8388608    # 8MB
    max_concurrency: 10
    
    # Retries
    max_retries: 3
    retry_mode: "standard"
```

### AWS S3

```yaml
storage:
  minio:
    endpoint: "s3.us-east-1.amazonaws.com"
    access_key: "${AWS_ACCESS_KEY_ID}"
    secret_key: "${AWS_SECRET_ACCESS_KEY}"
    secure: true
    region: "us-east-1"
    default_bucket: "my-deep-research-bucket"
```

### Cloudflare R2

```yaml
storage:
  minio:
    endpoint: "https://<account-id>.r2.cloudflarestorage.com"
    access_key: "${R2_ACCESS_KEY_ID}"
    secret_key: "${R2_SECRET_ACCESS_KEY}"
    secure: true
    region: "auto"
    default_bucket: "deep-research"
```

## Multi-Backend Configuration

Use different backends for different purposes:

```yaml
# configs/storage-multi-backend.yaml
workflow:
  name: multi-backend-example
  version: "1.0"

storage:
  # Default for general storage
  default: postgres
  
  # Vectors and embeddings
  vectors: postgres
  
  # Knowledge graph
  graph: neo4j
  
  # Media files metadata
  media: mongodb
  
  # Binary files
  files: minio
```

## Worker-Specific Configuration

Configure storage per worker:

```yaml
# configs/rag-with-storage.yaml
workflow:
  name: rag-pipeline
  version: "1.0"

storage:
  default: postgres
  postgres:
    url: "postgresql://postgres:password@localhost:25432/shared_db"
    embedding_dimension: 768

workers:
  - type: web_researcher
    storage:
      findings: postgres
  
  - type: chunk_processor
    storage:
      chunks: postgres
      vectors: postgres
  
  - type: multimodal
    storage:
      media_metadata: mongodb
      files: minio
```

## Environment Variable Templates

### Development

```bash
# .env.development
POSTGRES_URL=postgresql://postgres:postgres@localhost:25432/shared_db
NEO4J_URI=bolt://localhost:37687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=dev-password
MONGODB_URI=mongodb://localhost:27017
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### Production

```bash
# .env.production
POSTGRES_URL=postgresql://user:secure-password@db.example.com:5432/deep_research
NEO4J_URI=neo4j+s://cluster.example.com
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=secure-neo4j-password
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/deep_research
MINIO_ENDPOINT=s3.amazonaws.com
MINIO_ACCESS_KEY=${AWS_ACCESS_KEY_ID}
MINIO_SECRET_KEY=${AWS_SECRET_ACCESS_KEY}
MINIO_REGION=us-east-1
```

## Testing Configuration

Minimal configuration for testing:

```yaml
# configs/storage-test.yaml
workflow:
  name: test-workflow
  version: "1.0"

storage:
  # Use in-memory storage for fast tests
  default: memory
  
  # Or use test instances
  postgres:
    url: "postgresql://postgres:postgres@localhost:5432/test_db"
  
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "test"
  
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "test_db"
  
  minio:
    endpoint: "localhost:9000"
    access_key: "minioadmin"
    secret_key: "minioadmin"
    default_bucket: "test-bucket"
```

## Troubleshooting Configuration

### Connection String Format

```yaml
# Correct formats:
postgresql://user:pass@host:port/database
mongodb://user:pass@host:port/database
bolt://host:port
http://host:port
```

### Common Issues

**PostgreSQL**:
```yaml
# Wrong - missing database
url: "postgresql://user:pass@host:port"

# Correct
url: "postgresql://user:pass@host:port/database"
```

**MongoDB**:
```yaml
# Wrong - missing protocol
uri: "localhost:27017/database"

# Correct
uri: "mongodb://localhost:27017/database"
```

**MinIO**:
```yaml
# Wrong - includes protocol in endpoint
endpoint: "http://localhost:9000"

# Correct - protocol added automatically based on 'secure' setting
endpoint: "localhost:9000"
secure: false
```

## Next Steps

- See `docs/storage/integration.md` for Docker integration
- See `docs/workers/` for worker implementations
- See `reference/langchain/docs/` for LangChain integration
