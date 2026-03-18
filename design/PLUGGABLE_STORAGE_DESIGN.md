# Pluggable Storage Backend Design

## Problem

The original Context Engine required ALL backends (Postgres + Neo4j + MinIO) to be running. This is overkill for simple workflows like PDF summarization that just need to write a markdown file.

## Solution: Pluggable Storage Registry

Each storage backend is:
1. **Optional** - Enable only what you need in YAML config
2. **Independent** - No cross-dependencies
3. **Uniform interface** - Same ABC for all backends
4. **Graceful degradation** - Engine works with just InMemoryStorage

---

## Storage Backend ABC

```python
"""StorageBackend - Abstract Base Class for all storage backends.

Design Principles:
1. Uniform interface across all backends
2. Backend-specific optimizations allowed
3. Graceful fallbacks when backend unavailable
4. No cross-backend dependencies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass


@dataclass
class StorageResult:
    """Result from a storage operation."""
    success: bool
    id: Optional[str] = None              # Resource ID (for retrieval)
    metadata: Dict[str, Any] = None       # Backend-specific metadata
    error: Optional[str] = None


class StorageBackend(ABC):
    """Abstract base class for storage backends.
    
    All backends (Postgres, MongoDB, Neo4j, MinIO, Memory)
    implement this interface.
    """
    
    # Class-level metadata
    backend_name: str = "base"
    description: str = "Base storage backend"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with configuration.
        
        Args:
            config: Backend configuration from YAML
        """
        self.config = config
        self.initialized = False
        self.health_status = "unknown"
    
    # -------------------------------------------------------------------------
    # LIFECYCLE METHODS
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize backend (connections, pools, etc.).
        
        Raises:
            StorageInitializationError: If initialization fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources (close connections, etc.)."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy.
        
        Returns:
            True if backend is operational
        """
        pass
    
    # -------------------------------------------------------------------------
    # CORE OPERATIONS (All backends support these)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        """Store data in backend.
        
        Args:
            data: Data to store
            namespace: Logical namespace (e.g., "artifacts", "media")
            
        Returns:
            StorageResult with ID for retrieval
        """
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve data by ID.
        
        Args:
            id: Resource ID
            namespace: Logical namespace
            
        Returns:
            Data dict or None if not found
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        """Delete data by ID.
        
        Args:
            id: Resource ID
            namespace: Logical namespace
            
        Returns:
            True if deleted successfully
        """
        pass
    
    # -------------------------------------------------------------------------
    # QUERY OPERATIONS (Optional - backends can implement)
    # -------------------------------------------------------------------------
    
    async def query(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query data (semantic or keyword).
        
        Default implementation returns empty list.
        Backends can override to support querying.
        
        Args:
            query: Query string
            namespace: Logical namespace
            limit: Max results to return
            
        Returns:
            List of matching results
        """
        return []
    
    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List items in namespace.
        
        Args:
            namespace: Logical namespace
            limit: Max items to return
            
        Returns:
            List of item metadata (not full data)
        """
        return []
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message with backend context."""
        from .utils.logging import get_logger
        logger = get_logger(f"storage.{self.backend_name}")
        
        log_method = getattr(logger, level, logger.info)
        log_method(
            f"[{self.backend_name}] {message}",
            backend=self.backend_name,
            **kwargs
        )


# =============================================================================
# STORAGE REGISTRY (Pluggable Backends)
# =============================================================================

class StorageRegistry:
    """Registry for storage backends.
    
    Backends register themselves, and the engine looks them up by name.
    
    Usage:
        # In your backend module
        from deep_research_engine import StorageRegistry, StorageBackend
        
        @StorageRegistry.register("postgres")
        class PostgresStorage(StorageBackend):
            backend_name = "postgres"
            description = "PostgreSQL + pgvector storage"
            
            async def store(self, data, namespace):
                # Implementation
                pass
        
        # In engine
        backend_class = StorageRegistry.get("postgres")
        backend = backend_class(config)
    """
    
    _backends: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a backend."""
        def decorator(backend_class: type) -> type:
            cls._backends[name] = backend_class
            backend_class.backend_name = name
            return backend_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get backend class by name."""
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise StorageNotFoundError(
                f"Backend '{name}' not found. Available: {available}"
            )
        return cls._backends[name]
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backends."""
        return list(cls._backends.keys())


# =============================================================================
# STORAGE MANAGER (Orchestrates Multiple Backends)
# =============================================================================

class StorageManager:
    """Manage multiple storage backends.
    
    Usage:
        manager = StorageManager()
        await manager.initialize("postgres", config)
        await manager.initialize("neo4j", config)
        
        # Store in all backends
        await manager.store_all(data)
        
        # Store in specific backend
        await manager.store("postgres", data)
    """
    
    def __init__(self):
        self.backends: Dict[str, StorageBackend] = {}
        self.initialized = False
    
    async def initialize(
        self,
        backend_name: str,
        config: Dict[str, Any],
    ) -> None:
        """Initialize a storage backend.
        
        Args:
            backend_name: Backend type (e.g., "postgres", "mongodb")
            config: Backend configuration
        """
        backend_class = StorageRegistry.get(backend_name)
        backend = backend_class(config)
        
        await backend.initialize()
        self.backends[backend_name] = backend
        
        self._log(f"Initialized backend: {backend_name}")
    
    async def cleanup(self) -> None:
        """Cleanup all backends."""
        for name, backend in self.backends.items():
            try:
                await backend.cleanup()
                self._log(f"Cleaned up backend: {name}")
            except Exception as e:
                self._log(f"Error cleaning up {name}: {e}", level="error")
    
    async def store(
        self,
        backend_name: str,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> Optional[str]:
        """Store data in specific backend.
        
        Args:
            backend_name: Backend to use
            data: Data to store
            namespace: Logical namespace
            
        Returns:
            Resource ID or None if failed
        """
        if backend_name not in self.backends:
            self._log(f"Backend not initialized: {backend_name}", level="warning")
            return None
        
        backend = self.backends[backend_name]
        result = await backend.store(data, namespace)
        
        if result.success:
            self._log(f"Stored in {backend_name}: {result.id}")
            return result.id
        else:
            self._log(f"Store failed in {backend_name}: {result.error}", level="error")
            return None
    
    async def store_all(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> Dict[str, Optional[str]]:
        """Store data in all initialized backends.
        
        Args:
            data: Data to store
            namespace: Logical namespace
            
        Returns:
            Dict of backend_name → resource_id
        """
        import asyncio
        
        tasks = [
            self.store(name, data, namespace)
            for name in self.backends.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return dict(zip(self.backends.keys(), results))
    
    async def retrieve(
        self,
        backend_name: str,
        id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve data from specific backend."""
        if backend_name not in self.backends:
            return None
        
        backend = self.backends[backend_name]
        return await backend.retrieve(id, namespace)
    
    async def query(
        self,
        backend_name: str,
        query: str,
        namespace: str = "default",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query data in specific backend."""
        if backend_name not in self.backends:
            return []
        
        backend = self.backends[backend_name]
        return await backend.query(query, namespace, limit)
    
    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message."""
        from .utils.logging import get_logger
        logger = get_logger("storage.manager")
        
        log_method = getattr(logger, level, logger.info)
        log_method(message, **kwargs)


# =============================================================================
# BACKEND IMPLEMENTATIONS
# =============================================================================

# Example 1: In-Memory Storage (Always Available)
@StorageRegistry.register("memory")
class InMemoryStorage(StorageBackend):
    """In-memory storage backend.
    
    Default fallback when no other backends are configured.
    Data is lost on restart.
    """
    
    backend_name = "memory"
    description = "In-memory storage (ephemeral)"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._namespaces: Dict[str, set] = {}
    
    async def initialize(self) -> None:
        self.initialized = True
        self.health_status = "healthy"
        self._log("Initialized in-memory storage")
    
    async def cleanup(self) -> None:
        self._store.clear()
        self._namespaces.clear()
        self.initialized = False
    
    async def health_check(self) -> bool:
        return self.initialized
    
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        import uuid
        
        resource_id = str(uuid.uuid4())
        self._store[resource_id] = {
            "data": data,
            "namespace": namespace,
            "created_at": str(datetime.now()),
        }
        
        # Track namespace
        if namespace not in self._namespaces:
            self._namespaces[namespace] = set()
        self._namespaces[namespace].add(resource_id)
        
        self._log(f"Stored {resource_id} in namespace {namespace}")
        
        return StorageResult(
            success=True,
            id=resource_id,
            metadata={"namespace": namespace},
        )
    
    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        item = self._store.get(id)
        if not item:
            return None
        
        if item["namespace"] != namespace:
            return None
        
        return item["data"]
    
    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        if id in self._store:
            del self._store[id]
            if namespace in self._namespaces:
                self._namespaces[namespace].discard(id)
            return True
        return False
    
    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        items = []
        for id in self._namespaces.get(namespace, set()):
            if len(items) >= limit:
                break
            item = self._store.get(id)
            if item:
                items.append({
                    "id": id,
                    "namespace": namespace,
                    "created_at": item["created_at"],
                })
        return items


# Example 2: PostgreSQL Storage
@StorageRegistry.register("postgres")
class PostgresStorage(StorageBackend):
    """PostgreSQL storage with pgvector for semantic search.
    
    Features:
    - Artifact storage with deduplication
    - Vector embeddings for semantic search
    - Full-text search
    """
    
    backend_name = "postgres"
    description = "PostgreSQL + pgvector storage"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.connection_string = config.get("connection")
        self.pool = None
    
    async def initialize(self) -> None:
        import asyncpg
        from pgvector.asyncpg import register_vector
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.config.get("pool_min_size", 5),
                max_size=self.config.get("pool_max_size", 20),
            )
            
            # Register pgvector
            async with self.pool.acquire() as conn:
                await register_vector(conn)
            
            # Create tables if not exist
            await self._create_tables()
            
            self.initialized = True
            self.health_status = "healthy"
            self._log("Initialized PostgreSQL storage")
            
        except Exception as e:
            self._log(f"Failed to initialize PostgreSQL: {e}", level="error")
            raise StorageInitializationError(f"PostgreSQL initialization failed: {e}")
    
    async def cleanup(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.initialized = False
    
    async def health_check(self) -> bool:
        if not self.pool:
            return False
        
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except:
            return False
    
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        import json
        import uuid
        from datetime import datetime
        
        resource_id = str(uuid.uuid4())
        
        try:
            async with self.pool.acquire() as conn:
                # Insert artifact
                await conn.execute(
                    """
                    INSERT INTO artifacts (id, namespace, data, created_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE SET data = $3
                    """,
                    resource_id,
                    namespace,
                    json.dumps(data),
                    datetime.now(),
                )
                
                # Insert embedding if present
                if "embedding" in data:
                    await conn.execute(
                        """
                        INSERT INTO artifact_embeddings (artifact_id, embedding)
                        VALUES ($1, $2)
                        ON CONFLICT (artifact_id) DO UPDATE SET embedding = $2
                        """,
                        resource_id,
                        data["embedding"],
                    )
            
            self._log(f"Stored {resource_id} in PostgreSQL")
            
            return StorageResult(
                success=True,
                id=resource_id,
                metadata={"namespace": namespace},
            )
            
        except Exception as e:
            self._log(f"PostgreSQL store failed: {e}", level="error")
            return StorageResult(
                success=False,
                error=str(e),
            )
    
    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        import json
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM artifacts WHERE id = $1 AND namespace = $2",
                id,
                namespace,
            )
            
            if row:
                return json.loads(row["data"])
            return None
    
    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM artifacts WHERE id = $1 AND namespace = $2",
                id,
                namespace,
            )
            return result == "DELETE 1"
    
    async def query(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Semantic search using pgvector."""
        import numpy as np
        
        # Generate embedding for query
        from .llm import create_embedding
        embedding_model = create_embedding()
        query_embedding = await embedding_model.aembed_query(query)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT a.id, a.data, a.created_at,
                       1 - (e.embedding <=> $1::vector) as similarity
                FROM artifacts a
                JOIN artifact_embeddings e ON a.id = e.artifact_id
                WHERE a.namespace = $2
                ORDER BY e.embedding <=> $1::vector
                LIMIT $3
                """,
                query_embedding,
                namespace,
                limit,
            )
            
            results = []
            for row in rows:
                item = json.loads(row["data"])
                item["_similarity"] = row["similarity"]
                item["_id"] = row["id"]
                results.append(item)
            
            return results
    
    async def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS artifact_embeddings (
                    artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id),
                    embedding vector(1024)
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_artifacts_namespace
                ON artifacts(namespace)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_artifact_embeddings
                ON artifact_embeddings USING ivfflat (embedding vector_cosine_ops)
            """)


# Example 3: MongoDB Storage (Media Files)
@StorageRegistry.register("mongodb")
class MongoDBStorage(StorageBackend):
    """MongoDB storage for media files and metadata."""
    
    backend_name = "mongodb"
    description = "MongoDB storage for media files"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.connection_string = config.get("connection")
        self.client = None
        self.db = None
    
    async def initialize(self) -> None:
        from pymongo import MongoClient
        
        self.client = MongoClient(self.connection_string)
        self.db = self.client.get_default_database()
        
        # Create indexes
        self.db.media.create_index([("namespace", 1)])
        self.db.media.create_index([("metadata.type", 1)])
        
        self.initialized = True
        self.health_status = "healthy"
        self._log("Initialized MongoDB storage")
    
    async def cleanup(self) -> None:
        if self.client:
            self.client.close()
            self.client = None
            self.initialized = False
    
    async def health_check(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except:
            return False
    
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        from bson.objectid import ObjectId
        
        result = self.db.media.insert_one({
            "namespace": namespace,
            "data": data,
            "created_at": datetime.now(),
        })
        
        return StorageResult(
            success=True,
            id=str(result.inserted_id),
            metadata={"namespace": namespace},
        )
    
    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        from bson.objectid import ObjectId
        
        item = self.db.media.find_one({
            "_id": ObjectId(id),
            "namespace": namespace,
        })
        
        if item:
            return item["data"]
        return None
    
    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        from bson.objectid import ObjectId
        
        result = self.db.media.delete_one({
            "_id": ObjectId(id),
            "namespace": namespace,
        })
        return result.deleted_count > 0


# Example 4: Neo4j Storage (Knowledge Graph)
@StorageRegistry.register("neo4j")
class Neo4jStorage(StorageBackend):
    """Neo4j storage for knowledge graphs."""
    
    backend_name = "neo4j"
    description = "Neo4j graph database"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.uri = config.get("uri")
        self.username = config.get("username")
        self.password = config.get("password")
        self.driver = None
    
    async def initialize(self) -> None:
        from neo4j import AsyncGraphDatabase
        
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )
        
        # Verify connection
        async with self.driver.session() as session:
            await session.run("MATCH (n) RETURN count(n)")
        
        self.initialized = True
        self.health_status = "healthy"
        self._log("Initialized Neo4j storage")
    
    async def cleanup(self) -> None:
        if self.driver:
            await self.driver.close()
            self.driver = None
            self.initialized = False
    
    async def health_check(self) -> bool:
        try:
            async with self.driver.session() as session:
                await session.run("MATCH (n) RETURN count(n)")
            return True
        except:
            return False
    
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        """Store nodes and relationships."""
        async with self.driver.session() as session:
            # Create node
            result = await session.run(
                """
                CREATE (n:Entity $data)
                RETURN elementId(n) as id
                """,
                data={**data, "namespace": namespace},
            )
            
            record = await result.single()
            
            return StorageResult(
                success=True,
                id=record["id"],
                metadata={"namespace": namespace},
            )
    
    async def query(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query graph with Cypher."""
        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH (n:Entity {{namespace: $namespace}})
                WHERE n.content CONTAINS $query
                RETURN n
                LIMIT $limit
                """,
                namespace=namespace,
                query=query,
                limit=limit,
            )
            
            records = await result.data()
            return [record["n"] for record in records]


# Example 5: MinIO/S3 Storage (Large Files)
@StorageRegistry.register("minio")
class MinIOStorage(StorageBackend):
    """MinIO/S3 storage for large files."""
    
    backend_name = "minio"
    description = "MinIO/S3 object storage"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint")
        self.bucket = config.get("bucket")
        self.access_key = config.get("access_key")
        self.secret_key = config.get("secret_key")
        self.client = None
    
    async def initialize(self) -> None:
        import boto3
        
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        
        # Create bucket if not exists
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except:
            self.client.create_bucket(Bucket=self.bucket)
        
        self.initialized = True
        self.health_status = "healthy"
        self._log(f"Initialized MinIO storage (bucket: {self.bucket})")
    
    async def cleanup(self) -> None:
        self.client = None
        self.initialized = False
    
    async def health_check(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except:
            return False
    
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        import io
        import json
        
        # Store as JSON object
        key = f"{namespace}/{data.get('id', 'unknown')}"
        
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data),
        )
        
        return StorageResult(
            success=True,
            id=key,
            metadata={"bucket": self.bucket, "key": key},
        )
    
    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        import json
        
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=id,
            )
            
            return json.loads(response["Body"].read())
        except:
            return None
    
    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=id,
            )
            return True
        except:
            return False
